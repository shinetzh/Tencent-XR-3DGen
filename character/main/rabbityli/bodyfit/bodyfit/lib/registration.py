import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import  trimesh

import open3d as o3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from .smpl_w_scale import SMPL_with_scale
from .utils.rigid_body import exp_so3

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from .utils.lossfunc import compute_truncated_chamfer_distance
from .lndp_deformer  import Deformer_Mirror as LNDP_Deformer_Mirror
from .lndp_deformer  import Deformer as LNDP_Deformer

import torch.nn as nn
from scipy.spatial.transform import Rotation as R
import json
from easydict import EasyDict as edict
import pathlib






from .utils import util, lossfunc, rotation_converter
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import copy
from .index_config import *

def obtain_viz_mesh (vertices, faces, color = [0, 0.506, 0.6] ):
    smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    smplmesh.vertices = o3d.utility.Vector3dVector(vertices)
    smplmesh.paint_uniform_color(color)
    smplmesh.triangles = o3d.utility.Vector3iVector(faces)
    smplmesh.compute_vertex_normals()
    return smplmesh


def initilize_body_pose( param_len, bone_index, bone_angle ):

    _zeros =np.zeros( param_len )
    if bone_index and bone_angle:
        assert len(bone_index) == len(bone_angle)
        for ind, v in zip ( bone_index, bone_angle):
            _zeros [ ind ] = v
    return _zeros



def mute_param_with_idx (bone_params, mute_ids ):
    # shoulder fixed y
    device = bone_params.device
    muted_param = []
    for i in range(  bone_params.shape[1]) :
        if i in mute_ids :
            muted_param.append( torch.zeros([1, 1]).to(device))
        else :
            muted_param.append( bone_params[:, i:i+1])
    muted_param = torch.concatenate( muted_param, dim = 1 )
    return muted_param


def mirror_param ( bone_params, flip_dict, reuse_dict ):
    mirrored_params = []
    for i in range(  bone_params.shape[1]) :
        if i in flip_dict:
            mirrored_params.append(bone_params[0][flip_dict[i]] * -1)
        elif i in reuse_dict:
            mirrored_params.append(bone_params[0][reuse_dict[i]])
        else:
            mirrored_params.append(bone_params[0][i])

    mirrored_params = torch.stack( mirrored_params)[None]
    return mirrored_params



def filter_isolated_smpl_verts( smplverts,  faces):
    indexs = "/home/rabbityl/workspace/auto_rig/bodyfit/bodyfit/smplx_vert_segmentation.json"
    with open(indexs) as f:
        indexs = json.load(f)
        eyeballs = indexs["leftEye"] + indexs["rightEye"]
        body = list(np.arange(smplverts.shape[0]))
        body_wo_eye = [x for x in body if x not in eyeballs]
        # body_wo_eye = indexs["rightEye"]

    full_indices = torch.arange(smplverts.shape[0])
    body2full_ind = full_indices[body_wo_eye]
    full_mask = torch.zeros([smplverts.shape[0]])
    full_mask[body_wo_eye] = 1

    body_indices = torch.arange(len(body_wo_eye))
    full2body_ind = torch.zeros_like(full_indices) - 1
    full2body_ind[body_wo_eye] = body_indices

    # obtain body mesh without eye
    body_triangles = full2body_ind[faces]
    valid_triangle_mask = torch.logical_and(
        torch.logical_and(body_triangles[:, 0] > -1, body_triangles[:, 1] > -1),
        body_triangles[:, 2] > -1
    )
    body_triangles = body_triangles[valid_triangle_mask]
    body_triangles = body_triangles.to("cuda")
    body_verts = smplverts[body_wo_eye]

    # check unrefered verts
    refered_verts = torch.zeros_like(body_indices)
    refered_verts[body_triangles.reshape(-1)] = 1
    assert refered_verts.sum() == len(body_indices)

    return body_verts, body_triangles, full2body_ind, body2full_ind


class ParamNet(nn.Module):
    """direclty optimize parameters"""

    def __init__(self, size, init_way=None, device=None, last_op=None, scale=1., batch_size=1):
        super().__init__()
        self.param_size = size
        self.init_way = init_way
        self.device = device
        self.scale = scale
        self.last_op = last_op
        self.batch_size = batch_size

        init_parameter = init_way(size).float().to(self.device)
        self.register_parameter('param', torch.nn.Parameter(init_parameter))

    def forward(self): # x, y for placeholder
        param = self.last_op(self.param)*self.scale
        param = param.expand(self.batch_size, -1, -1)
        return param


class SMPLRegistration(torch.nn.Module):
    def __init__(self, config=None, rank=0,  mesh_trans = None, mesh = None):
        super(SMPLRegistration, self).__init__()

        self.cfg = config

        # pair
        self.mirror_pairs_x_dict = {}
        for pair in self.cfg.mirror_pairs_x:
            l, r = pair
            self.mirror_pairs_x_dict[r] = l
        self.mirror_pairs_yz_dict = {}
        for pair in self.cfg.mirror_pairs_yz:
            l, r = pair
            self.mirror_pairs_yz_dict[r] = l



        device = torch.device(rank)
        self.device = device

        self.batch_size = self.cfg.train.batch_size

        self.model = SMPL_with_scale(self.cfg).to(self.device)

        if self.cfg.loss.mesh_w_reg_edge > 0.:
            reg_verts = self.model.smplx.verts.cpu().numpy().squeeze()
            reg_faces = self.model.smplx.faces.cpu().numpy().squeeze()
            verts_per_edge = lossfunc.get_vertices_per_edge(len(reg_verts), reg_faces)
            self.verts_per_edge = torch.from_numpy(verts_per_edge).float().to(self.device).long()



        # parameters
        self.init_pose_np = initilize_body_pose( 21*3, self.cfg.init.bone_index, self.cfg.init.bone_angle )
        self.body_poses_opt = nn.Parameter(torch.tensor( self.init_pose_np, requires_grad=True )[None].float().to(self.device) )

        self.transl = nn.Parameter(torch.zeros([1, 3], requires_grad=True).float().to(self.device))

        self.betas_opt = nn.Parameter(torch.zeros([1, self.cfg.model.n_shape], requires_grad=True).float().to(self.device) )
        self.head_scale = nn.Parameter(torch.ones([1], requires_grad=True).float().to(self.device))
        self.body_scale = nn.Parameter(torch.ones([1], requires_grad=True).float().to(self.device))

        self.full_scale = nn.Parameter(torch.ones([1], requires_grad=True).float().to(self.device))



        r = R.from_euler('x', 90, degrees=True).as_rotvec()
        self.global_orient = torch.from_numpy(r).to(self.device).view(1, 3).float()


        # load target mesh

        if mesh is not  None:
            self.mesh = mesh

        else :
            print("config.mesh_path", config.mesh_path)
            self.mesh = o3d.io.read_triangle_mesh( config.mesh_path)
            self.mesh.compute_vertex_normals()
            self.mesh.paint_uniform_color([0.5, 0.5, 0])
        # else :

        if mesh_trans is not None:
            mesh2 = copy.deepcopy( self.mesh )
            mesh2.paint_uniform_color([0.0, 0.5, 0.8])
            self.mesh.transform( mesh_trans)


            o3d.visualization.draw_geometries( [self.mesh, mesh2] )





        self.smplx_offset_dump_path = str( Path( config.mesh_path ).parent / "smplx_and_offset.npz" )
        # self.mesh.compute_vertex_normals()
        # self.mesh.paint_uniform_color([0.5, 0.5, 0])

        # o3d.visualization.draw_geometries( [ self.mesh ] )

        self.n_sample = 1640
        # self.pcd_t = torch.from_numpy(np.asarray(self.mesh.sample_points_uniformly(number_of_points=200000).points))[
        #     None, ...].to(self.device).float()
        self.mesh_target_py3d =  Meshes(
            verts=[ torch.from_numpy( np.asarray(self.mesh.vertices)).float() ],
            faces=[ torch.from_numpy( np.asarray(self.mesh.triangles)) ] ).to(self.device)
        self.mesh_target_py3d._compute_vertex_normals()
        self.pcd_t, self.pcd_nml_t = sample_points_from_meshes(self.mesh_target_py3d, 200000, return_normals=True)

        if config.matches:
            mfiles = glob.glob( os.path.join( config.matches, "*.json") )
            character_pts = []
            smpl_baryc_coords = []
            smpl_verts_id = []
            matches = {}
            for m in mfiles :
                with open(m) as f:
                    matches = json.load(f)
                    character_pts.append( torch.from_numpy(np.asarray(matches["character_pts"])).to(self.device) )
                    smpl_baryc_coords.append( torch.from_numpy(np.asarray(matches["smpl_baryc_coords"])).to(self.device) )
                    smpl_verts_id.append( torch.from_numpy(np.asarray(matches["smpl_verts_id"])).to(self.device) )
            matches["character_pts"], matches["smpl_baryc_coords"], matches[ "smpl_verts_id" ] = \
                [torch.concatenate( X , dim= 0 ) for X in [character_pts, smpl_baryc_coords, smpl_verts_id] ]
        else:
            matches = None
        self.matches = matches


        if config.curve_matches:
            with open(config.curve_matches) as f:
                curve_matches = json.load(f)
                curve_matches["character_pts"] = torch.from_numpy(np.asarray(curve_matches["character_pts"])).to(self.device)
                curve_matches["smpl_baryc_coords"] = torch.from_numpy(np.asarray(curve_matches["smpl_baryc_coords"])).to(self.device)
                curve_matches["smpl_verts_id"] = torch.from_numpy(np.asarray(curve_matches["smpl_verts_id"])).to(self.device)
        else:
            curve_matches = None
        self.curve_matches = curve_matches

    def align_vroid(self, viz=True):




        #step 1. optimize translation and scale
        optimizer_1 = torch.optim.Adam( [self.transl, self.body_scale, self.head_scale], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        # optimizer_1 = torch.optim.Adam( [self.transl,  self.betas_opt], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        loss_config = self.cfg.scale_trn
        _ = self.optimize_smplx(
            optimizer_1,
            steps=loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            mute_trans_id=[0],
            viz=viz
        )


        print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)
        print("self.transl ", self.transl)


        # step 2. optimize translation , body_pose and betas
        optimizer_2 = torch.optim.Adam( [self.transl,  self.body_poses_opt, self.betas_opt], lr=self.cfg.pose_trn_betas.e_lr, weight_decay=self.cfg.pose_trn_betas.e_wd)
        loss_config = self.cfg.pose_trn_betas
        data = self.optimize_smplx(
            optimizer_2,
            steps=loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            mute_trans_id=[0],
            viz=viz
        )



        print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)
        print( "self.transl ", self.transl )


        # offset = self.ndp_align()

        offset, warpped_verts, warpped_mesh  = self.ndp_align_mirror( viz=viz )



        data[ "offset" ] = offset
        # torch.save( data , self.smplx_offset_dump_path )

        return data, warpped_mesh


    def align_daz(self, viz=False):

        #step 1. optimize translation and scale
        # optimizer_1 = torch.optim.Adam( [self.transl], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        optimizer_1 = torch.optim.Adam( [self.transl, self.full_scale], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )

        # optimizer_1 = torch.optim.Adam( [self.transl,  self.betas_opt], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        loss_config = self.cfg.scale_trn
        _ = self.optimize_smplx(
            optimizer_1,
            steps=loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            viz=viz,
            full_scale = True
        )

        print( "self.full_scale", self.full_scale)

        # step 2. optimize translation , body_pose and betas
        optimizer_2 = torch.optim.Adam( [self.transl,  self.body_poses_opt, self.betas_opt], lr=self.cfg.pose_trn_betas.e_lr, weight_decay=self.cfg.pose_trn_betas.e_wd)
        loss_config = self.cfg.pose_trn_betas
        data = self.optimize_smplx(
            optimizer_2,
            steps=loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            viz=viz,
            full_scale=True

        )


        offset, warpped_verts, warpped_mesh  = self.ndp_align_mirror( viz=viz )

        data[ "offset" ] = offset

        return data, warpped_mesh



    def align_cartoon(self, viz=True):

        debug = False

        #step 1. optimize translation and scale
        optimizer_1 = torch.optim.Adam( [self.transl, self.body_poses_opt, self.body_scale, self.head_scale], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        loss_config = self.cfg.scale_trn
        _ = self.optimize_smplx(
            optimizer_1,
            steps=1 if debug else loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            viz=viz
        )


        print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)


        if not debug:

            # step 2. optimize translation , body_pose and betas
            optimizer_2 = torch.optim.Adam( [self.transl,  self.body_scale, self.head_scale, self.body_poses_opt, self.betas_opt], lr=self.cfg.pose_trn_betas.e_lr, weight_decay=self.cfg.pose_trn_betas.e_wd)
            loss_config = self.cfg.pose_trn_betas
            data = self.optimize_smplx(
                optimizer_2,
                steps=loss_config.step,
                loss_config=loss_config,
                mute_body_id=self.cfg.mute_id,
                viz=viz
            )


            print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)




        # offset, warpped_verts, warpped_mesh  = self.ndp_align_mirror()

        offset, warpped_verts, warpped_mesh = self.jacobian_align()

        # offset, warpped_verts, warpped_mesh = self.param_align_new()





        data[ "offset" ] = offset
        return data, warpped_mesh

        # return None, None



    def align_yuanmeng(self, viz=True):

        debug = False

        #step 1. optimize translation and scale
        optimizer_1 = torch.optim.Adam( [self.transl, self.body_poses_opt], lr=self.cfg.scale_trn.e_lr, weight_decay=self.cfg.scale_trn.e_wd )
        loss_config = self.cfg.scale_trn
        _ = self.optimize_smplx(
            optimizer_1,
            steps=1 if debug else loss_config.step,
            loss_config=loss_config,
            mute_body_id=self.cfg.mute_id,
            viz=viz
        )


        print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)


        if not debug:

            # step 2. optimize translation , body_pose and betas
            optimizer_2 = torch.optim.Adam( [self.transl,    self.body_poses_opt, self.betas_opt], lr=self.cfg.pose_trn_betas.e_lr, weight_decay=self.cfg.pose_trn_betas.e_wd)
            loss_config = self.cfg.pose_trn_betas
            data = self.optimize_smplx(
                optimizer_2,
                steps=loss_config.step,
                loss_config=loss_config,
                mute_body_id=self.cfg.mute_id,
                viz=viz
            )



            print( "self.body_scale, self.head_scale", self.body_scale, self.head_scale)




        offset1, warpped_verts, warpped_mesh  = self.ndp_align_mirror(viz=viz)



        # offset2, warpped_verts, warpped_mesh = self.jacobian_align( self.smplverts + offset1, self.matches , viz=viz)


        offset3, warpped_verts, warpped_mesh = self.jacobian_align( self.smplverts + offset1, self.curve_matches ,viz=viz)
        # offset3, warpped_verts, warpped_mesh = self.jacobian_align( self.smplverts + offset1, self.matches ,viz=viz)


        # offset, warpped_verts, warpped_mesh = self.param_align_new()





        data[ "offset" ] = offset1 + offset3
        return data, warpped_mesh

        # return None, None





    def optimize_smplx(self,
                       optimizer,
                       steps,
                       loss_config,
                       mute_body_id = None,
                       mute_trans_id = None,
                       log_freq=100,
                       viz=True,
                       full_scale = False):



        # non-rigid fitting of body pose, shape parameters.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.999)


        for k in range(steps):

            body_pose_params = mute_param_with_idx(self.body_poses_opt, mute_ids=mute_body_id )

            if mute_trans_id:
                transl = mute_param_with_idx(self.transl, mute_ids=mute_trans_id)
            else:
                transl = self.transl


            body_pose_params = mirror_param( body_pose_params, self.mirror_pairs_yz_dict, self.mirror_pairs_x_dict)


            if full_scale :
                head_s = self.full_scale
                body_s = self.full_scale
            else :
                head_s = self.head_scale
                body_s = self.body_scale

            head_s = 3.2
            body_s = 1.1

            data = { "body_pose" : body_pose_params,
                     "betas": self.betas_opt,
                     "global_pose": self.global_orient,
                     "transl": transl,
                     "head_scale": head_s,
                     "body_scale": body_s
                     }


            faces, vertices = self.model.forward ( data )

            if k ==0 and viz:
                smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                smplmesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
                smplmesh.paint_uniform_color([0, 0.506, 0.6])
                smplmesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
                smplmesh.compute_vertex_normals()
                self.faces = faces.detach().cpu()  # .numpy()
                o3d.visualization.draw([smplmesh, self.mesh])

            losses = self.compute_losses(
                vertices,
                faces,
                w_chamfer=loss_config.w_chamfer,
                w_ldmk = loss_config.w_ldmk,
                w_shape_reg= loss_config.w_shape_reg,
                n_sample = loss_config.n_sample,
                w_p2plane = loss_config.w_p2plane
            )



            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['l-all'] = all_loss
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()


            # self.summary_writer.add_scalar(f'LearningRate', self.get_lr(self.e_opt), k )


            if (k+1) % log_freq == 0 :
                log="step-" + str(k) + ": "
                for k, v in losses.items():
                    log = log +  k + f"={losses[k]:.4f}, "
                print( log )


        smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        smplmesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        smplmesh.paint_uniform_color([0, 0.506, 0.6])
        smplmesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
        smplmesh.compute_vertex_normals()

        self.faces = faces.detach().cpu() #.numpy()





        if viz:
            o3d.visualization.draw( [smplmesh,   self.mesh])



        self.smplmesh = smplmesh
        self.smplverts = vertices.detach()
        # self.data = data


        return data





    def compute_losses(self,
                       vertices,
                       faces,
                       w_chamfer = 0,
                       w_ldmk = 0,
                       w_shape_reg = 0,
                       w_p2plane = 0,
                       n_sample = 1640):

        # sample points on target mesh
        losses = {}

        # chamfer distance
        if w_chamfer > 0:
            perm = torch.randperm(self.pcd_t.shape[1])
            tgt_pnts = self.pcd_t[:, perm[:n_sample]]
            # sample on SMPL mesh
            face_mesh = Meshes(verts=[vertices], faces=[faces])
            face_pnts = sample_points_from_meshes(face_mesh, n_sample )
            _, _, cd_tgt = compute_truncated_chamfer_distance(face_pnts, tgt_pnts, point_reduction="mean", trunc=10)
            chamfer_dist = cd_tgt * w_chamfer
            losses['l-cd'] = chamfer_dist


            # p2plane_loss, _, _, normal_reference = lossfunc.point_2_plane_distance( sample_pred, sample_target, x_normals=normal_pred, y_normals=normal_tgt )
        else :
            losses['l-cd'] = 0


        #p2plane distance
        if w_p2plane > 0:
            pred_mesh = Meshes(verts=[vertices], faces=[faces])
            pred_mesh._compute_vertex_normals()
            sample_pred, normal_pred = sample_points_from_meshes(pred_mesh, n_sample, return_normals=True)

            perm = torch.randperm(self.pcd_t.shape[1])
            tgt_pnts = self.pcd_t[:, perm[:n_sample]]
            tgt_nrml = self.pcd_nml_t[:, perm[:n_sample]]

            p2plane_loss, _, _, normal_reference = \
                lossfunc.point_2_plane_distance(sample_pred, tgt_pnts, x_normals=normal_pred, y_normals=tgt_nrml)

            losses['l-p2plane'] =  p2plane_loss * w_p2plane

        else :
            losses['l-p2plane'] = 0



        # landmark loss
        if w_ldmk > 0 and self.matches:
            character_ldmks = self.matches["character_pts"]
            smpl_ldmk_coords = self.matches["smpl_baryc_coords"]
            smpl_ldmk_id = self.matches["smpl_verts_id"]
            smpl_ldmk_verts = vertices[smpl_ldmk_id.view(-1)]
            smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
            smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
            ldmk_loss =w_ldmk * torch.norm(smpl_ldmks - character_ldmks, dim=-1).mean()
            losses["l-ldmk"] = ldmk_loss
        else:
            losses["l-ldmk"] = 0


        #shape regularization
        if w_shape_reg > 0 :
            shape_reg = w_shape_reg * torch.sum(self.betas_opt ** 2) / 2
            losses["l-reg"] = shape_reg
        else :
            losses["l-reg"] = 0


        return losses


    def ndp_align(self, viz=True):


        config = {
            "gpu_mode": True,
            "iters": 1000,
            "lr": 0.001,
            "max_break_count": 50,
            "break_threshold_ratio": 0.0002,
            "samples": 20000,
            "motion_type": "sflow",
            "rotation_format": "euler",
            "m": 8,
            "k0": -4,
            "depth": 3,
            "width": 128,
            "act_fn": "relu",
            "w_reg": 0,
            "w_ldmk": 1,
            "w_cd": 1000,
            'trunc_cd':100
        }



        config = edict(config)
        config.device = self.device
        deformer = LNDP_Deformer( config )


        if self.matches:
            character_ldmks = self.matches["character_pts"]
            smpl_ldmk_coords = self.matches["smpl_baryc_coords"]
            smpl_ldmk_id = self.matches["smpl_verts_id"]

            smpl_ldmk_verts = self.smplverts [ smpl_ldmk_id.view(-1) ]
            smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
            smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
            landmarks = [smpl_ldmks.float() , character_ldmks.float()]
        else :
            landmarks = None


        src_pcd =   np.asarray( self.smplmesh.sample_points_uniformly(number_of_points=config.samples).points, dtype=np.float32)
        tgt_pcd =  np.asarray( self.mesh.sample_points_uniformly(number_of_points=config.samples).points, dtype=np.float32)

        deformer.train_field( src_pcd, tgt_pcd, landmarks, cancel_translation=False )

        warpped_verts = deformer.warp_points( self.smplverts )

        # print( "warpped_verts.device", warpped_verts.device)
        # print(  "self.smplverts.device", self.smplverts.device)

        offset = warpped_verts - self.smplverts


        # cano_vertices = self.model._2_canonical_verts( self.data, warpped_verts )


        self.smplmesh.vertices= o3d.utility.Vector3dVector(warpped_verts.detach().cpu().numpy())
        self.smplmesh.paint_uniform_color([0, 0.506, 0.6])
        self.smplmesh.compute_vertex_normals()


        # smplmesh_canonical = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        # smplmesh_canonical.vertices = o3d.utility.Vector3dVector(cano_vertices.detach().cpu().numpy())
        # smplmesh_canonical.paint_uniform_color([1, 0.506, 0.6])
        # smplmesh_canonical.triangles = o3d.utility.Vector3iVector(np.asarray(self.smplmesh.triangles))
        # smplmesh_canonical.compute_vertex_normals()



        if viz:
            o3d.visualization.draw( [self.smplmesh,  self.mesh])


        return offset


    def ndp_align_mirror(self, viz=True):


        # config = {
        #     "gpu_mode": True,
        #     "iters": 800,
        #     "lr": 0.001,
        #     "max_break_count": 50,
        #     "break_threshold_ratio": 0.0002,
        #     "samples": 6000,
        #     "motion_type": "sflow",
        #     "rotation_format": "euler",
        #     "m": 8,
        #     "k0": -4,
        #     "depth": 3,
        #     "width": 128,
        #     "act_fn": "relu",
        #     "w_reg": 0,
        #     "w_ldmk": 1,
        #     "w_cd": 1000,
        #     'trunc_cd':100
        # }

        config = {
            "gpu_mode": True,
            "iters": 800,
            "lr": 0.001,
            "max_break_count": 50,
            "break_threshold_ratio": 0.0002,
            "samples": 6000,
            "motion_type": "sflow",
            "rotation_format": "euler",
            "m": 8,
            "k0": -4,
            "depth": 3,
            "width": 128,
            "act_fn": "relu",
            "w_reg": 0,
            "w_ldmk": 10,
            "w_cd": 100,
            'trunc_cd': 100
        }



        config = edict(config)
        config.device = self.device
        deformer = LNDP_Deformer_Mirror( config )


        if self.matches:
        # if False:
            character_ldmks = self.matches["character_pts"]
            smpl_ldmk_coords = self.matches["smpl_baryc_coords"]
            smpl_ldmk_id = self.matches["smpl_verts_id"]

            smpl_ldmk_verts = self.smplverts [ smpl_ldmk_id.view(-1) ]
            smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
            smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
            landmarks = [smpl_ldmks.float() , character_ldmks.float()]

        else :
            landmarks = None


        src_pcd = np.asarray( self.smplmesh.sample_points_uniformly(number_of_points=config.samples).points, dtype=np.float32)
        src_pcd_flip = src_pcd.copy()
        src_pcd_flip[:,0] = src_pcd_flip[:,0] * -1
        src_pcd_mirror = np.concatenate ( [ src_pcd, src_pcd_flip], axis=0 )

        tgt_pcd = np.asarray( self.mesh.sample_points_uniformly(number_of_points=config.samples  ).points, dtype=np.float32)
        tgt_pcd_flip = tgt_pcd.copy()
        tgt_pcd_flip[:,0] = tgt_pcd_flip[:,0] * -1
        tgt_pcd_mirror = np.concatenate ( [ tgt_pcd, tgt_pcd_flip], axis=0 )



        deformer.train_field( src_pcd_mirror, tgt_pcd_mirror, landmarks, cancel_translation=False )


        warpped_verts = deformer.warp_points(self.smplverts)


        print( "warpped_verts.device", warpped_verts.device)
        print(  "self.smplverts.device", self.smplverts.device)

        offset = warpped_verts - self.smplverts



        smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        smplmesh.vertices = o3d.utility.Vector3dVector(warpped_verts.detach().cpu().numpy())
        smplmesh.paint_uniform_color([0, 0.506, 0.6])
        smplmesh.triangles = o3d.utility.Vector3iVector(self.faces)
        smplmesh.compute_vertex_normals()




        if viz:
            o3d.visualization.draw( [smplmesh, self.mesh] )


        # pc = o3d.geometry.PointCloud()
        # vertices = warpped_verts.detach().cpu().numpy()
        # pc.points = o3d.utility.Vector3dVector(vertices)
        # pc.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([pc])

        return offset, warpped_verts, smplmesh


    def jacobian_align (self,  pre_deform_mesh , the_matches, viz=True):

        from .NeuralJacobianFields import SourceMesh

        # Left-Right symmetry
        # Better correspondence with line selection

        config = {
            "gpu_mode": True,
            "lr": 0.002,
            "epochs": 2000,
            #loss weights
            "cd_weight": 0,
            "p2plane": 20,
            "ldmk_weight": 1000.0,
            # "ldmk_weight": 0.0,
            "edge_weight": 0.1,
            "offset_reg": 1,
            "laplacian": 10,
            "normal": 5
        }

        # config = {
        #     "gpu_mode": True,
        #     "lr": 0.05,
        #     "epochs": 2000,
        #     #loss weights
        #     "cd_weight": 0,
        #     "p2plane": 0.01,
        #     "ldmk_weight": 10.0,
        #     "edge_weight": 0,
        #     "offset_reg": 0,
        #     "laplacian": 0,
        #     "normal": 0
        # }

        cfg = edict(config)


        # setup mesh for jacobian
        body_verts, body_triangles, full2body_ind, body2full_ind = filter_isolated_smpl_verts( pre_deform_mesh, self.faces)
        import time
        ts = time.time()
        tmp_dir = os.path.join( "/home/rabbityl/tboard/jacobian_tmp", str(ts))
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        jacobian_source = SourceMesh.SourceMesh(0, tmp_dir , {}, 1, ttype=torch.float)
        jacobian_source.load(body_verts.detach().cpu().numpy(), body_triangles.detach().cpu().numpy())
        jacobian_source.to(self.device)


        if the_matches:
            character_ldmks = the_matches["character_pts"]
            smpl_ldmk_coords = the_matches["smpl_baryc_coords"]
            smpl_ldmk_id = the_matches["smpl_verts_id"]

            ldmk_triangles = full2body_ind.to("cuda")[smpl_ldmk_id]
            valid_ldmk_mask = torch.logical_and(
                torch.logical_and(ldmk_triangles[:, 0] > -1, ldmk_triangles[:, 1] > -1),
                ldmk_triangles[:, 2] > -1
            )

            character_ldmks = character_ldmks [valid_ldmk_mask]
            smpl_ldmk_coords = smpl_ldmk_coords [valid_ldmk_mask]
            smpl_ldmk_id = smpl_ldmk_id [valid_ldmk_mask]

            def compute_smpl_landmark( deformed_verts):
                smpl_ldmk_verts = deformed_verts [smpl_ldmk_id.view(-1)]
                smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
                smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
                return smpl_ldmks




        with torch.no_grad():
            gt_jacobians = jacobian_source.jacobians_from_vertices(body_verts.detach().unsqueeze(0))
        gt_jacobians.requires_grad_(True)

        #
        # ### rotate verts
        #
        # from scipy.spatial.transform import Rotation as R
        # rot_z_30 = torch.from_numpy(R.from_euler('z', 30, degrees=True).as_matrix() ).to(body_verts)
        # print (rot_z_30)
        # body_verts_rotated = (rot_z_30 @ body_verts.T ).T
        # gt_jacobians_2 = jacobian_source.jacobians_from_vertices( body_verts_rotated.unsqueeze(0) )
        #



        optimizer = torch.optim.Adam([gt_jacobians], lr=cfg.lr)
        t_loop = tqdm(range(cfg.epochs), leave=False)

        target_mesh = Meshes(verts=[torch.from_numpy(np.array(self.mesh.vertices)).float().to("cuda")],  faces=[torch.from_numpy(np.array(self.mesh.triangles)).to("cuda")])

        # viz = False
        if viz:

            self.mesh.translate((1.5,0,0))

            character_pc = o3d.geometry.PointCloud()
            character_pc.points = o3d.utility.Vector3dVector( character_ldmks.cpu().numpy() )
            character_pc.paint_uniform_color([0, 1, 0])

            tmp_points = np.zeros_like( character_ldmks.cpu().numpy() )
            smpl_pc = o3d.geometry.PointCloud()
            smpl_pc.points = o3d.utility.Vector3dVector( tmp_points )
            smpl_pc.paint_uniform_color([1, 0, 0])
            viz_mesh_src = obtain_viz_mesh(body_verts.detach().cpu().numpy(), body_triangles.detach().cpu().numpy())

            # o3d.visualization.draw([smpl_pc, viz_mesh_src, character_pc, self.mesh])

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(smpl_pc)
            vis.add_geometry(character_pc)
            vis.add_geometry(self.mesh)
            vis.add_geometry(viz_mesh_src)




        offset = 0
        for it in t_loop:

            # updated vertices from jacobians
            deformed_verts = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()
            if it == 0 :
                offset = (deformed_verts - body_verts).detach()

                # if viz:
                #     viz_mesh_src_copy = copy.deepcopy(viz_mesh_src)
                #     viz_mesh_src_copy.vertices = o3d.utility.Vector3dVector((deformed_verts ).detach().cpu().numpy())
                #     viz_mesh_src_copy.paint_uniform_color([.9, .9, .9])
                #     vis.add_geometry(viz_mesh_src_copy)

            deformed_verts = deformed_verts - offset
            optimizer.zero_grad()
            pred_mesh = Meshes(verts=[deformed_verts], faces=[body_triangles])


            #sample vertices and normal
            pred_mesh._compute_vertex_normals()
            select_indices = torch.randperm(deformed_verts.shape[0])[:3000]
            sample_vert = deformed_verts [ select_indices ]
            sample_vert_normal = pred_mesh._verts_normals_packed [ select_indices ]
            #sample random points and normal
            sample_target, normal_tgt = sample_points_from_meshes(target_mesh, 6000, return_normals=True)
            sample_pred, normal_pred = sample_points_from_meshes(pred_mesh, 3000, return_normals=True)
            sample_pred = torch.cat( [sample_pred, sample_vert[None]], dim=1)
            normal_pred = torch.cat( [normal_pred, sample_vert_normal[None]], dim=1)

            losses = {}

            #### chamfer distance loss
            # cd_loss, _ = chamfer_distance(sample_pred, sample_target, single_directional=False)
            # losses ["chamfer"] = cd_loss * cfg.cd_weight

            #### point to plane loss
            p2plane_loss, _, _, normal_reference = lossfunc.point_2_plane_distance( sample_pred, sample_target, x_normals=normal_pred, y_normals=normal_tgt )
            # _, p2plane_loss, _, normal_reference = lossfunc.symmetric_point_2_plane_distance( sample_pred, sample_target, x_normals=normal_pred, y_normals=normal_tgt )
            losses["p2plane"] = p2plane_loss * cfg.p2plane

            ##### normal loss
            losses ["normal"] = torch.abs(  normal_pred[0] - normal_reference ).sum(dim=1).mean() * cfg.normal

            ##### laplacian loss
            lp_loss = mesh_laplacian_smoothing(pred_mesh, method="uniform")
            losses ["laplacian"] = lp_loss * cfg.laplacian

            ### trig loss
            triag_loss = mesh_edge_loss(pred_mesh)
            losses["edge"] = triag_loss * cfg.edge_weight

            ### landmark loss
            if the_matches:
                warped_ldmk = compute_smpl_landmark( deformed_verts )
                losses ["ldmk"] = torch.mean(torch.sum((warped_ldmk - character_ldmks) ** 2, dim=-1)) * cfg.ldmk_weight
            else :
                losses ["ldmk"] = 0


            total_loss  = 0
            for k, v in losses.items():
                total_loss = total_loss + v

            # total_loss = losses ["laplacian"] +\
            #              losses ["ldmk"] +\
            #              losses ["edge"] + \
            #              losses["p2plane"] +\
            #              losses ["normal"]


            total_loss.backward()
            optimizer.step()
            desc =f'Total={total_loss.item():.8f}|--| '
            for k, v  in losses.items() :
                 desc = desc + k + f'={v.item():.8f}| '
            t_loop.set_description( desc)



            if viz:

                viz_mesh_src.vertices = o3d.utility.Vector3dVector( deformed_verts.detach().cpu().numpy() )
                viz_mesh_src.compute_vertex_normals()
                vis.update_geometry(viz_mesh_src)

                smpl_pc.points = o3d.utility.Vector3dVector( warped_ldmk.detach().cpu().numpy() )
                vis.update_geometry(smpl_pc)

                vis.poll_events()
                vis.update_renderer()



        print( "msg:", desc)


        # o3d.io.write_triangle_mesh( "/home/rabbityl/workspace/auto_rig/bodyfit/reg_compare/jacobian.ply", viz_mesh_src)

        offset = torch.zeros_like( pre_deform_mesh)
        offset [ body2full_ind ] = deformed_verts - pre_deform_mesh  [ body2full_ind ]

        warpped_verts = pre_deform_mesh + offset

        warpped_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        warpped_mesh.vertices = o3d.utility.Vector3dVector(warpped_verts.detach().cpu().numpy())
        warpped_mesh.paint_uniform_color([0, 0.506, 0.6])
        warpped_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        warpped_mesh.compute_vertex_normals()


        if viz:
            o3d.visualization.draw([smpl_pc,viz_mesh_src,character_pc, self.mesh])


        # offset = deformed_verts - self.smplverts
        return offset, warpped_verts, warpped_mesh



    def param_align_new (self, viz=True):

        def arap_cost(R, t, g, e, w):
            '''
            :param R: vertex rotation [n,3,3]
            :param t: vertex translation [n,3]
            :param g: vertex raw position [n,3]
            :param e: graph edge [2, m]
            :param w: graph edge weight
            :return:
            '''

            i, j = e[0], e[1]
            R_i = R[i]
            g_i = g[i]
            t_i = t[i]

            g_j = g[j]
            t_j = t[j]

            e_ij = ( (R_i  @ (g_j - g_i)[..., None] ).squeeze() + g_i + t_i - g_j - t_j )**2 * w
            o = e_ij.sum(dim=-1).mean()

            return o

        # Left-Right symmetry
        # Better correspondence with line selection

        config = {
            "gpu_mode": True,
            "lr": 0.02,
            "epochs": 5000,
            "geo_out_dim": 3,

            #loss weights
            "cd_weight": 0,
            "p2plane": 0,
            "ldmk_weight": 50.0,
            "edge_weight": 0,
            "offset_reg": 0,
            "laplacian": 0,
            "normal": 0,
            "arap": 1000
        }

        cfg = edict(config)

        # setup mesh
        body_verts, body_triangles, full2body_ind, body2full_ind = filter_isolated_smpl_verts( self.smplverts, self.faces)


        t = torch.zeros_like(body_verts)
        t = torch.nn.Parameter(t, requires_grad=True)

        phi = torch.zeros_like(body_verts)
        phi = torch.nn.Parameter(phi, requires_grad=True)


        if self.matches:
            character_ldmks = self.matches["character_pts"]
            smpl_ldmk_coords = self.matches["smpl_baryc_coords"]
            smpl_ldmk_id = self.matches["smpl_verts_id"]

            ldmk_triangles = full2body_ind.to("cuda")[smpl_ldmk_id]
            valid_ldmk_mask = torch.logical_and(
                torch.logical_and(ldmk_triangles[:, 0] > -1, ldmk_triangles[:, 1] > -1),
                ldmk_triangles[:, 2] > -1
            )

            character_ldmks = character_ldmks [valid_ldmk_mask][:2]
            smpl_ldmk_coords = smpl_ldmk_coords [valid_ldmk_mask][:2]
            smpl_ldmk_id = smpl_ldmk_id [valid_ldmk_mask][:2]

            def compute_smpl_landmark( deformed_verts):
                smpl_ldmk_verts = deformed_verts [smpl_ldmk_id.view(-1)]
                smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
                smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
                return smpl_ldmks



        params = [ phi, t ]
        # params = [{ 'params': geo_model.parameters() }  ]
        optimizer = torch.optim.Adam(params=params, lr=cfg.lr)

        t_loop = tqdm(range(cfg.epochs), leave=False)

        target_mesh = Meshes(verts=[torch.from_numpy(np.array(self.mesh.vertices)).float().to("cuda")],  faces=[torch.from_numpy(np.array(self.mesh.triangles)).to("cuda")])

        if viz:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            self.mesh.translate((1.5,0,0))

            character_pc = o3d.geometry.PointCloud()
            character_pc.points = o3d.utility.Vector3dVector( character_ldmks.cpu().numpy() )
            character_pc.paint_uniform_color([0, 1, 0])
            vis.add_geometry(character_pc)

            tmp_points = np.zeros_like( character_ldmks.cpu().numpy() )
            smpl_pc = o3d.geometry.PointCloud()
            smpl_pc.points = o3d.utility.Vector3dVector( tmp_points )
            smpl_pc.paint_uniform_color([1, 0, 0])
            vis.add_geometry(smpl_pc)

            viz_mesh_src = obtain_viz_mesh(body_verts.detach().cpu().numpy(), body_triangles.detach().cpu().numpy())
            vis.add_geometry(self.mesh)
            vis.add_geometry(viz_mesh_src)

            # viz_mesh_src_cpy = copy.deepcopy(viz_mesh_src)
            # smpl_pc.paint_uniform_color([0.7, .1, .50])
            # vis.add_geometry(viz_mesh_src_cpy)



        ### construct graph edge from triangles
        e_i = torch.cat( [ body_triangles[:,0] , body_triangles[:,1] , body_triangles[:,2 ]], dim=0)
        e_j = torch.cat( [ body_triangles[:,1] , body_triangles[:,2] , body_triangles[:,0 ]], dim=0)
        edge = torch.stack( [e_i, e_j] )




        for it in t_loop:


            deformed_verts = body_verts + t


            optimizer.zero_grad()

            pred_mesh = Meshes(verts=[deformed_verts], faces=[body_triangles])


            #sample vertices and normal
            pred_mesh._compute_vertex_normals()
            select_indices = torch.randperm(deformed_verts.shape[0])[:3000]
            sample_vert = deformed_verts [ select_indices ]
            sample_vert_normal = pred_mesh._verts_normals_packed [ select_indices ]
            #sample random points and normal
            sample_target, normal_tgt = sample_points_from_meshes(target_mesh, 6000, return_normals=True)
            sample_pred, normal_pred = sample_points_from_meshes(pred_mesh, 3000, return_normals=True)
            sample_pred = torch.cat( [sample_pred, sample_vert[None]], dim=1)
            normal_pred = torch.cat( [normal_pred, sample_vert_normal[None]], dim=1)

            losses = {}

            #### chamfer distance loss
            # cd_loss, _ = chamfer_distance(sample_pred, sample_target, single_directional=False)
            # losses ["chamfer"] = cd_loss * cfg.cd_weight

            #### point to plane loss
            p2plane_loss, _, _, normal_reference = lossfunc.point_2_plane_distance( sample_pred, sample_target, x_normals=normal_pred, y_normals=normal_tgt )
            # _, p2plane_loss, _, normal_reference = lossfunc.symmetric_point_2_plane_distance( sample_pred, sample_target, x_normals=normal_pred, y_normals=normal_tgt )
            losses["p2plane"] = p2plane_loss * cfg.p2plane

            ##### normal loss
            losses ["normal"] = torch.abs(  normal_pred[0] - normal_reference ).sum(dim=1).mean() * cfg.normal

            ##### laplacian loss
            lp_loss = mesh_laplacian_smoothing(pred_mesh, method="uniform")
            losses ["laplacian"] = lp_loss * cfg.laplacian

            ### trig loss
            triag_loss = mesh_edge_loss(pred_mesh)
            losses["edge"] = triag_loss * cfg.edge_weight

            ### offset regularization
            losses ["offset_reg"] = (t**2).sum(-1).mean() * cfg.offset_reg

            ## arap cost
            R = pytorch3d.transforms.axis_angle_to_matrix( phi )
            losses ["arap"] = arap_cost( R, t, body_verts, edge, 1) * cfg.arap


            ## landmark loss
            if self.matches:
                warped_ldmk = compute_smpl_landmark( deformed_verts )

                losses ["ldmk"] = torch.mean(torch.sum((warped_ldmk - character_ldmks) ** 2, dim=-1))
            else :
                losses ["ldmk"] = 0


            total_loss  = 0
            for k, v in losses.items():
                total_loss = total_loss + v

            total_loss.backward()
            optimizer.step()
            desc =f'Total={total_loss.item():.5f}|--| '
            for k, v  in losses.items() :
                 desc = desc + k + f'={v.item():.5f}| '
            t_loop.set_description( desc)



            if viz:

                viz_mesh_src.vertices = o3d.utility.Vector3dVector( deformed_verts.detach().cpu().numpy() )
                viz_mesh_src.compute_vertex_normals()
                vis.update_geometry(viz_mesh_src)

                # smpl_pc.points = o3d.utility.Vector3dVector( warped_ldmk.detach().cpu().numpy() )
                # vis.update_geometry(smpl_pc)

                vis.poll_events()
                vis.update_renderer()



        print( "msg:", desc)


        # o3d.io.write_triangle_mesh( "/home/rabbityl/workspace/auto_rig/bodyfit/reg_compare/jacobian.ply", viz_mesh_src)

        offset = torch.zeros_like( self.smplverts)
        offset [ body2full_ind ] = deformed_verts - self.smplverts  [ body2full_ind ]

        warpped_verts = self.smplverts + offset

        warpped_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
        warpped_mesh.vertices = o3d.utility.Vector3dVector(warpped_verts.detach().cpu().numpy())
        warpped_mesh.paint_uniform_color([0, 0.506, 0.6])
        warpped_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        warpped_mesh.compute_vertex_normals()

        # offset = deformed_verts - self.smplverts

        return offset, warpped_verts, warpped_mesh



    def param_align (self, viz=True):

        config = {
            "gpu_mode": True,
            "iters": 1000,
            "lr": 0.002,
            "epochs": 2000,
            "lp_weight": 0.5,
            "geo_out_dim": 3,
            "geo_scale": 1,



            "cd_weight": 10,
            "ldmk_weight": 10.0,
            "edge_weight" : 0,
            "offset_reg" : 1,
            "laplacian" : 1
        }

        cfg = edict(config)

        geo_model = ParamNet(size=(self.smplverts.shape[0], cfg.geo_out_dim), init_way=torch.zeros, last_op=torch.tanh, scale=1)
        geo_model.to(self.device)

        # self.matches = None
        if self.matches:
        # if False:
            character_ldmks = self.matches["character_pts"]
            smpl_ldmk_coords = self.matches["smpl_baryc_coords"]
            smpl_ldmk_id = self.matches["smpl_verts_id"]

            def compute_smpl_landmark( deformed_verts):
                smpl_ldmk_verts = deformed_verts [smpl_ldmk_id.view(-1)]
                smpl_ldmk_verts = smpl_ldmk_verts.view(-1, 3, 3)  # n, bary_n(3), (3)
                smpl_ldmks = (smpl_ldmk_verts * smpl_ldmk_coords.view(-1, 3, 1)).sum(dim=1)
                return smpl_ldmks
                # landmarks = [smpl_ldmks.float() , character_ldmks.float()]

        params = [{ 'params': geo_model.parameters() }  ]
        optimizer = torch.optim.Adam(params=params, lr=cfg.lr)

        t_loop = tqdm(range(cfg.epochs), leave=False)

        target_mesh = Meshes(verts=[torch.from_numpy(np.array(self.mesh.vertices)).float().to("cuda")],  faces=[torch.from_numpy(np.array(self.mesh.triangles)).to("cuda")])

        faces = self.faces.to(self.device)

        for it in t_loop:


            # updated vertices from paramnet
            vert_offset = geo_model( )
            deformed_verts = self.smplverts + vert_offset[0]
            optimizer.zero_grad()
            pred_mesh = Meshes(verts=[deformed_verts], faces=[ faces])

            losses = {}

            #### chamfer distance loss
            sample_target = sample_points_from_meshes(target_mesh, 5000)
            sample_pred = sample_points_from_meshes(pred_mesh, 5000)
            cd_loss, _ = chamfer_distance(sample_pred, sample_target, single_directional=False)
            losses ["chamfer"] = cd_loss * cfg.cd_weight

            ##### laplacian loss
            lp_loss = mesh_laplacian_smoothing(pred_mesh, method="uniform")
            losses ["laplacian"] = lp_loss * cfg.laplacian

            ### trig loss
            losses ["edge"] = mesh_edge_loss(pred_mesh) * cfg.edge_weight

            ### edge loss
            # edge_loss = lossfunc.relative_edge_loss( deformed_verts,  self.smplverts, faces=faces.cpu())
            # losses ["edge"] = edge_loss * cfg.edge_weight



            ### offset regularization
            losses ["offset_reg"] = (vert_offset**2).sum(-1).mean() * cfg.offset_reg



            ### landmark loss
            if self.matches:
                warped_ldmk = compute_smpl_landmark( deformed_verts )
                losses ["ldmk"] = torch.mean(torch.sum((warped_ldmk - character_ldmks) ** 2, dim=-1))
                # pass
            else :
                losses ["ldmk"] = 0


            total_loss = losses["chamfer"] + \
                         losses ["laplacian"] +\
                         losses ["ldmk"] +\
                         losses ["edge"] + \
                         losses ["offset_reg"]

            # print( "total_loss:", total_loss )
            total_loss.backward()
            optimizer.step()
            desc =f'Total={total_loss.item():.5f}|--| '
            for k, v  in losses.items() :
                 desc = desc + k + f'={v.item():.5f}| '
            t_loop.set_description( desc )


            viz = True
            if viz and it%999 == 0:
                viz_mesh_src = obtain_viz_mesh(deformed_verts.detach().cpu().numpy(), self.faces.detach().cpu().numpy())
                o3d.visualization.draw_geometries([viz_mesh_src, self.mesh])

        print( "msg:", desc)

        o3d.io.write_triangle_mesh( "/home/rabbityl/workspace/auto_rig/bodyfit/reg_compare/paramnet_no_match.ply", viz_mesh_src)


        return None, None, None


    def obtain_head_neck_border(self):

        data = {"body_pose": self.body_poses_opt,
                "global_pose": self.global_orient,
                "transl": self.transl}

        body_pose = rotation_converter.batch_euler2matrix(data["body_pose"].reshape(21, 3))  # 21,3,3
        global_pose = rotation_converter.batch_euler2matrix(data["global_pose"])  # 21,3,3

        smplx = self.model.smplx

        posed_vertices, landmarks, joints = smplx.forward(
            body_pose=body_pose[None],
            global_pose=global_pose[None],
            transl=data["transl"]
        )

        smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        smplmesh.vertices = o3d.utility.Vector3dVector(posed_vertices[0].detach().cpu().numpy())

        smplmesh.triangles = o3d.utility.Vector3iVector(smplx.faces.detach().cpu().numpy())
        smplmesh.compute_vertex_normals()

        indexs = "/home/rabbityl/workspace/auto_rig/bodyfit/delta/smplx_vert_segmentation.json"
        with open(indexs) as f:
            indexs = json.load(f)
            head = indexs["head"]
            neck = indexs["neck"]

        color = np.array([[0.5, 0.5, 0.5]] * len(posed_vertices[0]))
        color[head] = np.array([0.2, 0.99, 0.8])
        smplmesh.vertex_colors = o3d.utility.Vector3dVector(color)

        face = smplx.faces.detach().cpu().numpy()
        verts = posed_vertices[0].detach().cpu().numpy()

        neck_border = []
        for fi in range(len(face)):
            a, b, c = face[fi]

            for x in [a, b, c]:
                if x in head:
                    for y in [a, b, c]:
                        if y in neck:
                            if y not in neck_border:
                                neck_border.append(y)

        # neck_border = np.asarray( neck_border )[:,1]
        # for ele in neck_border:
        #     print( ele, ",")

        print(neck_border)

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(verts[neck_border])
        pc.paint_uniform_color([1, 0, 0])

        viz = True
        if viz:
            o3d.visualization.draw([smplmesh, pc])

        # return smplx.faces, posed_vertices[0]





    def checkface(self ):


        data = {"body_pose": self.body_poses_opt,
                "global_pose": self.global_orient,
                "transl": self.transl}

        body_pose = rotation_converter.batch_euler2matrix(data["body_pose"].reshape(21, 3))  # 21,3,3
        global_pose = rotation_converter.batch_euler2matrix(data["global_pose"])  # 21,3,3

        smplx = self.model.smplx

        posed_vertices, landmarks, joints = smplx.forward(
            body_pose=body_pose[None],
            global_pose=global_pose[None],
            transl=data["transl"]
        )

        smplmesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        smplmesh.vertices = o3d.utility.Vector3dVector(posed_vertices[0].detach().cpu().numpy())

        smplmesh.triangles = o3d.utility.Vector3iVector(smplx.faces.detach().cpu().numpy())
        smplmesh.compute_vertex_normals()

        part_idx_dict = smplx.get_part_index()
        face_idx = part_idx_dict[ 'face' ]

        color = np.array([[0.5, 0.5, 0.5]] * len(posed_vertices[0]))
        color[face_idx] = np.array([0.2, 0.99, 0.8])
        smplmesh.vertex_colors = o3d.utility.Vector3dVector(color)



        viz = True
        if viz:
            o3d.visualization.draw([smplmesh])


