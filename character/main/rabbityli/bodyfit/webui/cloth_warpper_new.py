import copy
import importlib.util
import os, sys
import argparse
import shutil
from glob import glob
import trimesh
import torch
import pathlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import cv2
import scipy
import trimesh
import time

import numpy as np
from scipy import sparse


code_file_path = "/aigc_cfs_2/rabbityli/bodyfit/webui/cloth_warpper.py"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(code_file_path), '..')))

from lib.timer import Timers
timers = Timers()
from bodyfit.lib.utils.util import batch_transform
from pytorch3d.ops.knn import knn_points
import json
import open3d as o3d
import numpy as np
from pathlib import Path

def load_json(j):
    with open( j) as f:
        data = json.load(f)
    return data

base_body_map = load_json(os.path.join(os.path.dirname(code_file_path), "base_body_map.json"))
base_body_map_source = load_json(os.path.join(os.path.dirname(code_file_path), "base_body_map_source.json"))
indexs = str(Path(os.path.join(os.path.dirname(code_file_path))).parent / "bodyfit" / "smplx_vert_segmentation.json")



body_info = {}
with open(indexs) as f:
    indexs = json.load(f)
    head_index = indexs["head"]

    ear_index = indexs["right_ear"] + indexs["left_ear"]

    hair_index = [x for x in head_index if x not in ear_index]

    arms = ["leftArm", "rightArm",
            "leftForeArm", "rightForeArm",
            "leftHand", "rightHand",
            "leftHandIndex1", "rightHandIndex1",
            "rightShoulder", "leftShoulder"
            ]
    arms_index = []
    for ele in arms:
        arms_index = arms_index + indexs[ele]

    torsol = ["leftLeg", "leftToeBase", "leftFoot", "spine1", "spine2", "rightFoot", "rightLeg", "rightToeBase",
              "spine", "leftUpLeg", "hips", "rightUpLeg", "neck"]
    torsol_index = []
    for ele in torsol:
        torsol_index = torsol_index + indexs[ele]

    left_shoe = ["leftLeg", "leftToeBase", "leftFoot", "leftUpLeg"]
    left_shoe_index = []
    for ele in left_shoe:
        left_shoe_index = left_shoe_index + indexs[ele]
    left_foot_index = indexs["leftFoot"]

    right_shoe = ["rightLeg", "rightToeBase", "rightFoot", "rightUpLeg"]
    right_shoe_index = []
    for ele in right_shoe:
        right_shoe_index = right_shoe_index + indexs[ele]
    right_foot_index = indexs["rightFoot"]

    part_index_map = {
        "hair": hair_index,
        "left_shoe": left_shoe_index,
        "right_shoe": right_shoe_index,
        "left_foot": left_foot_index,
        "right_foot": right_foot_index,
        "torsol": torsol_index,
        "full_cloth": torsol_index + arms_index
    }


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh_path", type=str, required=True)
    # parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--config", choices=["daz", "vroid", "base"], default=None)
    args = parser.parse_args()
    return args



def Scene2Trimesh(m):
    meshes = []
    for k in m.geometry.keys():
        ms = m.geometry[k]
        meshes.append(ms)
    m = trimesh.util.concatenate(meshes)
    return m


def compute_scale_diff(src_head, tgt_head, scale_adjust=1.5):
    src_mean = src_head.mean(dim=1, keepdims=True)
    tgt_mean = tgt_head.mean(dim=1, keepdims=True)
    src_len = torch.norm(src_head - src_mean, dim=-1)
    tgt_len = torch.norm(tgt_head - tgt_mean, dim=-1)
    scale = (tgt_len / src_len).mean() * scale_adjust
    return scale



def fix_hair_mtl_names(obj_path):
    mtl_name = obj_path.split("/")[-1]

    mtl_file = os.path.join(str(pathlib.Path(obj_path).parent), "material.mtl")
    # fix material name path
    with open(mtl_file, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if line[:6] == "newmtl":
                lines[idx] = " ".join(["newmtl", mtl_name, "\n"])
    with open(mtl_file, 'w') as file:
        file.writelines(lines)

    with open(mtl_file, 'r') as file:
        lines = file.readlines()
        print(lines)


def interp_transform(points, template_points, template_transform, K=6):
    results = knn_points(points[None], template_points, K=K)
    dists, idxs = results.dists, results.idx
    neighbs_weight = torch.exp(-dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)
    neighbs_transform = template_transform[:, idxs[0], :, :].view(1, -1, 4, 4)
    points_K = points[:, None].repeat(1, K, 1).view(-1, 3)
    points_K_warpped = batch_transform(neighbs_transform, points_K).reshape(1, -1, K, 3)
    points_merge = (neighbs_weight[..., None] * points_K_warpped).sum(dim=-2)
    return points_merge.squeeze(), idxs.squeeze()


def interp_transform_scaled(points, template_points, point_scaled, template_transform, K=6):
    '''
    @point: asset point
    @template_points: smpl verts
    @point_scaled: asset point scaled
    @template_transform: smpl transformations
    '''

    results = knn_points(points[None], template_points, K=K)
    dists, idxs = results.dists, results.idx
    neighbs_weight = torch.exp(-dists)
    neighbs_weight = neighbs_weight / neighbs_weight.sum(-1, keepdim=True)

    neighbs_transform = template_transform[:, idxs[0], :, :].view(1, -1, 4, 4)
    points_K = point_scaled[:, None].repeat(1, K, 1).view(-1, 3)
    points_K_warpped = batch_transform(neighbs_transform, points_K).reshape(1, -1, K, 3)
    points_merge = (neighbs_weight[..., None] * points_K_warpped).sum(dim=-2)
    return points_merge.squeeze(), idxs.squeeze()


def interp_transform_geodesic(points, template_points, template_transform, geo_dists, K=6):
    v_2_arm, arms_index = geo_dists["arm"]
    v_2_torsol, torsol_index = geo_dists["torsol"]

    arm_pts = template_points[:, arms_index]
    arm_trans = template_transform[:, arms_index]

    torsol_pts = template_points[:, torsol_index]
    torsol_trans = template_transform[:, torsol_index]

    arm_deform, _ = interp_transform(points, arm_pts, arm_trans)  # points, template_points, template_transform, K = 6
    torsol_deform, _ = interp_transform(points, torsol_pts,
                                        torsol_trans)  # points, template_points, template_transform, K = 6

    temprature = 1
    v_2_arm = torch.exp(-v_2_arm / temprature)
    v_2_torsol = torch.exp(-v_2_torsol / temprature)
    gsum = v_2_arm + v_2_torsol
    w_arm = v_2_arm / gsum
    w_torsol = v_2_torsol / gsum

    p_deform = w_arm[:, None] * arm_deform + w_torsol[:, None] * torsol_deform

    return p_deform


class LayeredAvatar():
    def __init__(self, param_data, part_paths, body_manifold=None, trns=None, label="hair"):

        timers.tic('forward_skinning')
        param_data = torch.load(param_data, map_location='cuda:0')
        self.faces = param_data['faces'].to(torch.device(0))
        self.posed_verts = param_data['posed_verts'].to(torch.device(0))
        self.T = param_data['T'].to(torch.device(0))
        timers.toc('forward_skinning')

        self.transform = trns

        if self.transform is not None:
            self.transform = torch.from_numpy(self.transform).float().to(self.T.device)

        self.mesh_holder = None
        self.label = label
        if part_paths is not None:
            self.mesh_holder = trimesh.load(part_paths)

        self.body_manifold = None
        if body_manifold is not None:
            inmesh = o3d.io.read_triangle_mesh(body_manifold)
            inmesh.compute_vertex_normals()
            scene = o3d.t.geometry.RaycastingScene()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(inmesh)
            _ = scene.add_triangles(mesh)
            self.body_manifold = {"scene": scene, "mesh": inmesh}

    def warp_mesh_full(self, T_s2t, target_avatars=None, save_path=None):

        m = self.mesh_holder

        if isinstance(m, trimesh.scene.scene.Scene):  # need to handle uv pieces separately
            m = Scene2Trimesh(m)

        if self.label == "hair":
            m = self.warp_hair(T_s2t, m, target_avatars)
        elif self.label == "shoe":
            m = self.warp_shoes(T_s2t, m, target_avatars)
        # elif self.label == "trousers":
        #     pass
        else:
            m = self.warp_mesh(T_s2t, m, target_avatars)

        if save_path:

            if self.label == "shoe" :
                lm, rm = m
                mtl_name = save_path.split("/")[-1]
                # print("save mesh material", m.visual.material.name)
                lm.visual.material.name = mtl_name
                rm.visual.material.name = mtl_name
                lm.export(save_path+ "_left.obj")
                rm.export(save_path+ "_right.obj")

            else :

                mtl_name = save_path.split("/")[-1]
                # print("save mesh material", m.visual.material.name)
                m.visual.material.name = mtl_name
                m.export(save_path)
                # print("save mesh material", m.visual.material.name)
                print("save mesh", save_path)

                if self.label == "hair":
                    print("label:", self.label)
                    fix_hair_mtl_names(save_path)

    def scale_aware_warp(self, T_s2t, part_index, scale, smpl_vert_src, asset_vert, K=10):
        '''
        @T_s2t: transforms of all smpl verts
        @part_index: index of refered part
        @scale: scale
        @smpl_vert_src: smpl vert of refered part
        @asset_vert: vert on asset to be transformed
        '''
        T_left_2_tgt = T_s2t[:, part_index]
        T_scale_2_left = torch.eye(4)[None, None].repeat(1, T_left_2_tgt.shape[1], 1, 1).to(smpl_vert_src)
        T_scale_2_left[..., :3, 3:] = (smpl_vert_src - scale * smpl_vert_src).reshape(1, -1, 3, 1)
        T_scale_2_tgt = T_left_2_tgt @ T_scale_2_left
        asset_vert_scaled = asset_vert * scale
        asset_vert_deform, ref_idx = interp_transform_scaled(asset_vert, self.posed_verts[:, part_index], asset_vert_scaled, T_scale_2_tgt, K=K)
        return asset_vert_deform, ref_idx

    def warp_shoes(self, T_s2t, m, target_avatars):

        if isinstance(m, trimesh.base.Trimesh):

            # split to left and right shoe
            parts = trimesh.graph.connected_components(m.vertex_adjacency_graph.edges)
            vert = torch.from_numpy(m.vertices).to(self.T)
            left_indexes = []
            right_indexes = []
            for p in parts:
                is_left = vert[p][:, 0] > 0
                is_left = True if is_left.sum() > (0.5 * len(is_left)) else False
                if is_left:
                    left_indexes = left_indexes + list(p)
                else:
                    right_indexes = right_indexes + list(p)
            left_vert = vert[left_indexes]
            right_vert = vert[right_indexes]

            ## visualize center line
            # left_vert  = np.asarray(vert[ left_indexes ].cpu())
            # right_vert = np.asarray(vert[ right_indexes ].cpu())
            # pc_l = o3d.geometry.PointCloud()
            # pc_l.points = o3d.utility.Vector3dVector(left_vert)
            # pc_l.paint_uniform_color([1, 0, 0])
            # pc_r = o3d.geometry.PointCloud()
            # pc_r.points = o3d.utility.Vector3dVector(right_vert)
            # pc_r.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([pc_l, pc_r])

            # import pdb; pdb.set_trace()


            if self.transform is not None:
                # vert = (self.transform[:3, :3] @ vert.T + self.transform[:3, 3:]).T
                left_vert =  (self.transform[:3, :3] @ left_vert.T + self.transform[:3, 3:]).T
                right_vert = (self.transform[:3, :3] @ right_vert.T + self.transform[:3, 3:]).T

            # points on SMPL
            src_left = self.posed_verts[:, part_index_map["left_shoe"]]  # .detach().cpu().numpy()[0]
            tgt_left = target_avatars.posed_verts[:, part_index_map["left_shoe"]]  # .detach().cpu().numpy()[0]
            src_right = self.posed_verts[:, part_index_map["right_shoe"]]
            tgt_right = target_avatars.posed_verts[:, part_index_map["right_shoe"]]

            # with scale
            # scale_adjust = body_info [][][]
            scale_adjust = base_body_map[body_info[0]][body_info[1]]["shoes_scale"]
            scale = compute_scale_diff(src_left, tgt_left, scale_adjust=scale_adjust)  # compute scale difference
            left_vert_deform, _ = self.scale_aware_warp(T_s2t, part_index_map["left_shoe"], scale, src_left, left_vert,
                                                        K=10)
            right_vert_deform, _ = self.scale_aware_warp(T_s2t, part_index_map["right_shoe"], scale, src_right,
                                                         right_vert, K=10)

            # modify mesh vertex positions
            m.vertices[left_indexes] = left_vert_deform.squeeze().detach().cpu().numpy()
            m.vertices[right_indexes] = right_vert_deform.squeeze().detach().cpu().numpy()


            # save left and right shoes separately
            lm = copy.deepcopy(m)
            rm = copy.deepcopy(m)
            lmask = np.zeros((len(m.vertices))).astype(int)
            rmask = np.zeros((len(m.vertices))).astype(int)
            lmask[left_indexes] = 1
            rmask[right_indexes] = 1
            lfaces = np.asarray(m.faces)
            rfaces = np.asarray(m.faces)
            lface_mask = lmask[lfaces].sum(-1) ==3
            rface_mask = rmask[rfaces].sum(-1) ==3

            lm.update_faces(lface_mask)
            rm.update_faces(rface_mask)

        else:
            raise NotImplementedError()

        return (lm, rm)

    def warp_full_body_dress(self, T_s2t, m):

        posed_verts = self.posed_verts.detach().cpu().numpy()[0]

        if self.transform is not None:
            transform = self.transform.cpu().numpy()
            posed_verts = (transform[:3, :3].T @ (posed_verts.T - transform[:3, 3:])).T

        body_mesh = trimesh.Trimesh(posed_verts, self.faces.detach().cpu().numpy())
        body_vert_num = len(posed_verts)

        # part verts:
        head_verts = posed_verts[head_index]
        torsol_verts = posed_verts[torsol_index]
        arm_verts = posed_verts[arms_index]

        if isinstance(m, trimesh.scene.scene.Scene):  # need to handle uv pieces separately

            # merge scene to a single mesh
            meshes = [body_mesh]
            for k in m.geometry.keys():
                ms = m.geometry[k]
                meshes.append(ms)
            meshc = trimesh.util.concatenate(meshes)
            mesh_cloth = trimesh.util.concatenate(meshes[1:])

            vox = meshc.voxelized(pitch=0.015)
            n_vox = len(vox.sparse_indices)
            sparse_id = np.arange(n_vox)
            sparse_vox = np.array(vox.sparse_indices)

            id_holder = np.ones(vox.shape, dtype=int) * -1
            id_holder[sparse_vox[:, 0], sparse_vox[:, 1], sparse_vox[:, 2]] = sparse_id

            # assign body part labels
            # -1 others, arm 0, torsol 1, head 2,
            part_holder = np.zeros(vox.shape, dtype=int) - 1
            head_indices = vox.points_to_indices(head_verts)
            tors_indices = vox.points_to_indices(torsol_verts)
            arm_indices = vox.points_to_indices(arm_verts)
            part_holder[head_indices[:, 0], head_indices[:, 1], head_indices[:, 2]] = 2
            part_holder[arm_indices[:, 0], arm_indices[:, 1], arm_indices[:, 2]] = 0
            part_holder[tors_indices[:, 0], tors_indices[:, 1], tors_indices[:, 2]] = 1
            part_holder = part_holder[sparse_vox[:, 0], sparse_vox[:, 1], sparse_vox[:, 2]]

            ### construct the graph
            X, Y, Z = vox.shape
            x_p1, y_p1, z_p1 = sparse_vox.copy(), sparse_vox.copy(), sparse_vox.copy()
            x_p1[:, 0] = x_p1[:, 0] + 1
            x_p1[:, 0][x_p1[:, 0] == X] = X - 1
            y_p1[:, 1] = y_p1[:, 1] + 1
            y_p1[:, 1][y_p1[:, 1] == Y] = Y - 1
            z_p1[:, 2] = z_p1[:, 2] + 1
            z_p1[:, 2][z_p1[:, 2] == Z] = Z - 1
            edges = []
            for anchor in [x_p1, y_p1, z_p1]:
                mask = vox.matrix[anchor[:, 0], anchor[:, 1], anchor[:, 2]]
                id = id_holder[anchor[:, 0], anchor[:, 1], anchor[:, 2]]
                e = np.stack((sparse_id, id), axis=-1)
                e = e[mask]
                edges.append(e)
            edges = np.concatenate(edges, axis=0)
            coo_mat = trimesh.graph.edges_to_coo(edges)

            # compute shortest path
            dist_matrix = scipy.sparse.csgraph.dijkstra(coo_mat, directed=False, indices=None,
                                                        return_predecessors=False, unweighted=True, limit=np.inf,
                                                        min_only=False)

            dist_matrix = dist_matrix  # / max(vox.shape)

            vis_geo = True
            if vis_geo:
                vert_indices = vox.points_to_indices(meshc.vertices[body_vert_num:])
                vert_id = id_holder[vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]]
                vert_dist_map = dist_matrix[vert_id]
                v_2_torsol = vert_dist_map[:, part_holder == 1]  # torsol
                v_2_arm = vert_dist_map[:, part_holder == 0]  # arm
                v_2_torsol = v_2_torsol.min(axis=1)
                v_2_arm = v_2_arm.min(axis=1)

                color = np.ones_like(meshc.vertices[body_vert_num:]) * 0
                color[:, 0] = 1
                clr_2arm, clr_2tsl = color.copy(), color.copy()
                clr_2arm = clr_2arm * v_2_arm[:, None] / v_2_torsol.max()
                clr_2tsl = clr_2tsl * v_2_torsol[:, None] / v_2_torsol.max()

                mesh1 = o3d.geometry.TriangleMesh()
                mesh1.vertices = o3d.utility.Vector3dVector(meshc.vertices[body_vert_num:])
                mesh1.triangles = o3d.utility.Vector3iVector(mesh_cloth.faces)
                mesh1.vertex_colors = o3d.utility.Vector3dVector(clr_2arm)
                mesh1.compute_vertex_normals()

                mesh2 = o3d.geometry.TriangleMesh()
                mesh2.vertices = o3d.utility.Vector3dVector(meshc.vertices[body_vert_num:])
                mesh2.triangles = o3d.utility.Vector3iVector(mesh_cloth.faces)
                mesh2.vertex_colors = o3d.utility.Vector3dVector(clr_2tsl)
                mesh2.compute_vertex_normals()

                o3d.visualization.draw([mesh1, mesh2])

            for key in m.geometry.keys():

                # interp_transform(vert, self.posed_verts, T_s2t, K=6)

                # Query distance map
                vert_indices = vox.points_to_indices(m.geometry[key].vertices)
                vert_id = id_holder[vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]]
                vert_dist_map = dist_matrix[vert_id]
                v_2_torsol = vert_dist_map[:, part_holder == 1]  # torsol
                v_2_arm = vert_dist_map[:, part_holder == 0]  # arm
                v_2_torsol = v_2_torsol.min(axis=1)
                v_2_arm = v_2_arm.min(axis=1)
                geo_dists = {
                    "arm": [torch.from_numpy(v_2_arm).to(self.posed_verts.device),
                            torch.from_numpy(np.asarray(arms_index, dtype=int)).to(self.posed_verts.device)],
                    "torsol": [torch.from_numpy(v_2_torsol).to(self.posed_verts.device),
                               torch.from_numpy(np.asarray(torsol_index, dtype=int)).to(self.posed_verts.device)]
                }

                vis_geo = True
                if vis_geo:
                    mesh1 = o3d.geometry.TriangleMesh()
                    mesh1.vertices = o3d.utility.Vector3dVector(m.geometry[key].vertices)
                    mesh1.triangles = o3d.utility.Vector3iVector(m.geometry[key].faces)
                    mesh1.paint_uniform_color([0.7, 0.7, 0.7])
                    mesh1.compute_vertex_normals()
                    o3d.visualization.draw([mesh1])

                vert = torch.from_numpy(m.geometry[key].vertices).to(self.T)

                if self.transform is not None:
                    vert = (self.transform[:3, :3] @ vert.T + self.transform[:3, 3:]).T

                vert_deform = interp_transform_geodesic(vert, self.posed_verts, T_s2t, geo_dists, K=6)
                vert_deform = vert_deform.squeeze().detach().cpu().numpy()

                m.geometry[key].vertices = vert_deform


        elif isinstance(m, trimesh.base.Trimesh):

            vert = torch.from_numpy(m.vertices).to(self.T)

            if self.transform is not None:
                vert = (self.transform[:3, :3] @ vert.T + self.transform[:3, 3:]).T

            vert_deform, ref_idx = interp_transform(vert, self.posed_verts, T_s2t, K=6)
            m.vertices = vert_deform.squeeze().detach().cpu().numpy()

        else:
            raise NotImplementedError

        return m

    def warp_hair(self, T_s2t, m, target_avatars):

        if isinstance(m, trimesh.base.Trimesh):

            vert = torch.from_numpy(m.vertices).to(self.T)
            if self.transform is not None:
                vert = (self.transform[:3, :3] @ vert.T + self.transform[:3, 3:]).T

            # points on SMPL
            src_head = self.posed_verts[:, part_index_map["hair"]]  # .detach().cpu().numpy()[0]
            tgt_head = target_avatars.posed_verts[:, part_index_map["hair"]]  # .detach().cpu().numpy()[0]

            # compute rough scale difference
            scale = compute_scale_diff(src_head, tgt_head, scale_adjust=1.0)
            vert_deform, ref_idx = self.scale_aware_warp(T_s2t, part_index_map["hair"], scale, src_head, vert, K=10)
            m.vertices = vert_deform.squeeze().detach().cpu().numpy()

        else:
            raise NotImplementedError

        return m

    def warp_mesh(self, T_s2t, m, target_avatars):

        if isinstance(m, trimesh.base.Trimesh):
            vert = torch.from_numpy(m.vertices).to(self.T)
            if self.transform is not None:
                vert = (self.transform[:3, :3] @ vert.T + self.transform[:3, 3:]).T

            # compute rough scale difference using points on SMPL
            src_torsol = self.posed_verts[:, part_index_map["torsol"]]  # .detach().cpu().numpy()[0]
            tgt_torsol = target_avatars.posed_verts[:, part_index_map["torsol"]]  # .detach().cpu().numpy()[0]
            scale = compute_scale_diff(src_torsol, tgt_torsol, scale_adjust=1.0)

            # print("inside warp mesh, scale:", scale)

            src_full_cloth = self.posed_verts[:, part_index_map["full_cloth"]]  # .detach().cpu().numpy()[0]

            vert_deform, ref_idx = self.scale_aware_warp(T_s2t, part_index_map["full_cloth"], scale, src_full_cloth,
                                                         vert, K=4)
            print ( "k = 4")
            m.vertices = vert_deform.squeeze().detach().cpu().numpy()

        else:
            raise NotImplementedError
        return m

    def swap_cloth_to(self, target_avatars, save_path):
        # import pdb;
        # pdb.set_trace()
        T_s2t = target_avatars.T @ torch.inverse(self.T)
        self.warp_mesh_full(T_s2t, save_path=save_path, target_avatars=target_avatars)


def warp_one_cloth(asset_path, asset_body, T, G_trns, dump_path, label):
    A = LayeredAvatar(asset_body, asset_path, trns=G_trns, label=label)

    timers.tic('swap_cloth_to')

    A.swap_cloth_to(T, dump_path)

    timers.toc('swap_cloth_to')




def main():
    global body_info

    key_smpl_map = load_json( "/aigc_cfs_2/rabbityli/bodyfit/webui/20240711_key_smpl_map.json" )


    parser = argparse.ArgumentParser()
    parser.add_argument("--lst_path", type=str, required=True)
    args = parser.parse_args()

    lst_path = args.lst_path

    dump_root = Path(lst_path).parent

    print("---------------inside warp_clothes()---------------")
    print(lst_path)

    with open(lst_path, "rb") as f:
        data = json.load(f)
        part_info = data["path"]
        body_info = data["body_attr"]

        print("body_info", body_info)

    if len(part_info) == 0:
        print("no items in the list")
        exit()

    # config smplx
    from bodyfit.lib.configs.config_vroid import get_cfg_defaults
    cfg = get_cfg_defaults()

    # timers.tic('load smplx')
    # smplxs = SMPL_with_scale(cfg).to(torch.device(0))
    # timers.toc('load smplx')

    G_trns = np.eye(4)
    G_trns[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()

    print( "body_info", body_info )

    # body_info[0] = "male"
    # body_info[1] = "yuanmeng"
    # print(base_body_map)
    target_dir = base_body_map [body_info[0]][body_info[1]]["path"]
    # target_dir = "/aigc_cfs/rabbityli/base_bodies/MCWY2_F_T"

    target_body = os.path.join(target_dir, "smplx_and_offset_smplified.npz")
    print("target_body",target_body)
    T = LayeredAvatar(target_body, None)

    warp_lst = {}

    for idx, part in enumerate(part_info):


        label = part_info[part]["cat"]  # categories


        #check hair and shoes:
        if label == "shoe" and base_body_map[body_info[0]][body_info[1]]["use_shoes"]==False:
            print("use_shoes False , skip")
            continue
        if label == "hair" and base_body_map[body_info[0]][body_info[1]]["use_hair"] == False:
            print("use_hair False , skip")
            continue



        # if label == "others"

        name = "part_" + "%02d" % idx
        dump_dir = os.path.join(dump_root, name)
        Path(dump_dir).mkdir(exist_ok=True, parents=True)




        if label == "shoe":
            dump_path = os.path.join(dump_dir, name  )
        else :
            dump_path = os.path.join(dump_dir, name + ".obj")

        warp_lst[dump_dir] = label

        print("part", part)

        body_key = part_info[part]["key"]  # body key
        if body_key[1] in ["daz", "vroid"]:
            # asset_body = os.path.join( pathlib.Path (part).parent.parent, "smplx_and_offset_smplified.npz" )
            asset_key = part_info[part]["asset_key"]
            asset_body = key_smpl_map[ asset_key]

        else:
            nake_dir = base_body_map_source[body_key[0]][body_key[1]]["path"]
            asset_body = os.path.join(nake_dir, "smplx_and_offset_smplified.npz")

        print("asset_body" , asset_body)

        warp_one_cloth(part, asset_body, T, G_trns, dump_path, label)

    json_object = json.dumps(warp_lst, indent=4)
    with open(os.path.join(dump_root, "warp_lst.json"), "w") as f:
        f.write(json_object)

    with open(os.path.join(dump_root, "smplx-path.txt"), "w") as f:
        f.write(f"{target_dir}\n")


if __name__ == '__main__':
    timers.tic('total')
    main()
    timers.toc('total')
    timers.print()