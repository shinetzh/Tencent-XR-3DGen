import mcubes, trimesh
import torch
import numpy  as np
import os
from easydict import EasyDict as edict
from torch.autograd import grad
from tqdm import tqdm
import trimesh
import json
import matplotlib.pyplot as plt

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def plot_texmesh_cond(Gnet, Gtri, Tnet, Ttri, oid, reso, savedir):
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            query_func=lambda pts: - Gnet(Gtri(pts, oid))
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        v = torch.from_numpy(vertices).float().cuda()
        feat = torch.cat([Gtri(v,oid),Ttri(v,oid)], -1)
        colors = Tnet(feat).detach().cpu().numpy()
        # colors = np.concatenate([colors, np.ones([colors.shape[0], 1])], -1)
        colors = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)
        mesh.export(savedir)
    


def plot_texmesh(Gnet, Gtri, Tnet, Ttri, oid, reso, savedir):
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            query_func=lambda pts: - Gnet(Gtri(pts, 0))
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        v = torch.from_numpy(vertices).float().cuda()
        colors = Tnet(Ttri(v, oid)).detach().cpu().numpy()
        # colors = np.concatenate([colors, np.ones([colors.shape[0], 1])], -1)
        colors = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)
        mesh.export(savedir)
    

def plot_colors(net, oid, reso, savedir, triplane):
    with torch.no_grad():
        u, p = extract_fields(
            [-1., -1., -1.], 
            [1., 1., 1.], 
            reso, 
            lambda pts: net(triplane(pts, oid)),
            3
        )
        u = u.clip(0.0, 1.0)
        rgba = (np.concatenate([u, np.ones([reso, reso, reso, 1])], -1) * 255).astype(np.uint8)
        pcd = trimesh.points.PointCloud(p.reshape(-1,3), rgba.reshape(-1,4))
        pcd.export(savedir)

def vis_model(net, triplane,resolution, n_labels, savedir, oid):
    os.makedirs(savedir, exist_ok=True)
    plot_shapes(net, triplane, resolution, n_labels, 0.0, savedir, oid)


def plot_shapes(net, triplane, resolution, channel, threshold, savedir,  oid):
    mu = extract_fields(
        bound_min=[-1.0, -1.0, -1.0],
        bound_max=[ 1.0,  1.0,  1.0],
        resolution=resolution,
        query_func=lambda xyz: -net(triplane(xyz, oid)),
        channel=channel,
    )
    for pid in range(channel):
        u = mu[..., pid]  # occupancy of part
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        vertices = vertices / (resolution - 1.0) * 2 - 1
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export( os.path.join( savedir, "part-"+ str(pid) + ".ply"))


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):


    u, _ = extract_fields(bound_min, bound_max, resolution, query_func, 1)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    return vertices, triangles



def extract_fields(bound_min, bound_max, resolution, query_func, channel):
    N = 128 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, channel], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs), channel).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u




def update_lr(opt, epoch, config):
    lr_factor = (np.cos(np.pi * min(1.0, epoch/config.lr_max_iter)) + 1.0) * 0.5 * (1 - 0.05) + 0.05
    for g in opt.param_groups:
        if 'mlp' in g['name']:
            g['lr'] = lr_factor * config.lr
        elif  'tri' in g['name']:
            g['lr'] = lr_factor * config.lr_tri


def check_and_create_dict(config):
    if isinstance(config.objects, str):
        if config.objects.endswith('.txt'):
            with open(config.objects, 'r') as ftxt:
                config.objpaths = [line.strip('\n') for line in ftxt.readlines()]
        else:
            config.objpaths = [config.objects]
    elif isinstance(config.objects, list):
        config.objpaths = config.objects

    name2path = {}
    path2name = {}
    for objpath in tqdm(config.objpaths):
        objname = objpath.split('/')[-1] + '_' + config.part
        name2path[objname] = objpath
        path2name[objpath] = objname
    assert len(name2path.keys()) == len(path2name.keys()) # one-to-one mapping
    with open(os.path.join(config.savedir, 'name2path.json'), 'w') as jf:
        jf.write(json.dumps(name2path, indent=4))
    with open(os.path.join(config.savedir, 'path2name.json'), 'w') as jf:
        jf.write(json.dumps(path2name, indent=4))
    config.path2name = path2name


def plot_preview(loaddir, savedir):
    # print logged messages
    trimesh.util.attach_to_log()
    log = trimesh.util.log

    # load a mesh
    mesh = trimesh.load(loaddir)

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    scene = mesh.scene()

    # a 90 degree homogeneous rotation matrix around the X axis at the scene centroid
    rotate = trimesh.transformations.rotation_matrix(
        angle=np.radians(90.0),
        direction=[1, 0, 0],
        point=scene.centroid)

    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(rotate, camera_old)
    # apply the new transform
    scene.graph[scene.camera.name] = camera_new

    for i in range(1) :

        # rotate = trimesh.transformations.rotation_matrix(
        #     angle=np.radians(90.0),
        #     direction=[0, 1, 0],
        #     point=scene.centroid)
        #
        # camera_old, _geometry = scene.graph[scene.camera.name]
        # camera_new = np.dot(rotate, camera_old)
        # # apply the new transform
        # scene.graph[scene.camera.name] = camera_new

        try:
            # save a render of the object as a png
            png = scene.save_image(visible=False)  # resolution=[1080, 1080], visible=True)
            with open(savedir, 'wb') as f:
                f.write(png)
                f.close()

        except BaseException as E:
            log.debug("unable to save image", str(E))