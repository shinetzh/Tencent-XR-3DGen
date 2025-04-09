import mcubes, trimesh
import torch
import numpy  as np
import os
from easydict import EasyDict as edict
from torch.autograd import grad
from tqdm import tqdm
import json
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
import time
from glob import glob
import pymeshlab as pml
import xatlas
import nvdiffrast.torch as dr

import sys
root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

def cost_time(func):
    def fun(*args, **kwargs):
        tstart = time.perf_counter()
        print(f"Enter {func.__name__}")
        result = func(*args, **kwargs)
        print(f'Leave {func.__name__} cost time:{time.perf_counter() - tstart:.8f} s')
        return result

    return fun


@cost_time
def obj2glb(obj_path, output_glb_path, output_obj_path):
    blender_path = "/aigc_cfs_2/neoshang/software/blender-3.6.2-linux-x64/blender"
    obj2glb_path = os.path.join(root_dir, "obj2glb.py")
    result = os.system(f"{blender_path} -b -P {obj2glb_path} -- --input_path {obj_path} --output_obj_path {output_obj_path} --output_glb_path {output_glb_path} --rotate")
    return result >> 8

def plot_geomesh_glb(model, triplane_sdf, reso, savedir, decimate_target=30000):
    plot_geomesh_obj(model, triplane_sdf, reso, savedir, decimate_target=decimate_target)
    obj_path = glob(savedir + "/*.obj")[0]
    glb_path = os.path.join(savedir, "mesh.glb")
    obj2glb(obj_path, glb_path, obj_path)



@cost_time
def plot_geomesh_obj(model, triplane_sdf, reso, savedir, decimate_target=30000):
    device = triplane_sdf.device
    batch_size = triplane_sdf.shape[0]

    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            batch_size = batch_size,
            query_func=lambda pts: - model.forward_sdf(triplane_sdf, pts, geo_only=True)
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
    #     ### reduce floaters by post-processing...
        vertices, triangles = clean_mesh(
            vertices,
            triangles,
            min_f=64,
            min_d=20,
            repair=True,
            remesh=False,
        )


    ### decimation
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(
            vertices, triangles, decimate_target, remesh=False
        )

    @cost_time
    def _export_obj(v, f, h0, w0, ssaa=1):
        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        print(
            f"[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}"
        )

        # unwrap uvs
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        # pack_options.blockAlign = True
        # pack_options.bruteForce = False
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

        # render uv maps
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat(
            (uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])),
            dim=-1,
        )  # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        rast, _ = dr.rasterize(
            dr.RasterizeCudaContext(), uv.unsqueeze(0), ft, (h, w)
        )  # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        mask, _ = dr.interpolate(
            torch.ones_like(v[:, :1]).unsqueeze(0), rast, f
        )  # [1, h, w, 1]

        # masked query
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        if mask.any():
            xyzs = xyzs[mask]  # [M, 3]

            # # batched inference to avoid OOM
            # all_feats = []
            # head = 0
            # while head < xyzs.shape[0]:
            #     tail = min(head + 640000, xyzs.shape[0])
            #     with torch.no_grad():
            #         all_feats.append(model.forward_rgb(triplane_rgb, xyzs[head:tail].unsqueeze(0)).float())
            #     head += 640000
            # feats[mask] = torch.cat(all_feats, dim=0)   

            # all_feats = model.forward_rgb(triplane_rgb, xyzs.unsqueeze(0)).float()
            feats[mask] = 0.5

        feats = feats.view(h, w, -1)  # 6 channels
        mask = mask.view(h, w)

        # quantize [0.0, 1.0] to [0, 255]
        feats = feats.data.cpu().numpy()
        feats = (feats * 255).astype(np.uint8)

        ### NN search as a queer antialiasing ...
        mask = mask.data.cpu().numpy()

        inpaint_region = binary_dilation(mask, iterations=32)  # pad width
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            search_coords
        )
        _, indices = knn.kneighbors(inpaint_coords)

        feats[tuple(inpaint_coords.T)] = feats[
            tuple(search_coords[indices[:, 0]].T)
        ]

        # do ssaa after the NN search, in numpy
        feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo

        if ssaa > 1:
            feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(savedir, "uv.jpg"), feats0)

        # save obj (v, vt, f /)
        obj_file = os.path.join(savedir, "mesh.obj")
        mtl_file = os.path.join(savedir, "mesh.mtl")

        print(f"[INFO] writing obj mesh to {obj_file}")
        with open(obj_file, "w") as fp:
            fp.write("mtllib mesh.mtl \n")

            print(f"[INFO] writing vertices {v_np.shape}")
            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            print(f"[INFO] writing vertices texture coords {vt_np.shape}")
            for v in vt_np:
                fp.write(f"vt {v[0]} {1 - v[1]} \n")

            print(f"[INFO] writing faces {f_np.shape}")
            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1}/ {f_np[i, 1] + 1}/{ft_np[i, 1] + 1}/ {f_np[i, 2] + 1}/{ft_np[i, 2] + 1}/ \n"
                )

        with open(mtl_file, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            fp.write(f"map_Kd uv.jpg \n")

    vertices = torch.from_numpy(vertices.astype(np.float32)).to(device).contiguous()
    triangles = torch.from_numpy(triangles.astype(np.int32)).to(device).contiguous()
    _export_obj(vertices, triangles, 1024, 1024, 1)


@cost_time
def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(
            target_count=int(target), preserve_border=False, verbose=False
        )
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.Percentage(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

@cost_time
def clean_mesh(verts, faces, v_pct=1, min_f=64, min_d=20, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.Percentage(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.Percentage(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.Percentage(1)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

def plot_texmesh_obj(Gnet, Gtri, Tnet, Ttri, oid, reso, decimate_target, savedir, device='cuda', oname='no_name'):
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            query_func=lambda pts: - Gnet(Gtri(pts, 0))
        )
        vertices = vertices / (reso - 1.) * 2 - 1
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
        ### reduce floaters by post-processing...
        vertices, triangles = clean_mesh(
            vertices,
            triangles,
            min_f=8,
            min_d=5,
            repair=True,
            remesh=False,
        )

    ### decimation
    if decimate_target > 0 and triangles.shape[0] > decimate_target:
        vertices, triangles = decimate_mesh(
            vertices, triangles, decimate_target, remesh=False
        )


    def _export_obj(v, f, h0, w0, ssaa=1):
        # v, f: torch Tensor

        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        print(
            f"[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}"
        )

        # unwrap uvs
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        # pack_options.blockAlign = True
        # pack_options.bruteForce = False
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

        # render uv maps
        uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
        uv = torch.cat(
            (uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])),
            dim=-1,
        )  # [N, 4]

        if ssaa > 1:
            h = int(h0 * ssaa)
            w = int(w0 * ssaa)
        else:
            h, w = h0, w0

        rast, _ = dr.rasterize(
            dr.RasterizeCudaContext(), uv.unsqueeze(0), ft, (h, w)
        )  # [1, h, w, 4]
        xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
        mask, _ = dr.interpolate(
            torch.ones_like(v[:, :1]).unsqueeze(0), rast, f
        )  # [1, h, w, 1]

        # masked query
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)

        feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

        if mask.any():
            xyzs = xyzs[mask]  # [M, 3]

            # batched inference to avoid OOM
            all_feats = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                with torch.no_grad():
                    all_feats.append(Tnet(Ttri(xyzs[head:tail], oid)).float())
                head += 640000

            feats[mask] = torch.cat(all_feats, dim=0)

        feats = feats.view(h, w, -1)  # 6 channels
        mask = mask.view(h, w)

        # quantize [0.0, 1.0] to [0, 255]
        feats = feats.data.cpu().numpy()
        feats = (feats * 255).astype(np.uint8)

        ### NN search as a queer antialiasing ...
        mask = mask.data.cpu().numpy()

        inpaint_region = binary_dilation(mask, iterations=32)  # pad width
        inpaint_region[mask] = 0

        search_region = mask.copy()
        not_search_region = binary_erosion(search_region, iterations=3)
        search_region[not_search_region] = 0

        search_coords = np.stack(np.nonzero(search_region), axis=-1)
        inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

        knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
            search_coords
        )
        _, indices = knn.kneighbors(inpaint_coords)

        feats[tuple(inpaint_coords.T)] = feats[
            tuple(search_coords[indices[:, 0]].T)
        ]

        # do ssaa after the NN search, in numpy
        feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)  # albedo

        if ssaa > 1:
            feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(savedir, f"{oname}.jpg"), feats0)

        # save obj (v, vt, f /)
        obj_file = os.path.join(savedir, f"{oname}_mesh.obj")
        mtl_file = os.path.join(savedir, f"{oname}_mesh.mtl")

        print(f"[INFO] writing obj mesh to {obj_file}")
        with open(obj_file, "w") as fp:
            fp.write(f"mtllib {oname}_mesh.mtl \n")

            print(f"[INFO] writing vertices {v_np.shape}")
            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            print(f"[INFO] writing vertices texture coords {vt_np.shape}")
            for v in vt_np:
                fp.write(f"vt {v[0]} {1 - v[1]} \n")

            print(f"[INFO] writing faces {f_np.shape}")
            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n"
                )

        with open(mtl_file, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            fp.write(f"map_Kd {oname}.jpg \n")

    vertices = torch.from_numpy(vertices.astype(np.float32)).to(device).contiguous()
    triangles = torch.from_numpy(triangles.astype(np.int32)).to(device).contiguous()
    _export_obj(vertices, triangles, 1024, 1024, 1)

def plot_texmesh_split(model, triplane_list, reso, savedir):
    if isinstance(triplane_list, list):
        triplane_sdf, triplane_rgb = triplane_list
    else:
        triplane_sdf = triplane_list[:, 0:3, ...]
        triplane_rgb = triplane_list[:, 3:, ...]
    batch_size = triplane_sdf.shape[0]
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            batch_size = batch_size,
            query_func=lambda pts: - model.forward_sdf(triplane_sdf, pts, geo_only=True)
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        v = torch.from_numpy(vertices).float().cuda().unsqueeze(0)
        colors = model.forward_rgb(triplane_rgb, v).detach().cpu().numpy()
        # colors = np.concatenate([colors, np.ones([colors.shape[0], 1])], -1)

        colors = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors[0])
        mesh.export(savedir)


def plot_mesh(model, triplane, reso, savedir):
    triplane_sdf = triplane
    batch_size = triplane.shape[0]
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            batch_size = batch_size,
            query_func=lambda pts: - model.forward_sdf(triplane_sdf, pts, geo_only=True)
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        v = torch.from_numpy(vertices).float().cuda().unsqueeze(0)
        # colors = model.forward_rgb(triplane_rgb, v).detach().cpu().numpy()
        # colors = np.concatenate([colors, np.ones([colors.shape[0], 1])], -1)
        # colors = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(savedir)

def plot_texmesh(model, triplane, reso, savedir):
    triplane_sdf = triplane[:, 0:3, ...]
    triplane_rgb = triplane[:, 3:, ...]
    batch_size = triplane_sdf.shape[0]
    with torch.no_grad():
        vertices, triangles = extract_geometry(
            [-1., -1., -1.], 
            [1., 1., 1.],
            resolution=reso,
            threshold=0.0,
            batch_size=batch_size,
            query_func=lambda pts: - model.forward_sdf(triplane_sdf, pts, geo_only=True)
        )
        vertices = vertices / (reso - 1.) * 2 - 1

        v = torch.from_numpy(vertices).float().cuda().unsqueeze(0)
        colors = model.forward_rgb(triplane_rgb, v).detach().cpu().numpy()
        # colors = np.concatenate([colors, np.ones([colors.shape[0], 1])], -1)
        colors = (colors.clip(0.0, 1.0) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=colors[0])
        mesh.export(savedir)


def extract_geometry(bound_min, bound_max, resolution, threshold, batch_size, query_func):
    u, _ = extract_fields(bound_min, bound_max, resolution, batch_size, query_func, 1)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    return vertices, triangles


def extract_fields(bound_min, bound_max, resolution, batch_size, query_func, channels=1):
    N = 64 # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, channels], dtype=np.float32)
    p = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(1, -1, 1), yy.reshape(1, -1, 1), zz.reshape(1, -1, 1)], dim=-1).cuda()
                    pts = pts.repeat([batch_size, 1, 1])
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs), -1).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
                    p[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = pts.reshape(len(xs), len(ys), len(zs), -1).detach().cpu().numpy()
    if channels == 1:
        u = u.squeeze(-1)
    return u, p