from traceback import print_stack
import trimesh
import pysdf
import numpy as np
from pdb import set_trace as st
import warnings
try:
    import pyembree
    from pyembree import rtcore_scene
except:
    warnings.warn("could not load pyembree, this would cause error if visibilities are to be computed.")
from tqdm import tqdm


def query_sdf(queries, mesh_verts, mesh_faces, chunk_size=None, sdf_query_func=None, show_progress=False):
    n_pts = queries.shape[0]
    
    if sdf_query_func is None:
        sdf_query_func = pysdf.SDF(mesh_verts, mesh_faces)
    
    if chunk_size and chunk_size < n_pts:
        sdf_chunks = []
        
        if show_progress:
            prog = tqdm
        else:
            prog = lambda x: x
            
        for queries_chunk in prog(np.array_split(queries, n_pts//chunk_size, axis=0)):
            sdf_chunk = query_sdf(queries_chunk, None, None, sdf_query_func=sdf_query_func)
            sdf_chunks.append(sdf_chunk)
        return np.concatenate(sdf_chunks, axis=0)
    
    return sdf_query_func(queries)
    
def intersects_any(ray_org, ray_dir, scene):
    rtres = scene.run(ray_org.astype(np.float32), ray_dir.astype(np.float32), query='OCCLUDED')
    return rtres >= 0 # (n_rays)

def sample_unit_3dvec(n, seed=None):
    if seed:
        np.random.seed(seed)
    vec = np.random.randn(n,3).astype(np.float32)
    return vec / np.linalg.norm(vec, ord=2, axis=-1, keepdims=True)

def sample_visibility(pts, mesh_verts, mesh_faces, n_rays=128, chunk_size=None, rays=None, scene=None, show_progress=False):
    '''
    monte carlo visibility by shooting rays in random directions and test whether they are blocked
    
    pts: (n_pts, 3)
    mesh_verts: (n_vert, 3)
    mesh_faces: (n_face, 3)
    rays: random ray directions. if provided, can be array of shape (n_rays, 3) or (n_pts, n_rays, 3), 
          and n_rays and sample_per_pt will be ignored.
    scene: an embree rtcore scene object, if provided mesh_verts and mesh_faces will be ignored
    '''
    
    if scene is None:
        scene = rtcore_scene.EmbreeScene()
        mesh = pyembree.mesh_construction.TriangleMesh(scene, mesh_verts[mesh_faces].astype(np.float32))
    
    n_pts = pts.shape[0]
    
        
    if chunk_size and chunk_size < n_pts:
        vis_chunks = []
        if show_progress:
            prog = tqdm
        else:
            prog = lambda x: x
            
        if rays is not None and rays.ndim == 3:
            for pts_chunk, rays_chunk in zip(prog(np.array_split(pts, n_pts//chunk_size, axis=0)), np.array_split(rays, n_pts//chunk_size, axis=0)):
                vis_chunk = sample_visibility(pts_chunk, None, None, chunk_size=None, rays=rays_chunk, scene=scene)
                vis_chunks.append(vis_chunk) 
        else:
            for pts_chunk in prog(np.array_split(pts, n_pts//chunk_size, axis=0)):
                vis_chunk = sample_visibility(pts_chunk, None, None, n_rays=128, chunk_size=None, rays=rays, scene=scene)
                vis_chunks.append(vis_chunk) 
        return np.concatenate(vis_chunks, axis=0)
    
    if rays is None:
        rays = sample_unit_3dvec(n_rays) # (n_rays, 3)

    rays = np.zeros_like(pts).reshape(-1,1,3) + rays # (n_pts, n_rays, 3)
    n_rays = rays.shape[1]
    rays = np.reshape(rays, (-1,3)) # (n_pts*n_rays, 3)
    
    origins = np.repeat(pts, n_rays, axis=0) # (n_pts*n_rays, 3)
    
    pts_blocked = intersects_any(origins, rays, scene).reshape(n_pts, n_rays) # (n_pts, n_rays)
    
    visibility = 1.0 - pts_blocked.mean(axis=1) # (n_pts)
    
    return visibility.astype(np.float32)

def visibility_grid(mesh_verts, mesh_faces, grid_res=256, n_rays=128):
    pts = np.stack(np.meshgrid(
        np.linspace(-1,1,grid_res),
        np.linspace(-1,1,grid_res),
        np.linspace(-1,1,grid_res),
        indexing='ij'), axis=-1
    ) # (grid_res, grid_res, grid_res, 3)
    
    pts = pts.reshape(-1,3)
    
    return sample_visibility(pts, mesh_verts, mesh_faces, n_rays, chunk_size=65536, show_progress=True).reshape(grid_res,grid_res,grid_res)

def sdf_grid(mesh_verts, mesh_faces, grid_res=256):
    pts = np.stack(np.meshgrid(
        np.linspace(-1,1,grid_res),
        np.linspace(-1,1,grid_res),
        np.linspace(-1,1,grid_res),
        indexing='ij'), axis=-1
    ) # (grid_res, grid_res, grid_res, 3)
    
    pts = pts.reshape(-1,3)
    
    return query_sdf(pts, mesh_verts, mesh_faces, chunk_size=65536, show_progress=True).reshape(grid_res,grid_res,grid_res)

def load_mesh(mesh_path, transformation_path):
    
    manifold_mesh = trimesh.load_mesh(mesh_path)
    
    # yup2zup = np.array([
    #     [1,0,0,0],
    #     [0,0,-1,0],
    #     [0,1,0,0],
    #     [0,0,0,1]
    # ])
    # if transformation_path is not None:
    #     transform = yup2zup @ np.loadtxt(transformation_path) 
    # else:
    #     transform = yup2zup
    transform = np.loadtxt(transformation_path)

    manifold_mesh.apply_transform(transform)
    verts = manifold_mesh.vertices 
    faces = manifold_mesh.faces
    
    return verts, faces

def sample_surface(verts, faces, n_pts=10000, only_points=False):
    '''
    sample points on the surface of a mesh
    '''
    mesh = trimesh.Trimesh(verts, faces, validate=False, process=False)
    verts = mesh.vertices
    faces = mesh.faces
    vnorm = mesh.vertex_normals
    fnorm = mesh.face_normals
    
    face_areas = np.linalg.norm(np.cross(verts[faces[:,1]]-verts[faces[:,0]], verts[faces[:,2]]-verts[faces[:,0]]), ord=2, axis=-1) # (n_face)
    n_faces = np.random.choice(faces.shape[0], size=n_pts, replace=True, p=face_areas/face_areas.sum()) # (n_pts) in range [0, n_face)
    
    barycentric = np.random.random((n_pts, 3)) + 1e-6
    barycentric = barycentric / barycentric.sum(axis=-1, keepdims=True) # (n_pts, 3)
    
    pts = np.sum(verts[faces[n_faces]] * barycentric[...,None], axis=1) # (n_pts, 3)

    if not only_points:
        nrms = np.sum(vnorm[faces[n_faces]] * barycentric[...,None], axis=1) # (n_pts, 3)
        return pts, nrms, n_faces, barycentric
    return pts

def sample_near_surface(verts, faces, std, n_pts=10000):
    pts, nrms, n_faces, barycentric = sample_surface(verts, faces, n_pts)
    perturb = np.random.randn(n_pts, 1) * std
    return pts + perturb * nrms


def sample_pcd(mesh_path, transformation_path, n_surface_pts, n_near_surface_pts, n_space_pts, near_surface_std, n_visibility_random_samples=128):
    
    verts, faces = load_mesh(mesh_path, transformation_path)
    
    surface_pts, surface_norms, *_ = sample_surface(verts, faces, n_surface_pts)
    near_surface_pts = sample_near_surface(verts, faces, near_surface_std, n_near_surface_pts)
    space_pts = np.random.random((n_space_pts, 3)) * 2 - 1
    
    non_surace_pts = np.concatenate([near_surface_pts, space_pts], axis=0)
    non_surface_sdf = query_sdf(non_surace_pts, verts, faces, chunk_size=65536, show_progress=False)
    non_surface_visibility = sample_visibility(non_surace_pts, verts, faces, n_rays=n_visibility_random_samples, chunk_size=65536, show_progress=False)
    
    near_surface_sdf, space_sdf = np.split(non_surface_sdf, [near_surface_pts.shape[0]])
    near_surface_visibility, space_visibility = np.split(non_surface_visibility, [near_surface_pts.shape[0]])
    
    return surface_pts.astype(np.float32), surface_norms.astype(np.float32), near_surface_pts.astype(np.float32), near_surface_sdf.astype(np.float32), \
        near_surface_visibility.astype(np.float32), space_pts.astype(np.float32), space_sdf.astype(np.float32), space_visibility.astype(np.float32)

def sample_surface_from_path(mesh_path, transformation_path, n_surface_pts):
    verts, faces = load_mesh(mesh_path, transformation_path)
    
    surface_pts = sample_surface(verts, faces, n_surface_pts, only_points=True)

    return surface_pts.astype(np.float32)


if __name__ == "__main__":
    import json
    import open3d as o3d
    json_path = "/aigc_cfs_2/neoshang/data/data_list/20240808/part14_10w_color_normal_pc.json"
    with open(json_path, 'r') as fr:
        json_dict = json.load(fr)["data"]

    for classname, classdict in json_dict.items():
        for objname, objdict in classdict.items():
            mesh_path = objdict["Manifold"]
            transformation_path = objdict["Transformation"]

            surface_pts, surface_norms, \
            near_surface_pts, near_surface_sdf, near_surface_visibility, \
            space_pts, space_sdf, space_visibility = sample_pcd(
                mesh_path, transformation_path, n_surface_pts=100000, 
                n_near_surface_pts=100000, n_space_pts=100000, near_surface_std=0.05, 
                n_visibility_random_samples=128)
            
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(surface_pts)
            o3d.io.write_point_cloud("surface_pts.ply", point_cloud)

            point_cloud.points = o3d.utility.Vector3dVector(near_surface_pts)
            o3d.io.write_point_cloud("near_surface_pts.ply", point_cloud)

            point_cloud.points = o3d.utility.Vector3dVector(space_pts)
            o3d.io.write_point_cloud("space_pts.ply", point_cloud)

            breakpoint()