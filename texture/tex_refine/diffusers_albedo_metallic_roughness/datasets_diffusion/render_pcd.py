import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
import trimesh
from torchvision import utils as vutils
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    OrthographicCameras,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    look_at_view_transform,
)

def pcd2ply(pcd_path, save_path):
    xyz = np.load(pcd_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(save_path, pcd)

def mesh2pcd(mesh_path, pcd_save_path, cam_k_pose):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    xyz = np.asarray(pcd.points)
    np.save(pcd_save_path, xyz)
    breakpoint()

def render_pcd_with_pytorch3d(camk, cam_pose, pcd, save_path, color=None, ortho_cam=False):
    # # get campose in pytorch3d coordinate
    # pose = np.linalg.inv(cam_pose)
    # R = pose[:3, :3][None, ...]
    # T = pose[:3, 3][None, ...]
    # R = R.transpose((0, 2, 1))
    # R[:, :, :2] *= -1
    # T[:, :2] *= -1

    R, T = look_at_view_transform(dist = 3.0,
                        elev = 0.0,
                        azim = 0.0,
                        degrees = True,
                        eye = None,
                        at=((0, 0, 0),),  # (1, 3)
                        up=((0, 1, 0),),  # (1, 3)
                        device = "cpu",)

    image_size = 1024
    image_size_cam = torch.from_numpy(np.array([[image_size, image_size]]).astype(np.float32))


    if ortho_cam:
        cameras = FoVOrthographicCameras(
            znear = 0.1,
            zfar = 100.0,
            max_y = 1.0,
            min_y = -1.0,
            max_x= 1.0,
            min_x = -1.0,
            # scale_xyz=((0.92, 0.92, 0.92),),  # (1, 3)
            scale_xyz = ((1.0, 1.0, 1.0),),
            R=torch.tensor(R),
            T=torch.tensor(T),
            K = None,
            device= "cuda",
        )

    else:
        """
        ### perspectiveCameras
        """
        down_scale = 1.0
        focal_length = torch.from_numpy(np.array([[camk[0, 0]*down_scale,camk[1, 1]*down_scale]]).astype(np.float32))
        principal_point = torch.from_numpy(np.array([[camk[0, 2]*down_scale, camk[1, 2]*down_scale]]).astype(np.float32))

        cameras = PerspectiveCameras(
                    focal_length=focal_length,
                    principal_point=principal_point,
                    R=torch.tensor(R),
                    T=torch.tensor(T),
                    device= "cuda",
                    in_ndc= False,
                    image_size=image_size_cam
                )

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = 0.003,
        points_per_pixel = 10
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    ## load point cloud and corresponding point color
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    verts = torch.Tensor(pcd).to(device)
    if color is None:
        color = np.ones_like(pcd)
        color[:, 0:2] = 0
    rgb = torch.Tensor(color).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])

    images = renderer(point_cloud).permute(0, 3, 1, 2)
    vutils.save_image(images, save_path)