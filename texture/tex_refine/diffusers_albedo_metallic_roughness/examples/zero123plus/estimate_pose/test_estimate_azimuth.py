import glob
import os

import cv2
import json
import torch
import numpy as np
from tqdm.notebook import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

from utils_pose import pose_generation

root_dir = "/aigc_cfs_gdp/sz/result/general_generate_z123_v21_step37500_v2/d3e4d0cc-585e-46e1-a1f4-490798e4bf09/"
image_path = glob.glob(root_dir + "/*preprocess_z123.jpg")
image_mask_path = glob.glob(root_dir + "/*preprocess_z123_mask.png")
obj_path = os.path.join(root_dir, "obj_dir/obj_mesh_mesh.obj")

image_ref = cv2.imread(image_path[0], cv2.IMREAD_UNCHANGED)
image_ref = cv2.transpose(image_ref)
# cv2.imwrite("image_ref.png", image_ref)
image_ref = image_ref[:, :, ::-1]

image_mask = cv2.imread(image_mask_path[0], cv2.IMREAD_UNCHANGED)
image_mask = cv2.transpose(image_mask)
# cv2.imwrite("image_mask.png", image_mask)
image_mask = image_mask[..., None]

image_ref = np.concatenate([image_ref, image_mask], axis=-1)
image_ref = cv2.resize(image_ref, dsize=[512, 512], interpolation=cv2.INTER_CUBIC)
image_ref = image_ref / 255.0

image_ref = image_ref[None, ...]

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(obj_path)
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize a perspective camera.
cameras = FoVPerspectiveCameras(fov=10, znear=0.1, device=device)

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)


# We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

# Select the viewpoint using spherical angles  
distance = 14.322518   # distance from camera to the object
elevation = 0.0   # angle of elevation in degrees
azimuth = 90.0  # No rotation so the camera is positioned on the +Z axis. 

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

# # Render the teapot providing the values of R and T. 
# silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
# image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)

# silhouette = silhouette.cpu().numpy()
# image_ref = image_ref.cpu().numpy()

plt.imsave('sihouette.png', image_ref.squeeze()[..., 3])
plt.imsave("image_ref.png", image_ref.squeeze()[..., :3])


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        self.distance = 14.322518 
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref = torch.from_numpy((image_ref[..., 3]).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        # self.camera_position = nn.Parameter(
            # torch.from_numpy(np.array([3.0,  6.9, +2.5], dtype=np.float32)).to(meshes.device))
        # self.camera_azimuth = nn.Parameter(torch.tensor([90], dtype=torch.float32)).to(meshes.device)
        self.camera_azimuth = nn.Parameter(torch.from_numpy(np.array([90], dtype=np.float32)).to(meshes.device))
        # self.camera_elevation = nn.Parameter(torch.tensor(0, dtype=torch.float32)).to(meshes.device)


    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        # R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        
        R, T = look_at_view_transform(self.distance, 0.0, self.camera_azimuth, device=self.meshes.device)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # plt.imsave("image.png", image[..., 3].detach().cpu().numpy().squeeze())
        # plt.imsave("image_ref1.png", self.image_ref.cpu().numpy().squeeze())
        # breakpoint()

        # Calculate the silhouette loss
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image

# We will save images periodically and compose them into a GIF.
filename_output = "./result_optimization_demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)


# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


_, image_init = model()

# plt.imsave("image_init.png", image_init.detach().squeeze().cpu().numpy()[..., 3])
# plt.imsave("image_ref.png", model.image_ref.cpu().numpy().squeeze())

loop = tqdm(range(200))
for i in loop:
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()
    
    loop.set_description('Optimizing (loss %.4f)' % loss.data)
    
    if loss.item() < 200:
        break
    
    # Save outputs to create a GIF. 
    if i % 10 == 0:
        R, T = look_at_view_transform(model.distance, 0.0, model.camera_azimuth, device=model.meshes.device)
        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        # image_silhouette = silhouette_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        # image_silhouette = image_silhouette.detach().squeeze().cpu().numpy()[..., 3:]

        # image_bg = image_ref.copy()[0, ..., :3]
        # image_bg = image_bg * (1 - image_silhouette) + image * image_silhouette
        # image = image_bg
        
        image = img_as_ubyte(image)
        writer.append_data(image)
        print("iter: {}, loss: {}".format(i, loss.data))

target_elevation = 90.0 - model.camera_azimuth.detach().cpu().numpy()[0]
elevation_azimuth_save_path = os.path.join(root_dir, "preprocess_z123_cam_elevation_azimuth.json")
elevation_azimuth = {"elevation": target_elevation,
                     "azimuth": 0}
with open(elevation_azimuth_save_path, "w") as fw:
    json.dump(elevation_azimuth, fw, indent=2)

# pose = pose_generation(current_azimuth_list = [0],
#                     current_elevation_list = [target_elevation],
#                     current_fov_list=[10],
#                     image_size = 512)
# pose_save_path = os.path.join(root_dir, "preprocess_z123_cam_pose.npy")
# np.save(pose_save_path, pose)
# print(target_elevation)
# print(pose[1])

writer.close()