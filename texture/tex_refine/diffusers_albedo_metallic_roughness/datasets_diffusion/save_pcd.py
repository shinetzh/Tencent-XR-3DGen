import open3d as o3d
import numpy as np

pcd_path = "/apdcephfs_cq8/share_2909871/Assets/objaverse/render/3d_diffusion/unify_20240730/part3/proc_data/pod_12/objaverse/ad99670274254e4aa539a90a5dbdb24e/proc_data/geometry/surface_point_500000.npy"
pcd_points = np.load(pcd_path)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pcd_points)
o3d.io.write_point_cloud("cond_pcd.ply", point_cloud)