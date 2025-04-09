import json
import numpy as np
import os
import scipy.spatial


def euler_to_rotation_matrix(euler_angles):
    r = scipy.spatial.transform.Rotation.from_euler(
        'xyz', euler_angles, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def read_json(json_path: str):
    with open(json_path, encoding='utf-8') as f:
        json_struct = json.load(f)
        return json_struct


def write_json(json_path: str, json_struct):
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_struct, f, indent=4, ensure_ascii=False)


def read_list(in_list_txt):
    str_list = []
    if not os.path.exists(in_list_txt):
        print('Cannot find input list txt file ', in_list_txt)
        return str_list

    with open(in_list_txt, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            mesh_path = line.strip()
            if len(mesh_path) > 1:
                str_list.append(mesh_path)
    return str_list


def write_list(path: str, write_list: list):
    with open(path, 'w') as f:
        for index in range(len(write_list)):
            f.write(write_list[index] + "\n")


def pose_generation(current_azimuth_list: list,
                    current_elevation_list: list,
                    current_fov_list: list,
                    image_size: int = 512):
    def opencv_to_blender(T):
        """T: ndarray 4x4
           usecase: cam.matrix_world =  world_to_blender( np.array(cam.matrix_world))
        """
        origin = np.array(((1, 0, 0, 0), (0, -1, 0, 0),
                           (0, 0, -1, 0), (0, 0, 0, 1)))
        return np.matmul(T, origin)  # T * origin

    def angles_to_unit_vectors(azimuth, elevation):
        """
        Converts azimuth and elevation angles to unit vector coordinates.

        Parameters:
        azimuth (numpy.ndarray): Array of azimuth angles in degrees. Shape: (n,)
        elevation (numpy.ndarray): Array of elevation angles in degrees. Shape: (n,)

        Returns:
        numpy.ndarray: Array of unit vector coordinates corresponding to the angles. Shape: (n, 3)
        """
        # Convert angles from degrees to radians
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        # Calculate Cartesian coordinates
        x = np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = np.sin(elevation_rad)

        # Combine into nx3 array
        vectors = np.stack((x, y, z), axis=-1)

        return vectors

    def look_at_opencv(camera_positions, target=np.array([0, 0, 0]), world_up=np.array([0, 0, 1])):
        """
        Generates camera-to-world transformation matrices for cameras following the OpenCV convention.

        Parameters:
        camera_positions (numpy.ndarray): An array of camera positions.
                                          Shape: (n_views, 3), where n_views is the number of camera views.
                                          Each row should be [x, y, z] coordinates of a camera position.
        target (numpy.ndarray): A 3D point that each camera is looking at. Defaults to the world origin [0, 0, 0].
                                Shape: (3,), representing the [x, y, z] coordinates of the target point.
        world_up (numpy.ndarray): The up direction in world coordinates. Defaults to [0, 0, 1].
                                  Shape: (3,), representing the [x, y, z] coordinates of the world's up vector.

        Returns:
        numpy.ndarray: An array of 4x4 transformation matrices corresponding to each camera position.
                       Shape: (n_views, 4, 4), where each 4x4 matrix is a transformation matrix for a camera.
        """
        matrices = []

        for pos in camera_positions:
            # Forward vector (Z-axis)
            forward = target - pos
            forward /= np.linalg.norm(forward)

            # Right vector (X-axis)
            right = np.cross(forward, world_up)
            right /= np.linalg.norm(right)

            # Camera down vector (Y-axis, pointing downwards in OpenCV convention)
            down = np.cross(forward, right)

            # Construct transformation matrix
            mat = np.eye(4)
            mat[0:3, 0] = right
            mat[0:3, 1] = down
            mat[0:3, 2] = forward
            mat[0:3, 3] = pos

            matrices.append(mat)

        return np.array(matrices)

    def get_grouped_intrinsic_matrix(fov: np.ndarray, image_size: int):
        """
        Calculate the intrinsic camera matrix for a square image.

        :param fov: Field of view in degrees.
        :param image_size: Width or height of the square image in pixels.
        :return: 3x3 numpy array representing the intrinsic camera matrix.
        """
        # Calculate the focal length in pixels
        focal_length_px = image_size / (2 * np.tan(np.radians(fov) / 2))

        # The principal point is typically at the center of the image
        cx = cy = image_size / 2

        intrinsic_matrices = []

        for f in focal_length_px:
            # Constructing the intrinsic matrix
            intrinsic_matrix = np.array([[f, 0, cx],
                                         [0, f, cy],
                                         [0, 0, 1]])
            intrinsic_matrices.append(intrinsic_matrix)
        return np.asarray(intrinsic_matrices)

    def get_radius(fov, bbox_size):
        '''
        fov: 1d array
        bbox_size: scalar or 1d array, size of the tight bounding box that covers unit sphere
                projected onto image plane, relative to image dimensions

        returns: radius, 1d array
        '''

        half_fov_rads = np.radians(fov / 2)
        radius = 1 / np.sin(np.arctan(np.tan(half_fov_rads) * bbox_size))
        return radius

    def gen_rotation_blender():
        return np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def generate_cam2world(radius, cams_azimuth, cams_elevation):
        """
        Roates a multi-camera system, returns the camera poses of each camera in each rotation

        Parameters:
        radius: array of radius Shape: (n_cams,)
        cams_azimuth (numpy.ndarray): Array of azimuth angles in degrees in a multi-camera system, reletaive to reference camera. Shape: (n_cams,)
        cams_elevation (numpy.ndarray): Array of elevation angles in degrees in a multi-camera system, reletaive to reference camera. Shape: (n_cams,)

        Returns:
        numpy.ndarray: Array of unit vector coordinates corresponding to the angles. Shape: (n_cam, 4, 4)
        """
        cam_locations = angles_to_unit_vectors(
            cams_azimuth, cams_elevation) * radius.reshape(-1, 1)

        cam2worlds = look_at_opencv(cam_locations)  # [n_cams, 4, 4]

        return cam2worlds

    size_ortho = 1 / 1.25

    intrinsics = []
    opencv_cam2worlds = []
    cam2worlds = []
    azimuth_ortho = np.array(current_azimuth_list)
    elevation_ortho = np.array(current_elevation_list)
    fov_ortho = np.array(current_fov_list)

    radius_ortho = get_radius(fov_ortho, size_ortho)
    intrinsic = get_grouped_intrinsic_matrix(fov_ortho, image_size)
    cam2world = generate_cam2world(radius_ortho, azimuth_ortho, elevation_ortho)

    rot = gen_rotation_blender()
    intrinsics.extend(intrinsic)
    opencv_cam2worlds.extend(rot @ cam2world)

    for pose in opencv_cam2worlds:
        # cam2worlds.append(opencv_to_blender(pose))
        cam2worlds.append(pose)

    return intrinsics, cam2worlds

if __name__ == "__main__":
    pose = pose_generation(current_azimuth_list = [45],
                        current_elevation_list = [30],
                        current_fov_list=[10],
                        image_size = 512)
    print(pose)