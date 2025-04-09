import numpy as np

x_rotate90 = np.array([[1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],])

x_rotate_inv90 = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1],])

x_rotate_180 = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1],])

y_rotate_180 = np.array([[-1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1],])

z_rotate_180 = np.array([[-1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],])


space_x_rotate90 = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, -1, 0, 0],
                             [0, 0, 0, 1],])

space_y_rotate90 = np.array([[0, 0, -1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1],])

space_z_rotate90 = np.array([[0, 1, 0, 0],
                             [-1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],])


# pose0 = np.array([[1, 0, 0, 0],
#                   [0, 0, 1, -3],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1],])

# rotate_m = y_rotate_180 @ x_rotate_180
# print(rotate_m)

# pose_torch3d = y_rotate_180 @ x_rotate_180 @ pose0
# print(pose_torch3d)


# pose0 = np.array([[1, 0, 0, 0],
#                   [0, 0, 1, -3],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1],])

# pose1 = np.array([[0, 0, -1, 3],
#                   [1, 0, 0, 0],
#                   [0, -1, 0, 0],
#                   [0, 0, 0, 1],])

# pose1_inv = np.linalg.inv(pose1)
# pcd_transpose_matrix = pose1_inv @ pose0
# print(pcd_transpose_matrix)

pose_zup0 = np.array([[1, 0, 0, 0],
                      [0, 0, 1, -3],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1],])

pose_zup1 = np.array([[0, 0, -1, 3],
                      [1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1],])

pose_zup2 = np.array([[-1, 0, 0, 0],
                      [0, 0, -1, 3],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1],])

pose_zup3 = np.array([[0, 0, 1, -3],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1],])



pose_yup0 = np.array([[-1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, -1, 3],
                      [0, 0, 0, 1],])

pose_yup1 = np.array([[0, 0, 1, 3],
                      [0, 1, 0, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1],])

pose_yup2 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, -3],
                      [0, 0, 0, 1],])

pose_yup3 = np.array([[0, 0, -1, -3],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],])



# rotate_m @ pose_zup0 = pose_yup0
rotate_m = pose_yup0 @ np.linalg.inv(pose_zup0)
print(rotate_m)

pose_yup0_1 = rotate_m @ pose_zup0
pose_yup1_1 = rotate_m @ pose_zup1
pose_yup2_1 = rotate_m @ pose_zup2
pose_yup3_1 = rotate_m @ pose_zup3

pose_yup0_1[0, -1] = -pose_yup0_1[0, -1]
pose_yup1_1[0, -1] = -pose_yup1_1[0, -1]
pose_yup2_1[0, -1] = -pose_yup2_1[0, -1]
pose_yup3_1[0, -1] = -pose_yup3_1[0, -1]

print(pose_yup0_1)
print(pose_yup1_1)
print(pose_yup2_1)
print(pose_yup3_1)

print(np.linalg.inv(pose_zup0))