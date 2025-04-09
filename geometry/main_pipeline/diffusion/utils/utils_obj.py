import os
import numpy as np

mtl_string = """newmtl Default
Ns 604.153320
Ka 0.677273 0.677273 0.677273
Kd 0.761274 0.761274 0.761274
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.500000
d 1.000000
illum 3"""


def export_obj(verts: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Exports vertices and faces as an OBJ file.

    Parameters:
    - verts: numpy array of shape [n_verts, 3], representing the vertices.
    - faces: numpy array of shape [n_faces, 3], representing the face indices (0-based).
    - filename: str, the name of the output OBJ file.
    """
    mtl_filename = os.path.splitext(filename)[0] + ".mtl"
    with open(mtl_filename, 'w') as mtl_file:
        mtl_file.write(mtl_string)

    with open(filename, 'w') as file:
        rel_mtl_filename = os.path.basename(mtl_filename)
        file.write(f"mtllib {rel_mtl_filename}\n")

        # Write vertices
        for vert in verts:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        file.write(f"usemtl Default\n")

        # Write faces (OBJ format is 1-based indexing)
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def export_obj_gradio(verts: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Exports vertices and faces as an OBJ file.

    Parameters:
    - verts: numpy array of shape [n_verts, 3], representing the vertices.
    - faces: numpy array of shape [n_faces, 3], representing the face indices (0-based).
    - filename: str, the name of the output OBJ file.
    """
    path =  os.path.dirname(filename)
    name = os.path.basename(filename).split(".")[0]
    # mtl_filename = os.path.splitext(filename)[0] + ".mtl"
    mtl_filename = os.path.join(path, name + ".mtl")
    # print(mtl_filename)
    # breakpoint()
    with open(mtl_filename, 'w') as mtl_file:
        mtl_file.write(mtl_string)

    with open(filename, 'w') as file:
        rel_mtl_filename = os.path.basename(mtl_filename)
        file.write(f"mtllib {rel_mtl_filename}\n")

        # Write vertices
        for vert in verts:
            file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")

        file.write(f"usemtl Default\n")

        # Write faces (OBJ format is 1-based indexing)
        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
