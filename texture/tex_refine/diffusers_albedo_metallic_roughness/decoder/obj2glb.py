import os
import sys

import bpy
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument(
    "--output_obj_path",
    type=str,
    required=True,
    help="Path to the output glb file",
)
parser.add_argument(
    "--output_glb_path",
    type=str,
    required=True,
    help="Path to the output glb file",
)

parser.add_argument("--rotate", action="store_true", 
                    help="if rotate obj")

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入OBJ文件
bpy.ops.import_scene.obj(filepath=args.input_path)

# 获取导入的所有物体
imported_objs = bpy.context.selected_objects


if args.rotate:
    # 设置旋转角度（弧度制）
    rotation_angle = math.radians(-90)

    # 遍历所有导入的物体并旋转
    for obj in imported_objs:
        # 绕X轴旋转90度
        obj.rotation_euler[0] += rotation_angle
        # 应用旋转，使其成为物体的一部分
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# 选择所有导入的物体
for obj in imported_objs:
    obj.select_set(True)

bpy.ops.export_scene.obj(filepath=args.output_obj_path, use_selection=True)
bpy.ops.export_scene.gltf(filepath=args.output_glb_path, export_format='GLB', use_selection=True)