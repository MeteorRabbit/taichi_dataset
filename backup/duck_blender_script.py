"""
渲染脚本（Blender）

用途：生成 toyduck 数据集格式：
 - 10 个相机视角（索引 0..9），
 - 每个视角 14 帧动画（frame 0..13）和 1 张背景帧（frame -1），
 - 输出图片到 `toyduck/data/`，并写入 `toyduck/all_data.json`，格式与其他 `data/*/all_data.json` 保持一致。

使用方法（示例，命令行在 Blender 安静模式下运行）：
 blender -b your_scene.blend -P render_script.py

注意：脚本假设场景中包含名为 `rubber_duck_toy` 的物体作为主体。
"""

import bpy
import math
import json
import os
import numpy as np
from mathutils import Vector


# 设置渲染路径和参数
render_path = "D:/Experiments/gic/toyduck/data"
# ensure output dir exists
os.makedirs(render_path, exist_ok=True)

bpy.context.scene.render.filepath = render_path + "/"
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 800
bpy.context.scene.render.resolution_percentage = 100
# frames 0..13 (14 frames) and a background frame -1
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 13
fps = 24

duck = bpy.data.objects.get("rubber_duck_toy")
if not duck:
    raise Exception("error1");

# 创建黄金螺旋分布的相机位置，确保Z坐标不为0
def fibonacci_sphere(samples=10):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # 黄金角度
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y从1到-1
        radius = math.sqrt(1 - y * y)  # 半径在x-z平面
        
        theta = phi * i  # 黄金角度增量
        
        x = math.cos(theta) * radius
        z = abs(math.sin(theta) * radius)
        
        # 确保Z坐标不为0
        if z < 0.1:
            z = 0.1
        
        points.append((x, y, z))
    
    return points

# 生成10个均匀分布的点
sphere_points = fibonacci_sphere(10)
cameras = []

# 创建相机
for i, point in enumerate(sphere_points):
    # 缩放点到半径为0.8的球面
    r = 0.8
    scaled_point = Vector((point[0]*r, point[1]*r, point[2]*r))
    
    # 创建相机
    bpy.ops.object.camera_add(location=scaled_point)
    cam = bpy.context.active_object
    cam.name = f"Camera_{i+1}"
    
    # 使相机朝向(0,0,1)
    direction = Vector((0, 0, 0.1)) - scaled_point
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    
    cameras.append(cam)

# 设置相机参数
focal_length = 50
sensor_width = 36
sensor_height = 36
principal_point_x = 400
principal_point_y = 400

for cam in cameras:
    cam.data.lens = focal_length
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

# 计算内参矩阵
def get_intrinsic(cam):
    # Use exact intrinsics consistent with existing datasets (800x800, principal point 400, focal_px fixed)
    focal_px = 965.6844046797067
    intrinsic = [
        [focal_px, 0.0, 400.0],
        [0.0, focal_px, 400.0],
        [0.0, 0.0, 1.0]
    ]
    return intrinsic

# 计算相机到世界矩阵
#def get_c2w(cam):
#    matrix = cam.matrix_world
#    R = matrix.to_3x3()
#    T = matrix.translation
#    
#    # 转换为标准相机坐标系
#    c2w = [
#        [R[0][0], R[0][2], -R[0][1], T.x],
#        [R[2][0], R[2][2], -R[2][1], T.z],
#        [-R[1][0], -R[1][2], R[1][1], -T.y]
#    ]
#    return c2w


def get_c2w(cam):
    # 获取 Blender 世界矩阵
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

# 准备渲染和JSON数据
all_data = []

# 首先渲染每个相机的特殊帧（鸭子不可见）
for i, cam in enumerate(cameras):
    # 设置活动相机
    bpy.context.scene.camera = cam

    # 隐藏鸭子
    duck.hide_render = True

    # 设置输出文件名（index 0..9）
    bpy.context.scene.render.filepath = f"{render_path}/r_{i}_-1.png"

    # 渲染
    bpy.ops.render.render(write_still=True)

    # 计算时间（使用-1/fps） 表示背景帧
    time = -1.0 / fps

    # 收集数据
    file_path = f"./data/r_{i}_-1.png"
    c2w = get_c2w(cam)
    intrinsic = get_intrinsic(cam)

    data = {
        "file_path": file_path,
        "time": time,
        "c2w": c2w,
        "intrinsic": intrinsic
    }

    all_data.append(data)

    # 恢复鸭子可见性
    duck.hide_render = False

# 然后渲染正常帧（0-13，共14帧）
for frame in range(0, 14):
    bpy.context.scene.frame_set(frame)
    time = frame / fps

    for i, cam in enumerate(cameras):
        # 设置活动相机
        bpy.context.scene.camera = cam

        # 设置输出文件名
        bpy.context.scene.render.filepath = f"{render_path}/r_{i}_{frame}.png"

        # 渲染
        bpy.ops.render.render(write_still=True)

        # 收集数据
        file_path = f"./data/r_{i}_{frame}.png"
        c2w = get_c2w(cam)
        intrinsic = get_intrinsic(cam)

        data = {
            "file_path": file_path,
            "time": time,
            "c2w": c2w,
            "intrinsic": intrinsic
        }

        all_data.append(data)

# 写入JSON文件
all_data_path = os.path.join(os.path.dirname(render_path), "all_data.json")
with open(all_data_path, "w") as f:
    json.dump(all_data, f, indent=4)

# 简单验证信息
print(f"已导出 {len(all_data)} 条记录到 {all_data_path}")
expected = 10 * (14 + 1)  # 10 cameras * (14 frames + 1 background)
if len(all_data) != expected:
    print(f"警告：导出记录数 ({len(all_data)}) != 预期 ({expected})，请检查脚本或重试渲染。")
else:
    print(f"渲染和JSON导出完成！数据保存在: {render_path}, all_data.json: {all_data_path}")