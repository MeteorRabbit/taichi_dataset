import bpy
import os
import sys
import math
import json
import numpy as np
from mathutils import Vector

# --- 配置 ---
# 基础路径
# BASE_DIR = "D:/Experiments/gic/taichi_dataset"
if sys.platform == "win32":
    BASE_DIR = "D:/Experiments/gic/taichi_dataset"
else:
    BASE_DIR = "/root/workspace/taichi_dataset"

# 输入输出路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_curling")
OUTPUT_DIR = os.path.join(BASE_DIR, "render_output/curling")

# 默认渲染开关
SHOULD_RENDER_DEFAULT = False

# 渲染设置
RENDER_ENGINE = 'CYCLES'
SAMPLES = 128

# 定义对象映射：Key 是 Blender 场景中的对象名，Value 是 PLY 文件的前缀
# 这样我们可以支持多个独立的物体序列
OBJECTS_CONFIG = {
    "Curling_Thrower": {
        "prefix": "curling_0",
        "material_name": "Curling_Stone", # Ensure this matches your Blender material name
        "color": (0.8, 0.1, 0.1, 1) 
    },
    "Curling_Target": {
        "prefix": "curling_1",
        "material_name": "Curling_Stone", # Ensure this matches your Blender material name
        "color": (0.1, 0.1, 0.8, 1)
    }
}

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    # 停止动画
    if bpy.context.screen:
        bpy.ops.screen.animation_cancel(restore_frame=False)

    # 选中所有物体并删除
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 清理未使用的 Mesh 数据块
    for block in bpy.data.meshes:
        if not block.users: bpy.data.meshes.remove(block)

def ensure_material(mat_name, color):
    """确保材质存在"""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        # Only create if it really doesn't exist
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Roughness'].default_value = 0.4
            bsdf.inputs['Metallic'].default_value = 0.1
    return mat

def setup_cameras():
    # 使用 Fibonacci Sphere 生成均匀视点，或者针对冰壶使用特定视点
    # 这里沿用通用的 10 个均匀视角
    def fibonacci_sphere(samples=10):
        points = []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(samples):
            z = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - z * z)
            theta = phi * i
            x = math.cos(theta) * radius
            y = abs(math.sin(theta) * radius)
            if y < 0.1: y = 0.1
            points.append((x, y, z))
        return points

    sphere_points = fibonacci_sphere(10)
    cameras = []
    
    # 根据场景范围调整相机距离
    center_offset = Vector((0.5, 0.2, 0.5)) # 假设中心
    r = 2.5 # 距离
    
    for i, point in enumerate(sphere_points):
        scaled_point = Vector((point[0]*r, point[1]*r, point[2]*r)) + center_offset
        bpy.ops.object.camera_add(location=scaled_point)
        cam = bpy.context.active_object
        cam.name = f"Camera_{i+1}"
        
        direction = center_offset - scaled_point
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()
        cam.data.lens = 50
        cameras.append(cam)
    return cameras

def setup_lighting():
    # Sun
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 10, 5)
    light_data.energy = 5.0
    
    # Area
    area_data = bpy.data.lights.new(name="Area", type='AREA')
    area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (-2, 4, 3)
    area_data.energy = 300.0
    area_data.size = 5.0
    
    # Environment
    if not bpy.context.scene.world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes['Background']
        bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)

def get_intrinsic(cam):
    focal_px = 965.6844046797067 # Default for 800x800 50mm
    return [[focal_px, 0.0, 400.0], [0.0, focal_px, 400.0], [0.0, 0.0, 1.0]]

def get_c2w(cam):
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

def load_ply_frame(frame_idx):
    """
    加载指定帧的所有 PLY 文件并更新场景对象
    """
    if frame_idx < 0: return

    for obj_name, config in OBJECTS_CONFIG.items():
        prefix = config["prefix"]
        filename = f"{prefix}_frame_{frame_idx:04d}.ply"
        path = os.path.join(INPUT_DIR, filename)
        
        # 1. 如果场景中已经有这个名字的物体，先删除它
        if obj_name in bpy.data.objects:
            old_obj = bpy.data.objects[obj_name]
            bpy.data.objects.remove(old_obj, do_unlink=True)
        
        # 2. 检查新文件是否存在
        if not os.path.exists(path):
            continue

        # 3. 导入新的 PLY
        # 确保不选其他物体
        bpy.ops.object.select_all(action='DESELECT')
        
        try:
            bpy.ops.wm.ply_import(filepath=path)
        except:
            # Fallback for older Blender versions if needed
            try: bpy.ops.import_mesh.ply(filepath=path)
            except: pass

        if bpy.context.selected_objects:
            imported = bpy.context.selected_objects[0]
            imported.name = obj_name
            bpy.ops.object.shade_smooth()
            
            mat = ensure_material(config["material_name"], config["color"])
            if imported.data.materials:
                imported.data.materials[0] = mat
            else:
                imported.data.materials.append(mat)

def main(should_render=False):
    print(f"Curling Rendering Started. Mode: {'RENDER' if should_render else 'PREVIEW'}")
    
    clean_scene()
    cameras = setup_cameras()
    setup_lighting()
    
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.cycles.samples = SAMPLES
    scene.cycles.device = 'GPU'
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.image_settings.file_format = 'PNG'
    
    # Determine frame count
    max_frame = 30
    if os.path.exists(INPUT_DIR):
        prefix = list(OBJECTS_CONFIG.values())[0]["prefix"]
        files = [f for f in os.listdir(INPUT_DIR) if f.startswith(prefix) and f.endswith(".ply")]
        if files:
            max_frame = max([int(f.split('_')[-1].split('.')[0]) for f in files])
    
    scene.frame_start = 0
    scene.frame_end = max_frame
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Preview Mode ---
    if not should_render:
        print("Preview Mode: Loading Frame 0...")
        load_ply_frame(0)
        scene.frame_set(0)
        print(f"Setup complete. Frames detected: {max_frame+1}")
        return

    # --- Render Mode ---
    all_data = []
    fps = 24
    
    for frame in range(0, max_frame + 1):
        print(f"Rendering Frame {frame}/{max_frame}...")
        
        load_ply_frame(frame)
        scene.frame_set(frame)
        time = frame / fps
        
        for i, cam in enumerate(cameras):
            scene.camera = cam
            filename = f"r_{i}_{frame}.png"
            scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            
            bpy.ops.render.render(write_still=True)
            
            all_data.append({
                "file_path": f"./curling/{filename}",
                "time": time,
                "c2w": get_c2w(cam),
                "intrinsic": get_intrinsic(cam)
            })

    json_path = os.path.join(OUTPUT_DIR, "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Done! Metadata saved to {json_path}")

if __name__ == "__main__":
    is_render = SHOULD_RENDER_DEFAULT or "--render" in sys.argv
    
    def frame_change_handler(scene):
        # Interactive update
        load_ply_frame(scene.frame_current)
            
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_handler)
    
    main(should_render=is_render)
