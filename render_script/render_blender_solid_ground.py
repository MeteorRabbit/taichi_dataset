import bpy
import numpy as np
import os
import sys
import math
import json
from mathutils import Vector

# --- 配置 ---
# 基础路径
if sys.platform == "win32":
    BASE_DIR = "D:/Experiments/gic/taichi_dataset"
else:
    BASE_DIR = "/root/workspace/taichi_dataset"

# 默认渲染开关 (方便在 IDE 中直接修改运行)
SHOULD_RENDER_DEFAULT = False

# 输入输出路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_solid_ground")
OUTPUT_DIR = os.path.join(BASE_DIR, "render_output/solid_ground")

# 材质配置
MESH_CONFIG = {
    "material_name": "DuckMaterial", # Should match Blender material
    "color": (1.0, 0.8, 0.0, 1.0)
}

# 渲染设置
RENDER_ENGINE = 'CYCLES'
SAMPLES = 128

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    if bpy.context.screen:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if not block.users: bpy.data.meshes.remove(block)

def ensure_material(mat_name, color):
    """确保材质存在"""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
    return mat

def setup_cameras():
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
    center_offset = Vector((0.5, 0.5, 0.5)) 
    r = 2.0 
    
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
        bg_node = world.node_tree.nodes.get('Background')
        if bg_node:
            bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)

def get_intrinsic(cam):
    focal_px = 965.6844046797067 
    return [[focal_px, 0.0, 400.0], [0.0, focal_px, 400.0], [0.0, 0.0, 1.0]]

def get_c2w(cam):
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

def load_ply_frame(frame_idx):
    """
    加载指定帧的 PLY 文件并更新场景对象
    """
    filename = f"frame_{frame_idx:04d}.ply"
    path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(path):
        return

    obj_name = "SimulationMesh"
    
    # 1. 删除旧对象
    if obj_name in bpy.data.objects:
        old_obj = bpy.data.objects[obj_name]
        bpy.data.objects.remove(old_obj, do_unlink=True)
            
    # 2. 导入新 PLY
    try:
        bpy.ops.import_mesh.ply(filepath=path)
    except AttributeError:
        # Blender 4.0+ uses bpy.ops.wm.ply_import
        # Fallback just in case
        try: bpy.ops.wm.ply_import(filepath=path)
        except: pass
         
    # 3. 重命名导入的对象
    if bpy.context.selected_objects:
        imported_obj = bpy.context.selected_objects[0]
        imported_obj.name = obj_name
        
        # 赋予材质
        mat_name = MESH_CONFIG["material_name"]
        color = MESH_CONFIG["color"]
        mat = ensure_material(mat_name, color)
        
        if imported_obj.data.materials:
            imported_obj.data.materials[0] = mat
        else:
            imported_obj.data.materials.append(mat)
        
        try:
             bpy.ops.object.shade_smooth()
        except:
             pass

def main(should_render=False):
    print(f"Solid Ground Rendering Started. Mode: {'RENDER' if should_render else 'PREVIEW'}")
    print(f"Input Directory: {INPUT_DIR}")
    
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
    
    # 设置时间轴
    max_frame = 13
    if os.path.exists(INPUT_DIR):
        files = [f for f in os.listdir(INPUT_DIR) if f.startswith("frame_") and f.endswith(".ply")]
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
        print("Setup complete. Scrub the timeline to see particles.")
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
                "file_path": f"./solid_ground/{filename}",
                "time": time,
                "c2w": get_c2w(cam),
                "intrinsic": get_intrinsic(cam)
            })
            
    # Save Metadata
    json_path = os.path.join(OUTPUT_DIR, "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Done! Metadata saved to {json_path}")

if __name__ == "__main__":
    is_render = SHOULD_RENDER_DEFAULT or "--render" in sys.argv
    
    def frame_change_handler(scene):
        load_ply_frame(scene.frame_current)
            
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_handler)
    print("Frame change handler registered.")
    
    main(should_render=is_render)
