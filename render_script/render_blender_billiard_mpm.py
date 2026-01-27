import bpy
import numpy as np
import os
import sys

# --- 配置 ---
# 基础路径
BASE_DIR = "D:/Experiments/gic/taichi_dataset"

# 输入路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_billiard_mpm")

# 目标对象名称
# 材质配置
OBJECTS_CONFIG = {
    # 0 号球 (母球)
    "Billiard_0": {
        "prefix": "billiard_0",
        "material_name": "whiteBall", 
        "color": (0.9, 0.9, 0.9, 1) # White
    },
    # 1-6 号球 (子球)
    "Billiard_1": {
        "prefix": "billiard_1",
        "material_name": "N01", 
        "color": (0.9, 0.8, 0.0, 1) # Yellow
    },
    "Billiard_2": {
        "prefix": "billiard_2",
        "material_name": "N09", 
        "color": (0.1, 0.1, 0.8, 1) # Blue
    },
    "Billiard_3": {
        "prefix": "billiard_3",
        "material_name": "N06", 
        "color": (0.8, 0.1, 0.1, 1) # Red
    },
    "Billiard_4": {
        "prefix": "billiard_4",
        "material_name": "N07", 
        "color": (0.5, 0.0, 0.5, 1) # Purple
    },
    "Billiard_5": {
        "prefix": "billiard_5",
        "material_name": "N08", 
        "color": (1.0, 0.5, 0.0, 1) # Orange
    },
    "Billiard_6": {
        "prefix": "billiard_6",
        "material_name": "N04", 
        "color": (0.0, 0.5, 0.0, 1) # Green
    },
}

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    # 选中所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 清理未使用的 Mesh 数据块
    for block in bpy.data.meshes:
        if not block.users:
            bpy.data.meshes.remove(block)

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
            # 直接删除
            old_mesh = old_obj.data
            bpy.data.objects.remove(old_obj, do_unlink=True)
            if old_mesh:
                bpy.data.meshes.remove(old_mesh)
        
        # 2. 检查新文件是否存在
        if not os.path.exists(path):
            # print(f"Warning: File not found {path}")
            continue

        # 3. 导入新的 PLY
        # 确保不选其他物体
        bpy.ops.object.select_all(action='DESELECT')
        
        try:
            # 尝试使用标准导入算子
            bpy.ops.import_mesh.ply(filepath=path)
        except AttributeError:
            try:
                 # 兼容 Blender 4.0+
                 bpy.ops.wm.ply_import(filepath=path)
            except:
                 print("Error: Could not find PLY import operator.")
                 return
        except Exception as e:
            print(f"Error importing {path}: {e}")
            continue

        # 4. 获取导入的物体并重命名
        if bpy.context.selected_objects:
            imported_obj = bpy.context.selected_objects[0]
            imported_obj.name = obj_name
            
            # 设置平滑
            try:
                bpy.ops.object.shade_smooth()
            except:
                pass
            
            # 分配材质
            mat = ensure_material(config["material_name"], config["color"])
            if imported_obj.data.materials:
                imported_obj.data.materials[0] = mat
            else:
                imported_obj.data.materials.append(mat)

def update_particles_handler(scene):
    """
    Blender 帧变化回调函数
    当用户拖动时间轴时，自动加载对应帧的 PLY 数据
    """
    # 避免在渲染时频繁触发（视情况而定），但在 Viewport 播放时需要
    load_ply_frame(scene.frame_current)

def setup_handler():
    """注册帧更新回调"""
    # 先移除已存在的同名 handler (为了避免重复注册)
    handlers = bpy.app.handlers.frame_change_post
    handlers[:] = [h for h in handlers if h.__name__ != "update_particles_handler"]
    
    # 添加新的 handle
    handlers.append(update_particles_handler)
    print("Frame change handler registered.")

def main():
    print(f"Started PLY Sequence Loader...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # 1. 清理场景
    clean_scene()
    
    # 2. 初始加载第0帧
    load_ply_frame(0)
        
    # 3. 注册回调函数，实现时间轴联动
    setup_handler()
    
    # 设置时间轴长度 (扫描文件夹确定)
    try:
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".ply") and f.startswith("frame_")]
        if files:
            max_frame = max([int(f.split('_')[1].split('.')[0]) for f in files])
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = max_frame
            print(f"Detected {max_frame+1} frames.")
    except Exception as e:
        print(f"Could not scan directory for frame count: {e}")
        bpy.context.scene.frame_end = 250
    
    bpy.context.scene.frame_current = 0
    
    print("Setup complete. Scrub the timeline to see the mesh sequence.")

if __name__ == "__main__":
    main()
