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
OBJ_NAME = "BilliardSimulation"

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    # 选中所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 清理未使用的 Mesh 数据块
    for block in bpy.data.meshes:
        if not block.users:
            bpy.data.meshes.remove(block)

def load_ply_frame(frame_idx):
    """
    加载指定帧的 PLY 文件并更新场景对象
    """
    if frame_idx < 0: return
    
    filename = f"frame_{frame_idx:04d}.ply"
    path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(path):
        # 尝试查找是否存在，如果找不到可能还没生成或结束
        return

    # 1. 查找并删除旧对象 (及其 Mesh 数据)
    if OBJ_NAME in bpy.data.objects:
        old_obj = bpy.data.objects[OBJ_NAME]
        old_mesh = old_obj.data
        bpy.data.objects.remove(old_obj, do_unlink=True)
        if old_mesh:
            bpy.data.meshes.remove(old_mesh)

    # 2. 导入新的 PLY
    # 注意：bpy.ops.import_mesh.ply 会将导入的对象设为选中状态
    try:
        # 检查 Blender 版本，2.8+ 使用 import_mesh.ply
        bpy.ops.import_mesh.ply(filepath=path)
    except AttributeError:
        # 有些新版本或特定配置可能使用 wm.ply_import (4.0+)
        try:
             bpy.ops.wm.ply_import(filepath=path)
        except:
             print("Error: Could not find PLY import operator.")
             return

    # 3. 重命名并设置新对象
    if bpy.context.selected_objects:
        # 通常导入后它是选中的
        imported_obj = bpy.context.selected_objects[0]
        imported_obj.name = OBJ_NAME
        
        # 设置平滑着色
        bpy.ops.object.shade_smooth()
        
        # (可选) 分配材质
        mat_name = "BilliardMaterial"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            # 创建一个简单的默认材质
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = (0.8, 0.1, 0.1, 1) # Reddish
                bsdf.inputs['Roughness'].default_value = 0.2
                bsdf.inputs['Metallic'].default_value = 0.0
        
        # 赋予材质
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
