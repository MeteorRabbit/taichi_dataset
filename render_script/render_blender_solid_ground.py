import bpy
import numpy as np
import os
import sys

# --- 配置 ---
# 基础路径：使用当前脚本所在目录
BASE_DIR = "D:/Experiments/gic/taichi_dataset"

# 材质名称 (决定加载哪个文件夹的数据) - 此时不再作为路径的一部分
MATERIAL_TYPE = "simulation" 

# 输入路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_solid_ground")

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    # 选中所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 清理未使用的 Mesh 数据块
    for block in bpy.data.meshes:
        if not block.users:
            bpy.data.meshes.remove(block)

def update_mesh_handler(scene):
    """
    Blender 帧变化回调函数
    当用户拖动时间轴时，自动加载对应帧的 PLY 网格
    """
    frame_idx = scene.frame_current
    filename = f"frame_{frame_idx:04d}.ply"
    path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(path):
        print(f"Frame {frame_idx} not found: {path}")
        return

    obj_name = "SimulationMesh"
    
    # 1. 删除旧对象
    if obj_name in bpy.data.objects:
        old_obj = bpy.data.objects[obj_name]
        old_mesh = old_obj.data
        bpy.data.objects.remove(old_obj, do_unlink=True)
        if old_mesh:
            bpy.data.meshes.remove(old_mesh)
            
    # 2. 导入新 PLY
    # 注意: import_mesh.ply 可能会根据 blender 版本略有不同，这是通用写法
    # 它会将导入的对象设为选中状态
    try:
        bpy.ops.import_mesh.ply(filepath=path)
    except AttributeError:
         # Blender 4.0+ uses bpy.ops.wm.ply_import
         bpy.ops.wm.ply_import(filepath=path)
         
    # 3. 重命名导入的对象
    if bpy.context.selected_objects:
        # 获取最近导入的对象 (假设是选中的那一个)
        imported_obj = bpy.context.selected_objects[0]
        imported_obj.name = obj_name
        
        # 可选: 设置平滑
        # bpy.context.view_layer.objects.active = imported_obj
        # bpy.ops.object.shade_smooth()

def setup_handler():
    """注册帧更新回调"""
    # 先移除已存在的同名 handler (为了避免重复注册)
    handlers = bpy.app.handlers.frame_change_post
    handlers[:] = [h for h in handlers if h.__name__ != "update_mesh_handler"]
    
    # 添加新的 handle
    handlers.append(update_mesh_handler)
    print("Frame change handler registered.")

def main():
    print(f"Started Mesh PLY Loader...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # 1. 清理场景
    clean_scene()
    
    # 2. 初始加载第0帧
    # 设置当前帧为 0，然后手动调用一次 handler 来加载
    bpy.context.scene.frame_set(0)
    update_mesh_handler(bpy.context.scene)
    
    # 3. 注册回调
    setup_handler()
        
    # 4. 注册回调函数，实现时间轴联动
    setup_handler()
    
    # 设置时间轴长度 (假设有 14 帧，可根据需要调整)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 13
    bpy.context.scene.frame_current = 0
    
    print("Setup complete. Scrub the timeline to see particles.")

if __name__ == "__main__":
    main()
