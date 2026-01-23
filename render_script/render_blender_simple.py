import bpy
import numpy as np
import os
import sys

# --- 配置 ---
# 基础路径：使用当前脚本所在目录
BASE_DIR = "/root/workspace/taichi_dataset"

# 材质名称 (决定加载哪个文件夹的数据) - 此时不再作为路径的一部分
MATERIAL_TYPE = "simulation" 

# 输入路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_sim_multi")

def clean_scene():
    """清理场景中的所有物体，保留基本环境"""
    # 选中所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 清理未使用的 Mesh 数据块
    for block in bpy.data.meshes:
        if not block.users:
            bpy.data.meshes.remove(block)

def load_frame_data(frame_idx):
    """读取指定帧的 .npy 粒子位置数据"""
    if frame_idx < 0: return None
    
    filename = f"frame_{frame_idx:04d}.npy"
    path = os.path.join(INPUT_DIR, filename)
    
    if os.path.exists(path):
        try:
            return np.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    return None

def update_particles_handler(scene):
    """
    Blender 帧变化回调函数
    当用户拖动时间轴时，自动加载对应帧的粒子数据并更新网格
    """
    frame_idx = scene.frame_current
    obj = bpy.data.objects.get("ParticleLoader")
    
    if obj is None:
        return

    data = load_frame_data(frame_idx)
    if data is None:
        return

    # 获取当前网格
    old_mesh = obj.data
    
    # 创建新网格并填入数据
    new_mesh = bpy.data.meshes.new("Particles_Mesh")
    new_mesh.from_pydata(data, [], [])
    
    # 替换对象的网格
    obj.data = new_mesh
    
    # 删除旧网格以防内存泄漏
    if old_mesh:
        bpy.data.meshes.remove(old_mesh)

def setup_handler():
    """注册帧更新回调"""
    # 先移除已存在的同名 handler (为了避免重复注册)
    handlers = bpy.app.handlers.frame_change_post
    handlers[:] = [h for h in handlers if h.__name__ != "update_particles_handler"]
    
    # 添加新的 handle
    handlers.append(update_particles_handler)
    print("Frame change handler registered.")

def main():
    print(f"Started Simple Particle Loader...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # 1. 清理场景
    clean_scene()
    
    # 2. 创建承载粒子的对象
    mesh = bpy.data.meshes.new("Particles_Mesh")
    obj = bpy.data.objects.new("ParticleLoader", mesh)
    bpy.context.collection.objects.link(obj)
    
    # 3. 初始加载第0帧 (如果有)
    data = load_frame_data(0)
    if data is not None:
        print(f"Loaded frame 0 with {len(data)} particles.")
        mesh.from_pydata(data, [], [])
    else:
        print("Warning: Frame 0 data not found.")
        
    # 4. 注册回调函数，实现时间轴联动
    setup_handler()
    
    # 设置时间轴长度 (假设有 14 帧，可根据需要调整)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 13
    bpy.context.scene.frame_current = 0
    
    print("Setup complete. Scrub the timeline to see particles.")

if __name__ == "__main__":
    main()
