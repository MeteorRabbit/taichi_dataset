import bpy
import os

# --- 配置 ---
# 基础路径
BASE_DIR = "D:/Experiments/gic/taichi_dataset"

# 输入路径
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/output_soft_hard")

# 定义对象映射：Key 是 Blender 场景中的对象名，Value 是 PLY 文件的前缀
# 这样我们可以支持多个独立的物体序列
OBJECTS_CONFIG = {
    "Curling_Thrower": {
        "prefix": "curling_0",
        "material_name": "RedHandleMaterial",
        "color": (0.8, 0.1, 0.1, 1) 
    },
    "Curling_Target": {
        "prefix": "curling_1",
        "material_name": "BlueStoneMaterial",
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
        if not block.users:
            bpy.data.meshes.remove(block)

def ensure_material(mat_name, color):
    """确保材质存在，如果不存在则创建"""
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Roughness'].default_value = 0.4
            bsdf.inputs['Metallic'].default_value = 0.1
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
            # 记录下它之前的变换（如果需要保持位置，但这里 ply 直接包含位置，所以不需要）
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
    """
    load_ply_frame(scene.frame_current)

def setup_handler():
    """注册帧更新回调"""
    handlers = bpy.app.handlers.frame_change_post
    handlers[:] = [h for h in handlers if h.__name__ != "update_particles_handler"]
    handlers.append(update_particles_handler)
    print("Frame change handler registered.")

def main():
    print(f"Started Multi-PLY Sequence Loader...")
    print(f"Input Directory: {INPUT_DIR}")
    
    # 1. 清理
    clean_scene()
    
    # 2. 扫描文件以确定帧数范围 (只扫描第一个序列)
    try:
        if os.path.exists(INPUT_DIR):
             first_prefix = list(OBJECTS_CONFIG.values())[0]["prefix"]
             files = [f for f in os.listdir(INPUT_DIR) if f.startswith(first_prefix) and f.endswith(".ply")]
             if files:
                 max_frame = max([int(f.split('_')[-1].split('.')[0]) for f in files])
                 bpy.context.scene.frame_start = 0
                 bpy.context.scene.frame_end = max_frame
                 print(f"Detected {max_frame+1} frames.")
             else:
                  bpy.context.scene.frame_end = 30
    except Exception as e:
        print(f"Could not scan directory, defaulting frames: {e}")
        bpy.context.scene.frame_end = 30
    
    # 3. 加载第0帧
    load_ply_frame(0)
    bpy.context.scene.frame_current = 0
        
    # 4. 注册回调
    setup_handler()
    
    print("Setup complete.")

if __name__ == "__main__":
    main()
