import bpy
import numpy as np
import os

# ======================
# 配置
# ======================
NPY_DIR = r"D:\Experiments\gic\taichi\output_sim"  # 改成你的路径
FRAME_STEP = 5                                # 和你导出一致
POINT_RADIUS = 0.01

# ======================
# 清空场景
# ======================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ======================
# 创建点云对象
# ======================
mesh = bpy.data.meshes.new("PointCloud")
obj = bpy.data.objects.new("PointCloud", mesh)
bpy.context.collection.objects.link(obj)

# ======================
# 创建材质
# ======================
mat = bpy.data.materials.new(name="PointMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial")
emission = nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 5.0

# Add Attribute Node for Vertex Colors
attr_node = nodes.new("ShaderNodeAttribute")
attr_node.attribute_name = "Color"
links.new(attr_node.outputs["Color"], emission.inputs["Color"])

links.new(emission.outputs[0], output.inputs[0])

mesh.materials.append(mat)

# ======================
# 加载颜色
# ======================
COLORS_PATH = os.path.join(NPY_DIR, "colors.npy")
global_colors = None
if os.path.exists(COLORS_PATH):
    print(f"Loading colors from {COLORS_PATH}")
    c = np.load(COLORS_PATH)
    # Ensure RGBA
    if c.shape[1] == 3:
        ones = np.ones((c.shape[0], 1), dtype=c.dtype)
        c = np.hstack([c, ones])
    global_colors = c.flatten()
else:
    print("No colors.npy found, using default color.")

# ======================
# 帧更新函数
# ======================
def update_pointcloud(scene):
    frame = scene.frame_current
    file_id = frame * FRAME_STEP
    path = os.path.join(NPY_DIR, f"frame_{file_id:04d}.npy")

    if not os.path.exists(path):
        return

    points = np.load(path)

    # ---- 安全检查 ----
    if points.ndim != 2 or points.shape[1] < 3:
        print(f"[WARN] Invalid shape {points.shape} in {path}")
        return

    # 只取前 3 维 (x, y, z)
    points = points[:, :3]

    mesh.clear_geometry()
    verts = [tuple(p) for p in points]
    mesh.from_pydata(verts, [], [])
    
    # Apply Colors
    if global_colors is not None:
        n_points = len(verts)
        n_colors = len(global_colors) // 4
        
        if n_points == n_colors:
            if "Color" not in mesh.attributes:
                mesh.attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
            mesh.attributes["Color"].data.foreach_set("color", global_colors)
        else:
            # Only print warning once per run to avoid spam, or just print
            print(f"[WARN] Particle count {n_points} != Color count {n_colors}")

    mesh.update()

# ======================
# 注册 handler
# ======================
bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(update_pointcloud)

# ======================
# 时间轴
# ======================
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 14

print("Point cloud animation (NPY) ready.")
