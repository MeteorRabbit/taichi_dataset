import bpy
import numpy as np
import os
import sys
import math
import json
from mathutils import Vector
# -- Blender 粒子转网格处理示例脚本
# --- 配置 ---
# 基础路径
BASE_DIR = "D:/Experiments/gic/taichi_dataset"

# 预设5种材质名称 (修改这里选择当前渲染的材质)
MATERIAL_TYPE = "sand"  # 可选: water, sand, elastic, plasticine, non_newtonian

# 根据材质名称自动生成输入输出路径
INPUT_DIR = os.path.abspath(f"{BASE_DIR}/output_npy/output_{MATERIAL_TYPE}")
OUTPUT_DIR = os.path.abspath(f"{BASE_DIR}/render_output/{MATERIAL_TYPE}")

# 粒子设置
PARTICLE_RADIUS = 0.008
SAND_GRAIN_RADIUS = 0.0008  # 沙子颗粒半径 (比流体粒子小)
RESOLUTION_PERCENT = 100
RENDER_ENGINE = 'CYCLES' # 'CYCLES' 或 'BLENDER_EEVEE'
SAMPLES = 128

# --- 材质预设 ---
MATERIALS = {
    "water": {"color": (0.1, 0.3, 0.9, 1), "roughness": 0.0, "transmission": 1.0, "ior": 1.33},
    "sand": {"color": (0.76, 0.6, 0.4, 1), "roughness": 1.0, "transmission": 0.0, "ior": 1.45},
    "elastic": {"color": (0.8, 0.1, 0.1, 1), "roughness": 0.2, "transmission": 0.8, "ior": 1.45},
    "plasticine": {"color": (0.2, 0.8, 0.2, 1), "roughness": 0.6, "transmission": 0.0, "ior": 1.45},
    "toothpaste": {"color": (0.9, 0.95, 0.9, 1), "roughness": 0.3, "transmission": 0.0, "subsurface": 0.5},
    "toothpaste_custom": {"color": (0.2, 0.6, 1.0, 1), "roughness": 0.2, "transmission": 0.0, "subsurface": 0.5}, # Added custom blue toothpaste
    "cream": {"color": (1.0, 0.98, 0.9, 1), "roughness": 0.4, "transmission": 0.0, "subsurface": 0.8},
    "non_newtonian": {"color": (1.0, 0.9, 0.8, 1), "roughness": 0.4, "transmission": 0.0, "subsurface": 0.2},
}

# 判断是否为沙子材质 (需要特殊处理)
def is_sand_material():
    return MATERIAL_TYPE == "sand"

def clean_scene():
    # 选中所有物体
    bpy.ops.object.select_all(action='SELECT')

    # 如果存在 'Floor' 物体，取消选中它以免被删除
    if 'Floor' in bpy.data.objects:
        bpy.data.objects['Floor'].select_set(False)

    # 删除选中的物体（保留 'Floor'）
    bpy.ops.object.delete()

    # 清理无用的数据块 (mesh, material 等)
    for block in bpy.data.meshes:
        if not block.users: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if not block.users: bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if not block.users: bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if not block.users: bpy.data.images.remove(block)

# 创建黄金螺旋分布的相机位置，确保Z坐标不为0
def fibonacci_sphere(samples=10):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # 黄金角度
    
    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # z从1到-1
        radius = math.sqrt(1 - z * z)  # 半径在x-y平面
        
        theta = phi * i  # 黄金角度增量
        
        x = math.cos(theta) * radius
        y = abs(math.sin(theta) * radius)
        
        # 确保Y坐标不为0
        if y < 0.1:
            y = 0.1
        
        points.append((x, y, z))
    
    return points

def setup_cameras():
    # 生成10个均匀分布的点
    sphere_points = fibonacci_sphere(10)
    cameras = []

    # 创建相机
    for i, point in enumerate(sphere_points):
        # 围绕中心 (0, 0.005, 0)
        center_offset = Vector((0, 0.005, 0))
        r = 2.0 # 半径设为 2.0 以确保能看到整个场景
        
        scaled_point = Vector((point[0]*r, point[1]*r, point[2]*r)) + center_offset
        
        # 创建相机
        bpy.ops.object.camera_add(location=scaled_point)
        cam = bpy.context.active_object
        cam.name = f"Camera_{i+1}"
        
        # 使相机朝向中心 (0, 0.005, 0)
        direction = center_offset - scaled_point
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()
        
        cameras.append(cam)

    # 设置相机参数
    focal_length = 50
    sensor_width = 36
    sensor_height = 36
    
    for cam in cameras:
        cam.data.lens = focal_length
        cam.data.sensor_width = sensor_width
        cam.data.sensor_height = sensor_height
        
    return cameras

def get_intrinsic(cam):
    # Use exact intrinsics consistent with existing datasets (800x800, principal point 400, focal_px fixed)
    focal_px = 965.6844046797067
    intrinsic = [
        [focal_px, 0.0, 400.0],
        [0.0, focal_px, 400.0],
        [0.0, 0.0, 1.0]
    ]
    return intrinsic

def get_c2w(cam):
    # 获取 Blender 世界矩阵
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

def setup_lighting():
    # Sun Light
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 10, 5)
    light_data.energy = 10.0 # Increased from 3.0
    
    # Area Light (Fill)
    area_data = bpy.data.lights.new(name="Area", type='AREA')
    area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (-2, 3, 2)
    area_data.energy = 500.0
    area_data.size = 5.0
    
    # Point Light (Back/Rim)
    point_data = bpy.data.lights.new(name="Point", type='POINT')
    point_obj = bpy.data.objects.new(name="Point", object_data=point_data)
    bpy.context.collection.objects.link(point_obj)
    point_obj.location = (0.5, 2.0, -1.0)
    point_data.energy = 200.0

    # World Background (Ambient Light)
    if not bpy.context.scene.world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        
        # Get Background node safely
        bg_node = None
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                bg_node = node
                break
                
        if bg_node is None:
            bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
            output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
            world.node_tree.links.new(bg_node.outputs[0], output_node.inputs[0])

        bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)
        bg_node.inputs['Strength'].default_value = 1.0
    else:
        print("Using existing World background.")

def create_material(mat_type):
    mat = bpy.data.materials.new(name="ParticleMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 清空默认节点
    nodes.clear()
    
    # 创建 Principled BSDF
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.location = (0, 0)
    
    # 创建输出
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    links.new(shader.outputs[0], output.inputs[0])
    
    # 设置属性 (兼容 Blender 4.0+)
    props = MATERIALS.get(mat_type, MATERIALS["non_newtonian"])
    
    shader.inputs['Base Color'].default_value = props.get("color", (1,1,1,1))
    shader.inputs['Roughness'].default_value = props.get("roughness", 0.5)
    
    if 'Transmission Weight' in shader.inputs:
        shader.inputs['Transmission Weight'].default_value = props.get("transmission", 0.0)
    elif 'Transmission' in shader.inputs:
        shader.inputs['Transmission'].default_value = props.get("transmission", 0.0)
        
    shader.inputs['IOR'].default_value = props.get("ior", 1.45)

    return mat

def setup_particles_fluid():
    """流体类材质的粒子系统设置 (Points -> Volume -> Mesh)"""
    # 1. 创建容器网格 (仅顶点)
    mesh = bpy.data.meshes.new("ParticleContainer")
    container = bpy.data.objects.new("ParticleContainer", mesh)
    bpy.context.collection.objects.link(container)
    
    # 2. 添加 Geometry Nodes 修改器用于网格化
    mod = container.modifiers.new(name="Meshing", type='NODES')
    node_group = mod.node_group
    if not node_group:
        node_group = bpy.data.node_groups.new(name="MeshingGroup", type='GeometryNodeTree')
        mod.node_group = node_group
        
    # 清空默认节点
    node_group.nodes.clear()
    
    # 创建节点
    input_node = node_group.nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    
    # 点转体积 Points to Volume
    p2v_node = node_group.nodes.new('GeometryNodePointsToVolume')
    p2v_node.location = (-200, 0)
    p2v_node.inputs['Radius'].default_value = PARTICLE_RADIUS * 2.5 # 增加半径以平滑融合
    p2v_node.inputs['Voxel Amount'].default_value = 256 # 更高分辨率
    p2v_node.inputs['Density'].default_value = 10.0
    
    # 体积转网格 Volume to Mesh
    v2m_node = node_group.nodes.new('GeometryNodeVolumeToMesh')
    v2m_node.location = (0, 0)
    v2m_node.inputs['Threshold'].default_value = 0.3
    v2m_node.inputs['Adaptivity'].default_value = 0.0
    
    # 平滑处理
    # Set Position
    set_pos_node = node_group.nodes.new('GeometryNodeSetPosition')
    set_pos_node.location = (200, 0)
    
    # 位置输入 Position Input
    pos_input_node = node_group.nodes.new('GeometryNodeInputPosition')
    pos_input_node.location = (0, -200)
    
    # 模糊属性 Blur Attribute
    blur_node = node_group.nodes.new('GeometryNodeBlurAttribute')
    blur_node.location = (200, -200)
    blur_node.data_type = 'FLOAT_VECTOR'
    blur_node.inputs['Iterations'].default_value = 6 # 平滑迭代次数
    
    # 设置平滑着色 Set Shade Smooth
    smooth_node = node_group.nodes.new('GeometryNodeSetShadeSmooth')
    smooth_node.location = (400, 0)

    # 细分曲面 Subdivision Surface
    subdiv_node = node_group.nodes.new('GeometryNodeSubdivisionSurface')
    subdiv_node.location = (600, 0)
    subdiv_node.inputs['Level'].default_value = 1
    
    # 设置材质 Set Material
    mat_node = node_group.nodes.new('GeometryNodeSetMaterial')
    mat_node.location = (800, 0)
    mat = create_material(MATERIAL_TYPE)
    mat_node.inputs['Material'].default_value = mat
    
    # 输出 Output
    output_node = node_group.nodes.new('NodeGroupOutput')
    output_node.location = (1000, 0)
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # 连接节点
    links = node_group.links
    links.new(input_node.outputs[0], p2v_node.inputs['Points'])
    links.new(p2v_node.outputs[0], v2m_node.inputs['Volume'])
    links.new(v2m_node.outputs[0], set_pos_node.inputs['Geometry'])
    links.new(pos_input_node.outputs['Position'], blur_node.inputs['Value'])
    links.new(blur_node.outputs['Value'], set_pos_node.inputs['Position'])
    links.new(set_pos_node.outputs[0], smooth_node.inputs['Geometry'])
    links.new(smooth_node.outputs[0], subdiv_node.inputs['Mesh'])
    links.new(subdiv_node.outputs[0], mat_node.inputs['Geometry'])
    links.new(mat_node.outputs[0], output_node.inputs[0])
    
    return container, mesh

def setup_particles_sand():
    """沙子材质的粒子系统设置 (Points -> Instance Spheres) 保持颗粒感"""
    # 1. 创建容器网格 (仅顶点)
    mesh = bpy.data.meshes.new("ParticleContainer")
    container = bpy.data.objects.new("ParticleContainer", mesh)
    bpy.context.collection.objects.link(container)
    
    # 2. 添加 Geometry Nodes 修改器
    mod = container.modifiers.new(name="Meshing", type='NODES')
    node_group = mod.node_group
    if not node_group:
        node_group = bpy.data.node_groups.new(name="SandGroup", type='GeometryNodeTree')
        mod.node_group = node_group
        
    node_group.nodes.clear()
    
    # 输入
    input_node = node_group.nodes.new('NodeGroupInput')
    input_node.location = (-600, 0)
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    
    # Mesh to Points (将顶点转换为点云)
    m2p_node = node_group.nodes.new('GeometryNodeMeshToPoints')
    m2p_node.location = (-400, 0)
    
    # 创建 UV Sphere 作为沙粒实例
    uv_sphere_node = node_group.nodes.new('GeometryNodeMeshUVSphere')
    uv_sphere_node.location = (-400, -200)
    uv_sphere_node.inputs['Segments'].default_value = 8  # 低多边形以提高性能
    uv_sphere_node.inputs['Rings'].default_value = 6
    uv_sphere_node.inputs['Radius'].default_value = SAND_GRAIN_RADIUS
    
    # Instance on Points (在每个点上实例化球体)
    instance_node = node_group.nodes.new('GeometryNodeInstanceOnPoints')
    instance_node.location = (-100, 0)
    
    # 随机缩放 (让沙粒大小有变化)
    random_node = node_group.nodes.new('FunctionNodeRandomValue')
    random_node.location = (-300, -350)
    random_node.data_type = 'FLOAT_VECTOR'
    random_node.inputs[0].default_value = (0.7, 0.7, 0.7)  # 最小缩放
    random_node.inputs[1].default_value = (1.3, 1.3, 1.3)  # 最大缩放
    
    # 随机旋转 (让沙粒朝向有变化)
    random_rot_node = node_group.nodes.new('FunctionNodeRandomValue')
    random_rot_node.location = (-300, -500)
    random_rot_node.data_type = 'FLOAT_VECTOR'
    random_rot_node.inputs[0].default_value = (0, 0, 0)
    random_rot_node.inputs[1].default_value = (6.28, 6.28, 6.28)  # 0-2π
    
    # Realize Instances (将实例转换为真实几何体)
    realize_node = node_group.nodes.new('GeometryNodeRealizeInstances')
    realize_node.location = (150, 0)
    
    # 设置平滑着色
    smooth_node = node_group.nodes.new('GeometryNodeSetShadeSmooth')
    smooth_node.location = (350, 0)
    
    # 设置材质
    mat_node = node_group.nodes.new('GeometryNodeSetMaterial')
    mat_node.location = (550, 0)
    mat = create_material(MATERIAL_TYPE)
    mat_node.inputs['Material'].default_value = mat
    
    # 输出
    output_node = node_group.nodes.new('NodeGroupOutput')
    output_node.location = (750, 0)
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # 连接节点
    links = node_group.links
    links.new(input_node.outputs[0], m2p_node.inputs['Mesh'])
    links.new(m2p_node.outputs['Points'], instance_node.inputs['Points'])
    links.new(uv_sphere_node.outputs['Mesh'], instance_node.inputs['Instance'])
    links.new(random_node.outputs[1], instance_node.inputs['Scale'])  # Vector output
    links.new(random_rot_node.outputs[1], instance_node.inputs['Rotation'])  # Vector as Euler
    links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
    links.new(realize_node.outputs['Geometry'], smooth_node.inputs['Geometry'])
    links.new(smooth_node.outputs['Geometry'], mat_node.inputs['Geometry'])
    links.new(mat_node.outputs['Geometry'], output_node.inputs[0])
    
    return container, mesh

def setup_particles():
    """根据材质类型选择合适的粒子系统设置"""
    if is_sand_material():
        print("使用沙子渲染模式 (颗粒实例化)")
        return setup_particles_sand()
    else:
        print("使用流体渲染模式 (体积网格化)")
        return setup_particles_fluid()

def load_frame_data(frame_idx):
    if frame_idx < 0: return None
    filename = f"frame_{frame_idx:04d}.npy"
    path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    return None

def update_mesh_for_frame(container, frame_idx):
    data = load_frame_data(frame_idx)
    if data is None:
        return
        
    mesh = container.data
    if len(mesh.vertices) != len(data):
        new_mesh = bpy.data.meshes.new("ParticleContainer_Temp")
        new_mesh.from_pydata(data, [], [])
        container.data = new_mesh
        bpy.data.meshes.remove(mesh)
    else:
        mesh.vertices.foreach_set("co", data.flatten())
        mesh.update()

def main(should_render=None):
    # 解析参数
    # 如果未指定 should_render，则检查命令行参数 "--render"
    if should_render is None:
        should_render = "--render" in sys.argv
    print(f"渲染模式: {'启用' if should_render else '禁用'}")

    clean_scene()
    
    # 1. 设置相机 (Fibonacci Sphere)
    cameras = setup_cameras()
    
    # 2. 灯光设置 (已注释，保留手动设置)
    # setup_lighting()
    
    # 3. 设置粒子系统
    container, mesh = setup_particles()
    
    # 4. 渲染设置
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.cycles.samples = SAMPLES
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = 800
    scene.render.resolution_y = 800
    scene.render.image_settings.file_format = 'PNG'
    
    start_frame = 0
    end_frame = 13
    fps = 24
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_data = []
    
    # --- 第一阶段: 背景帧 (Frame -1) ---
    print("正在处理背景帧 (-1)...")
    container.hide_render = True # 隐藏物体
    # 禁用 Geometry Nodes 修改器以防止空网格崩溃
    container.modifiers["Meshing"].show_render = False
    container.modifiers["Meshing"].show_viewport = False
    
    for i, cam in enumerate(cameras):
        scene.camera = cam
        filename = f"r_{i}_-1.png"
        scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
        
        if should_render:
            bpy.ops.render.render(write_still=True)
        
        time = -1.0 / fps
        c2w = get_c2w(cam)
        intrinsic = get_intrinsic(cam)
        
        all_data.append({
            "file_path": f"./{MATERIAL_TYPE}/{filename}",
            "time": time,
            "c2w": c2w,
            "intrinsic": intrinsic
        })
        
    container.hide_render = False # 显示物体
    # 重新启用 Geometry Nodes 修改器
    container.modifiers["Meshing"].show_render = True
    container.modifiers["Meshing"].show_viewport = True
    
    # --- 第二阶段: 模拟帧 (0..13) ---
    print("正在处理模拟帧...")
    for frame in range(start_frame, end_frame + 1):
        print(f"设置帧 {frame}...")
        
        # 1. 先更新网格数据
        update_mesh_for_frame(container, frame) 
        
        # 2. 设置帧号 (触发修改器)
        scene.frame_set(frame) 
        
        time = frame / fps
        
        for i, cam in enumerate(cameras):
            scene.camera = cam
            filename = f"r_{i}_{frame}.png"
            scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            
            if should_render:
                 bpy.ops.render.render(write_still=True)
            
            c2w = get_c2w(cam)
            intrinsic = get_intrinsic(cam)
            
            all_data.append({
                "file_path": f"./{MATERIAL_TYPE}/{filename}",
                "time": time,
                "c2w": c2w,
                "intrinsic": intrinsic
            })

    # 保存元数据 (保存到材质对应的输出目录)
    json_path = os.path.join(OUTPUT_DIR, "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
        
    print(f"设置完成. 元数据已保存至 {json_path}.")
    if not should_render:
        print("注意: 渲染已跳过. 使用 '--render' 参数或在脚本中设置 should_render=True 以启用渲染.")

if __name__ == "__main__":
    # 设置 should_render=True 启用渲染, False 跳过
    main(should_render=True)
