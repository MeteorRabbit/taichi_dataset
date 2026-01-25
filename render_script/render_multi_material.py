import bpy
import numpy as np
import os
import sys
import math
import json
from mathutils import Vector

# --- 配置 ---
BASE_DIR = "D:/Experiments/gic/taichi_dataset"
INPUT_DIR = os.path.join(BASE_DIR, "particles_output/multi_material")
OUTPUT_DIR = os.path.join(BASE_DIR, "render_output/multi_material")

# 粒子大小设置
PARTICLE_RADIUS = 0.008
SAND_GRAIN_RADIUS = 0.0008  
RESOLUTION_PERCENT = 100
RENDER_ENGINE = 'CYCLES' 
SAMPLES = 128

# 材质定义
MATERIALS_CONFIG = {
    "sand": {"color": (0.76, 0.6, 0.4, 1), "roughness": 1.0, "transmission": 0.0, "ior": 1.45},
    "elastic": {"color": (0.8, 0.1, 0.1, 1), "roughness": 0.2, "transmission": 0.8, "ior": 1.45},
}

def clean_scene():
    """清理场景，只保留必要的环境节点"""
    bpy.ops.object.select_all(action='SELECT')
    if 'Floor' in bpy.data.objects:
        bpy.data.objects['Floor'].select_set(False)
    bpy.ops.object.delete()

    for block in bpy.data.meshes:
        if not block.users: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if not block.users: bpy.data.materials.remove(block)

def create_material(name, props):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.location = (0, 0)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(shader.outputs[0], output.inputs[0])
    
    shader.inputs['Base Color'].default_value = props.get("color", (1,1,1,1))
    shader.inputs['Roughness'].default_value = props.get("roughness", 0.5)
    
    if 'Transmission Weight' in shader.inputs:
        shader.inputs['Transmission Weight'].default_value = props.get("transmission", 0.0)
    elif 'Transmission' in shader.inputs:
        shader.inputs['Transmission'].default_value = props.get("transmission", 0.0)
        
    shader.inputs['IOR'].default_value = props.get("ior", 1.45)
    return mat

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

def setup_cameras():
    sphere_points = fibonacci_sphere(10)
    cameras = []
    for i, point in enumerate(sphere_points):
        center_offset = Vector((0, 0.005, 0))
        r = 2.0 
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
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 10, 5)
    light_data.energy = 10.0
    
    area_data = bpy.data.lights.new(name="Area", type='AREA')
    area_obj = bpy.data.objects.new(name="Area", object_data=area_data)
    bpy.context.collection.objects.link(area_obj)
    area_obj.location = (-2, 3, 2)
    area_data.energy = 500.0
    area_data.size = 5.0
    
    if not bpy.context.scene.world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes['Background']
        bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)

def setup_particles_ground_sand():
    """Setup Ground as Sand Particle System (Instancing)"""
    # 1. 创建容器网格 (仅顶点)
    mesh = bpy.data.meshes.new("GroundMesh")
    container = bpy.data.objects.new("GroundContainer", mesh)
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
    uv_sphere_node.inputs['Segments'].default_value = 8
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
    mat = create_material("SandMaterial", MATERIALS_CONFIG["sand"])
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
    links.new(random_node.outputs[1], instance_node.inputs['Scale'])
    links.new(random_rot_node.outputs[1], instance_node.inputs['Rotation'])
    links.new(instance_node.outputs['Instances'], realize_node.inputs['Geometry'])
    links.new(realize_node.outputs['Geometry'], smooth_node.inputs['Geometry'])
    links.new(smooth_node.outputs['Geometry'], mat_node.inputs['Geometry'])
    links.new(mat_node.outputs['Geometry'], output_node.inputs[0])
    
    return container

def update_ground(container, frame):
    npy_path = os.path.join(INPUT_DIR, f"frame_{frame:04d}_ground.npy")
    if os.path.exists(npy_path):
        data = np.load(npy_path)
        mesh = container.data
        
        n_verts_new = len(data)
        n_verts_old = len(mesh.vertices)
        
        if n_verts_new != n_verts_old:
            # Resize mesh geometry in-place
            if n_verts_new > n_verts_old:
                mesh.vertices.add(n_verts_new - n_verts_old)
            else:
                # Blender doesn't have a direct 'remove' for vertices batch easily reachable
                # So we create a new mesh and copy data if size shrinks or radically changes
                # But to preserve modifier links, we should try to keep the object wrapper
                
                # Re-creating mesh data block is safer for potentially shrinking arrays
                # and ensuring clean state. 
                # CRITICAL: We must re-link the modifier target if needed, but usually 
                # modifiers are on the Object, not the Mesh.
                # However, changing container.data might be the issue.
                
                # Let's try fully clearing and rebuilding in the SAME mesh block if possible?
                # No, standard API replaces list.
                
                # Fallback to the replace method but ensure we touch the object to update dependency graph
                new_mesh = bpy.data.meshes.new("GroundMesh_Temp")
                new_mesh.from_pydata(data, [], [])
                
                old_mesh = container.data
                container.data = new_mesh
                
                # Check consistency
                if not container.modifiers:
                     print("Warning: Modifiers lost on ground container!")
                
                bpy.data.meshes.remove(old_mesh)
                return 

        # If counts match or we just added vertices (which are at 0,0,0), update positions
        # 注意: 如果刚刚 add 了顶点，它们的坐标默认是 0,0,0
        mesh.vertices.foreach_set("co", data.flatten())
        mesh.update()
    else:
        print(f"Warning: Ground file not found at {npy_path}")

def update_object(frame, mat_obj):
    ply_path = os.path.join(INPUT_DIR, f"frame_{frame:04d}_obj.ply")
    # print(f"Trying to load object from: {ply_path}")
    obj_name = "ObjectContainer"
    
    # Check if object exists
    if obj_name in bpy.data.objects:
        old_obj = bpy.data.objects[obj_name]
        bpy.data.objects.remove(old_obj, do_unlink=True)
    
    if os.path.exists(ply_path):
        # Import PLY
        bpy.ops.wm.ply_import(filepath=ply_path)
        # It's selected after import
        if bpy.context.selected_objects:
            imported = bpy.context.selected_objects[0]
            imported.name = obj_name
            # Assign material
            if len(imported.data.materials) == 0:
                imported.data.materials.append(mat_obj)
            else:
                imported.data.materials[0] = mat_obj
                
            # Shade Auto Smooth
            bpy.context.view_layer.objects.active = imported
            # 'use_auto_smooth' 已经被 Blender 4.1+ 移除，对于较新版本直接使用 shade_smooth()
            # 如果需要锐边效果，应当添加 'Smooth by Angle' 修改器，这里对于弹性体直接平滑即可
            bpy.ops.object.shade_smooth()
    else:
        print(f"Warning: Object file not found at {ply_path}")

def get_intrinsic(cam):
    focal_px = 965.6844046797067
    return [[focal_px, 0.0, 400.0], [0.0, focal_px, 400.0], [0.0, 0.0, 1.0]]

def get_c2w(cam):
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

def main(should_render=False):
    print("Multi-Material Rendering Started.")
    clean_scene()
    
    cameras = setup_cameras()
    setup_lighting()
    
    # 1. Setup Ground Container (Simulated Sand)
    ground_container = setup_particles_ground_sand()
    
    # 2. Setup Object Material (Elastic)
    mat_obj = create_material("ElasticMaterial", MATERIALS_CONFIG["elastic"])
    
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.cycles.samples = SAMPLES
    scene.cycles.device = 'GPU' # Use GPU if possible
    scene.render.image_settings.file_format = 'PNG'
    
    start_frame = 0
    end_frame = 29
    fps = 30
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_data = []

    print("Rendering Frames...")
    for frame in range(start_frame, end_frame + 1):
        print(f"Processing Frame {frame}...")
        
        # Update Scene
        update_ground(ground_container, frame)
        update_object(frame, mat_obj)
        
        scene.frame_set(frame)
        time = frame / fps
        
        # Render for each camera
        for i, cam in enumerate(cameras):
            scene.camera = cam
            filename = f"r_{i}_{frame}.png"
            scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Render if enabled
            if should_render:
                bpy.ops.render.render(write_still=True)
            
            # Metadata
            all_data.append({
                "file_path": f"./multi_material/{filename}",
                "time": time,
                "c2w": get_c2w(cam),
                "intrinsic": get_intrinsic(cam)
            })
            
    # Save Metadata
    json_path = os.path.join(OUTPUT_DIR, "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Done! Metadata saved to {json_path}")
    if not should_render:
        print("Rendering was skipped (preview mode).")

if __name__ == "__main__":
    # Check for --render argument
    is_render_mode = "--render" in sys.argv
    
    # 注册 Frame Change Handler
    def frame_change_handler(scene):
        frame = scene.frame_current
        # 只在非渲染模式或手动预览时触发
        # 这里为了演示，我们假设如果 container 存在就更新
        if "GroundContainer" in bpy.data.objects:
            ground_container = bpy.data.objects["GroundContainer"]
            update_ground(ground_container, frame)
        
        # 为了避免材质重复创建，可以只获取
        mat_obj = bpy.data.materials.get("ElasticMaterial")
        if mat_obj:
            update_object(frame, mat_obj)
            
    # 清除旧的 handler 以免重复
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.frame_change_post.append(frame_change_handler)
    print("Frame change handler registered.")
    
    # 执行主流程
    main(is_render_mode)
