import bpy
import numpy as np
import os
import sys
import math
import json
from mathutils import Vector
# --blender做粒子转网格处理demo
# --- Configuration ---
# You can change these or pass them via command line if you extend the script
INPUT_DIR = os.path.abspath("D:/Experiments/gic/taichi_dataset/output_npy/output_newtonian")
OUTPUT_DIR = os.path.abspath("D:/Experiments/gic/taichi_dataset/render_output")
MATERIAL_TYPE = "toothpaste_custom" # Options: water, sand, elastic, plasticine, toothpaste, cream

# Particle settings
PARTICLE_RADIUS = 0.008
RESOLUTION_PERCENT = 100
RENDER_ENGINE = 'CYCLES' # 'CYCLES' or 'BLENDER_EEVEE'
SAMPLES = 128

# --- Material Presets ---
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

def clean_scene():
    # Select all objects
    bpy.ops.object.select_all(action='SELECT')

    # If a 'floor' object exists, deselect it so it is not deleted
    if 'Floor' in bpy.data.objects:
        bpy.data.objects['Floor'].select_set(False)

    # Delete all selected objects (everything except 'floor')
    bpy.ops.object.delete()

    # Remove orphaned data-blocks (those with no users).
    # This is a safer way to clean the scene and will preserve the data
    # (like mesh and material) used by the 'floor' object.
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
        # 缩放点到半径为0.8的球面 (Logic from duck script)
        # Note: Original script had 0.8 radius. 
        # In this scene, object might be small? 
        # Previous camera was at (1.5, 1.2, 2.0) ~ dist 2.7. 
        # 0.8 might be too close for this scene if it's the same scale.
        # But user said "learn the pattern". 
        # I'll stick to 0.8 or maybe slightly adjust if I see the scene scale is different?
        # The duck script assumes a specific object size. 
        # The taichi simulation likely fits in [0,1]^3 box. Center is roughly 0.5.
        # Radius 0.8 around (0,0,0) might be looking at the corner box effectively?
        # Wait, the previous setup_camera tracked (0.5, 0.5, 0.5).
        # Duck script tracks (0,0,0) (implied by just creating at location and rotating to look at origin mostly? or just placement).
        # Duck script: direction = Vector((0, 0, 0.1)) - scaled_point. Looks at (0,0,0.1).
        # THIS scene (MPM) is usually in 0..1 coordinates.
        # So I should probably center the cameras around (0.5, 0.5, 0.5) 
        # OR move the object to (0,0,0). 
        # But `render_blender` keeps object in place.
        # Let's check `setup_camera` again. 
        # "Center_empty.location = (0.5, 0.5, 0.5)"
        # So I should offset my sphere center to (0.5, 0.5, 0.5).
        
        center_offset = Vector((0, 0.005, 0))
        r = 2.0 # Increased from 0.8 because 0.5 is center, need to see the whole 0..1 box.
        # Previous manual camera was at (1.5, 1.2, 2.0) -> dist ~1.8 from center.
        
        scaled_point = Vector((point[0]*r, point[1]*r, point[2]*r)) + center_offset
        
        # 创建相机
        bpy.ops.object.camera_add(location=scaled_point)
        cam = bpy.context.active_object
        cam.name = f"Camera_{i+1}"
        
        # 使相机朝向中心 (0, 0.005, 0)
        # Duck script looked at (0, 0, 0.1). 
        # We will look at (0, 0.005, 0) (center of sim box).
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
    
    # Clear default nodes
    nodes.clear()
    
    # Create Principled BSDF
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.location = (0, 0)
    
    # Create Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    links.new(shader.outputs[0], output.inputs[0])
    
    # Set properties for Blender 4.0+
    props = MATERIALS.get(mat_type, MATERIALS["non_newtonian"])
    
    shader.inputs['Base Color'].default_value = props.get("color", (1,1,1,1))
    shader.inputs['Roughness'].default_value = props.get("roughness", 0.5)
    
    # Blender 4.0+ changes
    if 'Transmission Weight' in shader.inputs:
        shader.inputs['Transmission Weight'].default_value = props.get("transmission", 0.0)
    elif 'Transmission' in shader.inputs:
        shader.inputs['Transmission'].default_value = props.get("transmission", 0.0)
        
    shader.inputs['IOR'].default_value = props.get("ior", 1.45)
    
    if "subsurface" in props:
        # Temporarily disable SSS to debug "black object" issue
        # SSS on very small particles often causes black artifacts in Cycles if scale is wrong
        pass 
        # if 'Subsurface Weight' in shader.inputs:
        #      shader.inputs['Subsurface Weight'].default_value = props["subsurface"]
        # elif 'Subsurface' in shader.inputs:
        #      shader.inputs['Subsurface'].default_value = props["subsurface"]
        
        # # Fix for black particles: Set Subsurface Scale to match particle size
        # # Default is 1.0 (meters), which is too big for 0.008 particles
        # if 'Subsurface Scale' in shader.inputs:
        #     shader.inputs['Subsurface Scale'].default_value = PARTICLE_RADIUS * 2.0
             
        # # Subsurface Color is removed in 4.0, it uses Base Color and Radius
        # if 'Subsurface Color' in shader.inputs:
        #     shader.inputs['Subsurface Color'].default_value = props.get("color", (1,1,1,1))

    return mat

def setup_particles():
    # 1. Create the Container Mesh (Vertices only)
    mesh = bpy.data.meshes.new("ParticleContainer")
    container = bpy.data.objects.new("ParticleContainer", mesh)
    bpy.context.collection.objects.link(container)
    
    # 2. Add Geometry Nodes Modifier for Meshing
    mod = container.modifiers.new(name="Meshing", type='NODES')
    node_group = mod.node_group
    if not node_group:
        node_group = bpy.data.node_groups.new(name="MeshingGroup", type='GeometryNodeTree')
        mod.node_group = node_group
        
    # Clear default nodes
    node_group.nodes.clear()
    
    # Create Nodes
    # Input
    input_node = node_group.nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)
    node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    
    # Points to Volume
    p2v_node = node_group.nodes.new('GeometryNodePointsToVolume')
    p2v_node.location = (-200, 0)
    p2v_node.inputs['Radius'].default_value = PARTICLE_RADIUS * 2.5 # Increased radius for smoother blending
    p2v_node.inputs['Voxel Amount'].default_value = 256 # Higher resolution
    p2v_node.inputs['Density'].default_value = 10.0
    
    # Volume to Mesh
    v2m_node = node_group.nodes.new('GeometryNodeVolumeToMesh')
    v2m_node.location = (0, 0)
    v2m_node.inputs['Threshold'].default_value = 0.3
    v2m_node.inputs['Adaptivity'].default_value = 0.0
    
    # --- Smoothing (Laplacian Smooth via Blur Attribute) ---
    # Set Position
    set_pos_node = node_group.nodes.new('GeometryNodeSetPosition')
    set_pos_node.location = (200, 0)
    
    # Position Input
    pos_input_node = node_group.nodes.new('GeometryNodeInputPosition')
    pos_input_node.location = (0, -200)
    
    # Blur Attribute
    blur_node = node_group.nodes.new('GeometryNodeBlurAttribute')
    blur_node.location = (200, -200)
    blur_node.data_type = 'FLOAT_VECTOR'
    blur_node.inputs['Iterations'].default_value = 6 # Smooth iterations
    
    # Set Shade Smooth
    smooth_node = node_group.nodes.new('GeometryNodeSetShadeSmooth')
    smooth_node.location = (400, 0)

    # Subdivision Surface
    subdiv_node = node_group.nodes.new('GeometryNodeSubdivisionSurface')
    subdiv_node.location = (600, 0)
    subdiv_node.inputs['Level'].default_value = 1
    
    # Set Material
    mat_node = node_group.nodes.new('GeometryNodeSetMaterial')
    mat_node.location = (800, 0)
    mat = create_material(MATERIAL_TYPE)
    mat_node.inputs['Material'].default_value = mat
    
    # Output
    output_node = node_group.nodes.new('NodeGroupOutput')
    output_node.location = (1000, 0)
    node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    # Links
    links = node_group.links
    links.new(input_node.outputs[0], p2v_node.inputs['Points'])
    links.new(p2v_node.outputs[0], v2m_node.inputs['Volume'])
    
    # Smoothing links
    links.new(v2m_node.outputs[0], set_pos_node.inputs['Geometry'])
    links.new(pos_input_node.outputs['Position'], blur_node.inputs['Value'])
    links.new(blur_node.outputs['Value'], set_pos_node.inputs['Position'])
    
    links.new(set_pos_node.outputs[0], smooth_node.inputs['Geometry'])
    links.new(smooth_node.outputs[0], subdiv_node.inputs['Mesh'])
    links.new(subdiv_node.outputs[0], mat_node.inputs['Geometry'])
    links.new(mat_node.outputs[0], output_node.inputs[0])
    
    return container, mesh

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
    # Parse arguments
    # Look for "--render" in command line args
    if should_render is None:
        should_render = "--render" in sys.argv
    print(f"Render mode: {'ENABLED' if should_render else 'DISABLED'}")

    clean_scene()
    
    # 1. Setup Cameras (Fibonacci Sphere)
    cameras = setup_cameras()
    
    # 2. Lighting (Commented out as requested)
    # setup_lighting()
    
    # 3. Setup Particles
    container, mesh = setup_particles()
    
    # 4. Render Settings
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
    
    # --- Phase 1: Background Frames (Frame -1) ---
    print("Processing Background Frames (-1)...")
    container.hide_render = True # Hide object
    # Disable Geometry Nodes modifier to prevent crash on empty mesh
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
            "file_path": f"./render_output/{filename}",
            "time": time,
            "c2w": c2w,
            "intrinsic": intrinsic
        })
        
    container.hide_render = False # Show object
    # Re-enable Geometry Nodes modifier
    container.modifiers["Meshing"].show_render = True
    container.modifiers["Meshing"].show_viewport = True
    
    # --- Phase 2: Simulation Frames (0..13) ---
    print("Processing Simulation Frames...")
    for frame in range(start_frame, end_frame + 1):
        print(f"Frame {frame} setup...")
        
        # 1. Update mesh data first
        update_mesh_for_frame(container, frame) 
        
        # 2. Set frame (triggers modifiers)
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
                "file_path": f"./render_output/{filename}",
                "time": time,
                "c2w": c2w,
                "intrinsic": intrinsic
            })

    # Save Metadata
    json_path = os.path.join(os.path.dirname(OUTPUT_DIR), "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
        
    print(f"Setup complete. Metadata saved to {json_path}.")
    if not should_render:
        print("Note: Rendering was skipped. Use '--render' argument to enable rendering.")

if __name__ == "__main__":
    # Set should_render=True to enable rendering, False to skip
    main(should_render=True)
