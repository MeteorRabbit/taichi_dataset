import bpy
import numpy as np
import os
import sys
import math
import json
import collections
from mathutils import Vector

# --- Configuration ---
if sys.platform == "win32":
    BASE_DIR = "D:/Experiments/gic/taichi_dataset"
else:
    BASE_DIR = "/root/workspace/taichi_dataset"

SHOULD_RENDER_DEFAULT = False

INPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "particles_output/output_billiard_n-mpm")
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "render_output/billiard")

RENDER_ENGINE = 'CYCLES'  # or 'BLENDER_EEVEE'
SAMPLES = 128
RESOLUTION_X = 800
RESOLUTION_Y = 800

# Billiard Ball Materials
# Standard Colors
BILLIARD_COLORS = [
    (1.0, 1.0, 1.0, 1.0), # Cue Ball (White)
    (1.0, 0.84, 0.0, 1.0), # 1 - Yellow
    (0.0, 0.0, 1.0, 1.0), # 2 - Blue
    (1.0, 0.0, 0.0, 1.0), # 3 - Red
    (0.5, 0.0, 0.5, 1.0), # 4 - Purple
    (1.0, 0.5, 0.0, 1.0), # 5 - Orange
    (0.0, 1.0, 0.0, 1.0), # 6 - Green
    (0.5, 0.0, 0.0, 1.0), # 7 - Maroon
    (0.0, 0.0, 0.0, 1.0), # 8 - Black
]

def clean_scene():
    """Clear the scene except for essential world nodes if any."""
    if bpy.context.screen:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clean meshes, materials, collections
    for block in bpy.data.meshes:
        if not block.users: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if not block.users: bpy.data.materials.remove(block)
    for block in bpy.data.collections:
        if not block.users: bpy.data.collections.remove(block)

def create_material(name, color, roughness=0.1, metallic=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.location = (0, 0)
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(shader.outputs[0], output.inputs[0])
    
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    shader.inputs['Specular IOR Level'].default_value = 0.5
    
    return mat

def create_floor():
    """Create a green felt table-like floor."""
    # bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    # floor = bpy.context.active_object
    # floor.name = "Floor"
    
    # mat = create_material("FeltMaterial", (0.05, 0.3, 0.05, 1.0), roughness=0.8)
    # if len(floor.data.materials) == 0:
    #     floor.data.materials.append(mat)
    # else:
    #     floor.data.materials[0] = mat
    pass

def setup_cameras():
    # 使用 Fibonacci Sphere 生成均匀视点
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
    center_offset = Vector((0, 0.5, 0)) # 这里的中心根据台球场景调整
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
            bg_node.inputs['Color'].default_value = (0.3, 0.3, 0.3, 1)

def get_intrinsic(cam):
    focal_px = 965.6844046797067 
    return [[focal_px, 0.0, 400.0], [0.0, focal_px, 400.0], [0.0, 0.0, 1.0]]

def get_c2w(cam):
    matrix = np.array(cam.matrix_world)
    return matrix[:3, :].tolist()

def get_mesh_center(filepath):
    """Read PLY file and compute mesh center without importing to Blender."""
    import struct
    
    with open(filepath, 'rb') as f:
        # Read header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Not a PLY file")
        
        vertex_count = 0
        is_binary = False
        is_little_endian = True
        header_end = False
        properties = []
        
        while not header_end:
            line = f.readline().decode('ascii').strip()
            if line.startswith('format'):
                if 'binary_little_endian' in line:
                    is_binary = True
                    is_little_endian = True
                elif 'binary_big_endian' in line:
                    is_binary = True
                    is_little_endian = False
                else:
                    is_binary = False
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == 'end_header':
                header_end = True
        
        # Read vertices
        vertices = []
        if is_binary:
            endian = '<' if is_little_endian else '>'
            # Assume x, y, z are first 3 floats
            vertex_size = 0
            for dtype, name in properties:
                if dtype == 'float':
                    vertex_size += 4
                elif dtype == 'double':
                    vertex_size += 8
                elif dtype in ('uchar', 'char'):
                    vertex_size += 1
                elif dtype in ('int', 'uint'):
                    vertex_size += 4
            
            for _ in range(vertex_count):
                data = f.read(vertex_size)
                x, y, z = struct.unpack(endian + 'fff', data[:12])
                vertices.append((x, y, z))
        else:
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append((x, y, z))
        
        if not vertices:
            return (0, 0, 0)
        
        # Compute center
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        center = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
        return center


def import_frame_balls(input_dir, frame_idx):
    """Imports all balls for a given frame."""
    
    # Find all ball files for this frame
    # Pattern: frame_{frame:04d}_ball_{b_idx}.ply
    files = [f for f in os.listdir(input_dir) if f.startswith(f"frame_{frame_idx:04d}_ball_") and f.endswith(".ply")]
    
    # Sort by ball index to maintain consistent coloring
    files.sort(key=lambda x: int(x.split('_ball_')[1].split('.')[0]))
    
    loaded_objects = []
    
    for i, f in enumerate(files):
        path = os.path.join(input_dir, f)
        
        # Import PLY
        bpy.ops.wm.ply_import(filepath=path)
        
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            obj.name = f"Ball_{i}"
            
            # Smooth shading
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.shade_smooth()
            
            # Subsurf for better look (optional)
            mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            mod.levels = 1
            mod.render_levels = 2
            
            # Assign Material
            # Cue ball is usually the last one in our simulation setup (ball loaded last), 
            # Or first depending on setup.
            # In simulate_billiard.py: balls_pos.append(cue_pos) which was LAST.
            # So the last file index should be White.
            # The others can be colored.
            
            # Let's map materials cyclically or specifically.
            # Simulation has 7 balls (6 triangle + 1 cue).
            # Last one (Index 6) is Cue.
            
            color = BILLIARD_COLORS[i % len(BILLIARD_COLORS)]
            
            # Override for Cue Ball logic if we know count
            # But here dynamic is safer.
            if i == len(files) - 1:
                color = BILLIARD_COLORS[0] # White
            else:
                color = BILLIARD_COLORS[i % (len(BILLIARD_COLORS)-1) + 1] # Skip white, pick others
                
            mat_name = f"BallMat_{i}"
            if mat_name in bpy.data.materials:
                mat = bpy.data.materials[mat_name]
            else:
                mat = create_material(mat_name, color)
                
            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat
            
            loaded_objects.append(obj)
            
    return loaded_objects

def clear_balls():
    """Deletes all objects starting with Ball_"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith("Ball_"):
            obj.select_set(True)
    bpy.ops.object.delete()


def setup_animated_scene(input_dir):
    """
    Create an animated scene where ball positions are keyframed.
    This allows scrubbing the timeline to see the animation.
    """
    print("Setting up animated scene with keyframes...")
    
    # Find all PLY files and organize by ball index
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    if not all_files:
        print(f"No PLY files found in {input_dir}")
        return
    
    # Extract max frame and ball count
    max_frame = 0
    ball_indices = set()
    for f in all_files:
        try:
            parts = f.replace('.ply', '').split('_')
            frame_num = int(parts[1])
            ball_idx = int(parts[3])
            max_frame = max(max_frame, frame_num)
            ball_indices.add(ball_idx)
        except:
            pass
    
    n_balls = len(ball_indices)
    print(f"Found {n_balls} balls, {max_frame + 1} frames")
    
    # Set timeline range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max_frame
    
    # Read all ball positions for all frames
    # Structure: ball_positions[ball_idx][frame] = (x, y, z)
    ball_positions = {b: {} for b in ball_indices}
    
    for frame in range(max_frame + 1):
        for b_idx in ball_indices:
            filepath = os.path.join(input_dir, f"frame_{frame:04d}_ball_{b_idx}.ply")
            if os.path.exists(filepath):
                center = get_mesh_center(filepath)
                ball_positions[b_idx][frame] = center
    
    # Import the first frame mesh for each ball and create keyframed animation
    ball_objects = {}
    
    for b_idx in sorted(ball_indices):
        # Import first frame mesh
        first_frame_file = os.path.join(input_dir, f"frame_0000_ball_{b_idx}.ply")
        if not os.path.exists(first_frame_file):
            continue
            
        bpy.ops.wm.ply_import(filepath=first_frame_file)
        
        if not bpy.context.selected_objects:
            continue
            
        obj = bpy.context.selected_objects[0]
        obj.name = f"Ball_{b_idx}"
        
        # Compute initial mesh center and move origin to center
        initial_center = ball_positions[b_idx].get(0, (0, 0, 0))
        
        # Move mesh vertices so that center is at origin (local coords)
        for v in obj.data.vertices:
            v.co.x -= initial_center[0]
            v.co.y -= initial_center[1]
            v.co.z -= initial_center[2]
        
        # Set object location to initial center
        obj.location = initial_center
        
        # Smooth shading
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        
        # Subsurf
        mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        mod.levels = 1
        mod.render_levels = 2
        
        # Material
        n_total_balls = len(ball_indices)
        if b_idx == n_total_balls - 1:
            color = BILLIARD_COLORS[0]  # Cue ball (White)
        else:
            color = BILLIARD_COLORS[b_idx % (len(BILLIARD_COLORS) - 1) + 1]
        
        mat_name = f"BallMat_{b_idx}"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = create_material(mat_name, color)
        
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        
        ball_objects[b_idx] = obj
    
    # Now add keyframes for each ball
    print("Adding keyframes...")
    for b_idx, obj in ball_objects.items():
        for frame in range(max_frame + 1):
            if frame in ball_positions[b_idx]:
                pos = ball_positions[b_idx][frame]
                obj.location = pos
                obj.keyframe_insert(data_path="location", frame=frame)
        
        # Set interpolation to linear for more accurate physics
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kf in fcurve.keyframe_points:
                    kf.interpolation = 'LINEAR'
    
    print(f"Animation setup complete! Timeline: 0 to {max_frame}")


def render_scene(should_render=True, input_dir=INPUT_DIR_DEFAULT, output_dir=OUTPUT_DIR_DEFAULT, use_animation=True):
    print(f"Billiard Rendering Started. Mode: {'RENDER' if should_render else 'PREVIEW'}")
    
    clean_scene()
    cameras = setup_cameras()
    setup_lighting()
    create_floor()
    
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.cycles.samples = SAMPLES
    scene.cycles.device = 'GPU'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check max frames logic
    frame_files = [f for f in os.listdir(input_dir) if f.endswith(".ply")]
    max_frame = 30
    if frame_files:
        try:
            # frame_XXXX_...
            nums = []
            for f in frame_files:
                parts = f.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                     nums.append(int(parts[1]))
            if nums:
                max_frame = max(nums)
        except: pass
    
    print(f"Found max frame: {max_frame}")
    scene.frame_start = 0
    scene.frame_end = max_frame

    # --- Setup Animation ---
    # Always set up animation in preview mode for scrubbing
    if use_animation:
        setup_animated_scene(input_dir)
    
    # --- Preview Mode ---
    if not should_render:
        print("Preview Mode: Scene setup complete.")
        scene.frame_set(0)
        return

    # --- Render Mode ---
    all_data = []
    fps = 24
    
    for frame in range(max_frame + 1):
        print(f"Rendering Frame {frame}/{max_frame}...")
        scene.frame_set(frame)
        time = frame / fps
        
        if not use_animation:
            # If not using animation, manually import per frame
            clear_balls()
            import_frame_balls(input_dir, frame)
        
        # Render Multi-View
        for i, cam in enumerate(cameras):
            scene.camera = cam
            filename = f"r_{i}_{frame}.png"
            scene.render.filepath = os.path.join(output_dir, filename)
            
            bpy.ops.render.render(write_still=True)
            
            all_data.append({
                "file_path": f"./billiard/{filename}",
                "time": time,
                "c2w": get_c2w(cam),
                "intrinsic": get_intrinsic(cam)
            })

    # Save Metadata
    json_path = os.path.join(output_dir, "all_data.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=4)
    print(f"Done! Metadata saved to {json_path}")


if __name__ == "__main__":
    # Argument Parser override for Blender
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
        
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT_DIR_DEFAULT, help='Input directory with particles')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR_DEFAULT, help='Output directory for images')
    parser.add_argument('--render', action='store_true', help='Force render mode')
    parser.add_argument('--no-animation', action='store_true', help='Use per-frame import instead of keyframes')
    
    args = parser.parse_args(argv)
    
    # Combine Default toggle with Arg
    is_render = SHOULD_RENDER_DEFAULT or args.render
    
    render_scene(should_render=is_render, input_dir=args.input, output_dir=args.output, 
                 use_animation=not args.no_animation)
