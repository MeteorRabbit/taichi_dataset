import bpy
import numpy as np
import os
import sys
import math

# --- Configuration ---
# You can change these or pass them via command line if you extend the script
INPUT_DIR = os.path.abspath("D:/Experiments/gic/taichi/output_sim")
OUTPUT_DIR = os.path.abspath("D:/Experiments/gic/taichi/render_output")
MATERIAL_TYPE = "non_newtonian" # Options: water, sand, elastic, plasticine, toothpaste, cream

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
    "cream": {"color": (1.0, 0.98, 0.9, 1), "roughness": 0.4, "transmission": 0.0, "subsurface": 0.8},
    "non_newtonian": {"color": (1.0, 0.9, 0.8, 1), "roughness": 0.4, "transmission": 0.0, "subsurface": 0.2},
}

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.textures: bpy.data.textures.remove(block)
    for block in bpy.data.images: bpy.data.images.remove(block)

def setup_camera():
    # Create Camera
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new(name='Camera', object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Position Camera (Looking at 0.5, 0.5, 0.5)
    # Adjust these coordinates based on your scene scale
    cam_obj.location = (1.5, 1.2, 2.0) 
    
    # Add constraint to track the center
    track_to = cam_obj.constraints.new(type='TRACK_TO')
    
    # Create an empty at center to track
    center_empty = bpy.data.objects.new("Center", None)
    bpy.context.collection.objects.link(center_empty)
    center_empty.location = (0.5, 0.5, 0.5)
    
    track_to.target = center_empty
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'

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
    
    # 2. Create the Instance Object (Sphere)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=PARTICLE_RADIUS)
    instance_sphere = bpy.context.active_object
    instance_sphere.name = "ParticleInstance"
    
    # Assign Material
    instance_sphere.data.materials.clear() # Clear existing materials
    mat = create_material(MATERIAL_TYPE)
    instance_sphere.data.materials.append(mat)
    
    # Shade Smooth
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = instance_sphere
    instance_sphere.select_set(True)
    bpy.ops.object.shade_smooth()
    
    # 3. Parent Sphere to Container
    instance_sphere.parent = container
    
    # 4. Enable Instancing on Vertices
    container.instance_type = 'VERTS'
    
    # IMPORTANT FIX: 
    # Do NOT hide the instance object itself, otherwise instances are hidden too.
    # Do NOT unlink it.
    # Do NOT move it far away (it might mess up bounding box).
    # Instead, we rely on the fact that the instance object is parented to the container.
    # But we don't want to see the original one at the center.
    # The standard way is to put the instance object in a separate collection and exclude it, 
    # OR just hide it. Wait, hiding parent usually hides children, but hiding the instanced object...
    
    # Let's try the most robust way: Geometry Nodes (if available) or Particle System.
    # But sticking to simple instancing:
    # If we hide_render=True on the instance_sphere, instances ALSO disappear in Cycles.
    
    # Solution: Scale the original one to 0? No, instances will scale to 0.
    # Solution: Move it out of camera view.
    instance_sphere.location = (0, -0, 0) 
    
    # Make sure the container itself is visible
    container.show_instancer_for_viewport = False # Don't show container vertices
    container.show_instancer_for_render = False
    
    return container, mesh

def load_frame_data(frame_idx):
    filename = f"frame_{frame_idx:04d}.npy"
    path = os.path.join(INPUT_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    return None

def update_handler(scene):
    frame = scene.frame_current
    # Assuming simulation frames match blender frames 1-to-1
    # Or you can map them: sim_frame = frame - start_frame
    
    data = load_frame_data(frame)
    if data is None:
        return
    
    container = bpy.data.objects.get("ParticleContainer")
    if container:
        mesh = container.data
        
        # Update mesh vertices
        # If vertex count changes, we need to create a new mesh or resize
        # For MPM, particle count is usually constant, but let's be safe
        
        if len(mesh.vertices) != len(data):
            # Re-create mesh data if count differs (slow but robust)
            new_mesh = bpy.data.meshes.new("ParticleContainer_Temp")
            new_mesh.from_pydata(data, [], [])
            container.data = new_mesh
            # Clean up old mesh
            bpy.data.meshes.remove(mesh)
        else:
            # Just update coordinates (faster)
            # Flatten data for foreach_set
            mesh.vertices.foreach_set("co", data.flatten())
            mesh.update()

def main():
    clean_scene()
    setup_camera()
    setup_lighting()
    
    container, mesh = setup_particles()
    
    # Load first frame to initialize
    data = load_frame_data(0)
    if data is not None:
        mesh.from_pydata(data, [], [])
    
    # Setup Render Settings
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.cycles.samples = SAMPLES
    scene.render.resolution_percentage = RESOLUTION_PERCENT
    scene.frame_start = 0
    scene.frame_end = 120 # Adjust based on your sim
    
    # Output settings
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    scene.render.filepath = os.path.join(OUTPUT_DIR, "render_")
    
    # Register Handler
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(update_handler)
    
    print("Scene setup complete. Ready to render.")

if __name__ == "__main__":
    # If running from command line with arguments, you might need to parse sys.argv
    # But for now, we use the constants at the top.
    main()
