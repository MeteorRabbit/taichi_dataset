import taichi as ti
import numpy as np
import math
import os
import trimesh
import argparse
import json

# Initialize Taichi (still used for vector math acceleration)
ti.init(arch=ti.cpu)

@ti.data_oriented
class RigidBodySimulator:
    def __init__(self, n_balls, radius, mass=0.17):
        self.n_balls = n_balls
        self.radius = radius
        self.mass = mass
        self.dt = 1.0 / 6000.0  # Higher sub-step frequency for collision accuracy and stability
        self.frame_dt = 1.0 / 60.0 # Output FPS
        
        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=n_balls)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=n_balls)
        # Friction coefficients - adjusted for per-frame application
        self.drag_per_frame = 0.995   # Air resistance / Rolling friction approximation (per frame)
        self.restitution = 0.98 # Bounciness (1.0 = perfect elastic, <1.0 = energy loss)
        
    def set_state(self, i: int, pos, vel):
        """Set state using numpy - avoids Taichi kernel overhead for initialization"""
        self.pos[i] = pos
        self.vel[i] = vel

    @ti.kernel
    def integrate(self):
        """Integration step - updates positions based on velocities"""
        for i in range(self.n_balls):
            self.pos[i] += self.vel[i] * self.dt

    def resolve_collisions(self):
        """Collision detection and resolution - run on CPU to avoid race conditions"""
        pos_np = self.pos.to_numpy()
        vel_np = self.vel.to_numpy()
        
        min_dist = 2 * self.radius
        
        for i in range(self.n_balls):
            for j in range(i + 1, self.n_balls):
                diff = pos_np[i] - pos_np[j]
                dist_sq = np.dot(diff, diff)
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    if dist > 1e-6:
                        norm = diff / dist
                    else:
                        norm = np.array([1.0, 0.0, 0.0])  # Fallback for perfect overlap
                    
                    # Relative Velocity
                    rel_vel = vel_np[i] - vel_np[j]
                    vel_along_normal = np.dot(rel_vel, norm)
                    
                    # Only resolve if moving towards each other
                    if vel_along_normal < 0:
                        # Impulse scalar (equal mass simplification)
                        j_impulse = -(1 + self.restitution) * vel_along_normal
                        j_impulse /= (1/self.mass + 1/self.mass)
                        
                        impulse = j_impulse * norm
                        vel_np[i] += impulse / self.mass
                        vel_np[j] -= impulse / self.mass
                    
                    # Positional Correction (prevent sinking)
                    correction = (min_dist - dist) / 2.0
                    pos_np[i] += norm * correction
                    pos_np[j] -= norm * correction
        
        # Write back to Taichi fields
        self.pos.from_numpy(pos_np)
        self.vel.from_numpy(vel_np)

    @ti.kernel
    def apply_drag(self, drag: ti.f32):
        """Apply drag/friction to velocities"""
        for i in range(self.n_balls):
            self.vel[i] *= drag

    def advance(self):
        steps = int(self.frame_dt / self.dt)
        # Calculate per-step drag so total effect matches drag_per_frame
        drag_per_step = self.drag_per_frame ** (1.0 / steps)
        
        for _ in range(steps):
            self.integrate()
            self.resolve_collisions()
            self.apply_drag(drag_per_step)

def load_mesh_geometry(filename, scale=1.0, offset=[0,0,0]):
    print(f"Loading mesh from {filename}...")
    mesh = trimesh.load(filename)
    if isinstance(mesh, trimesh.Scene):
         mesh = list(mesh.geometry.values())[0]
            
    # Rotate to Y-up
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)
    
    if scale != 1.0: mesh.apply_scale(scale)
    
    # Center mesh at (0,0,0)
    centroid = mesh.centroid
    mesh.vertices -= centroid
    
    # Calculate radius (approximate bounding box)
    radius = (mesh.extents.max() / 2.0)
    print(f"Mesh Radius: {radius}")
    
    return mesh, radius

def run_simulation(output_dir="workspace/taichi_dataset/output_billiard"):
    # Config
    ply_path = "/root/workspace/taichi_dataset/meshes/billiard.ply"
    
    # 1. Load Geometry (Only needed for exporting)
    mesh_template, ball_radius = load_mesh_geometry(ply_path, scale=1.0)
    
    # 2. Setup Scene
    balls_pos = []
    
    # Standard billiard ball radius is usually ~2.85cm or 0.0285m, mesh scale might vary.
    # We use the loaded radius.
    spacing = ball_radius * 2.01 # Tiny gap
    rows = 3
    cx, cy, cz = 0.0, ball_radius, 0.0 
    
    # Triangle Setup
    for r in range(rows):
        row_z = cz + r * spacing * math.sqrt(3)/2
        start_x = cx - (r * spacing) / 2.0
        for c in range(r + 1):
            pos_x = start_x + c * spacing
            balls_pos.append([pos_x, cy, row_z])
            
    # Cue Ball
    cue_dist = 0.5
    cue_pos = [cx, cy, cz - cue_dist]
    balls_pos.append(cue_pos)
    
    n_balls = len(balls_pos)
    print(f"Simulating {n_balls} balls (Rigid Body)...")
    
    # 3. Init Simulator
    sim = RigidBodySimulator(n_balls, ball_radius)
    
    # Set Init State
    for i, pos in enumerate(balls_pos):
        vel = [0.0, 0.0, 0.0]
        # Last ball is Cue
        if i == n_balls - 1:
            vel = [0.0, 0.0, 1.5] # Speed along Z
        sim.set_state(i, pos, vel)
        
    # 4. Loop
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {"n_frames": 60, "n_balls": n_balls}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    for frame in range(60):
        # Export
        current_pos = sim.pos.to_numpy()
        
        for i in range(n_balls):
            # Create a copy of mesh and move it
            # Trimesh transformation is slow in loop, better to just edit vertices
            # But deepcopy is safer for small n
            new_verts = mesh_template.vertices + current_pos[i]
            
            # Use lightweight export if possible, but trimesh is easy
            # Explicitly constructing minimal mesh object for speed
            export_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh_template.faces)
            export_mesh.export(os.path.join(output_dir, f"frame_{frame:04d}_ball_{i}.ply"))
            
        print(f"Frame {frame} saved.")
        
        # Step
        sim.advance()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="workspace/taichi_dataset/particles_output/output_billiard_n-mpm", help="Output directory")
    args = parser.parse_args()
    run_simulation(args.output_dir)
