import taichi as ti
import numpy as np
import math
import os
import trimesh
import argparse
import csv

# Initialize Taichi
ti.init(arch=ti.cuda, device_memory_fraction=0.9)

@ti.data_oriented
class MPMSimulator:
    # Material types
    elasticity = 10

    def __init__(self, dtype, dt, frame_dt, particle_layout, dx, inv_dx, n_particles, gravity=[0, -9.8, 0], material=elasticity, cuda_chunk_size=400):
        
        dim = self.dim = 3
        self.dtype = dtype
        self.material = material
        self.particle = particle_layout
        self.n_particles = n_particles
        self.dx = dx
        self.inv_dx = inv_dx
        self.frame_dt = frame_dt
        self.cfl_satisfy = ti.field(ti.i8, shape=())
        self.p_vol = ti.field(self.dtype, shape=())
        self.dt = ti.field(self.dtype, shape=())
        self.n_substeps = ti.field(ti.i32, shape=())
        self.cuda_chunk_size = cuda_chunk_size
        
        # The last one is the first one in the next chunk
        self.step_particle = self.particle.dense(ti.j, cuda_chunk_size+1) 
        
        # Position, velocity, affine velocity field, deformation gradient
        self.x = ti.Vector.field(dim, dtype=self.dtype)
        self.v = ti.Vector.field(dim, dtype=self.dtype)
        self.C = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.F = ti.Matrix.field(dim, dim, dtype=self.dtype)
        
        self.step_particle.place(self.x, self.v, self.C, self.F)
        
        self.damping_coeff = 0 # No damping for free fall

        # Material properties
        self.mu = ti.field(dtype=self.dtype)
        self.lam = ti.field(dtype=self.dtype)
        self.p_mass = ti.field(self.dtype)
        
        # Temporary fields for computation
        self.F_tmp = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.U = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.V = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig_out = ti.Vector.field(dim, dtype=self.dtype)

        self.particle.place(self.mu, self.lam, self.p_mass, self.F_tmp, self.U, self.V, self.sig, self.sig_out)

        # Grid setup
        grid_size = 4096
        offset = self.offset = tuple(-grid_size // 2 for _ in range(3))
        self.offset_vec = ti.Vector(list(offset), ti.i32)
        grid_block_size = 128
        leaf_block_size = self.leaf_block_size = 4
        
        grid = self.grid = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block = self.block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)

        self.grid_m = ti.field(dtype=self.dtype)
        self.grid_v_in = ti.Vector.field(dim, dtype=self.dtype)
        self.grid_v_out = ti.Vector.field(dim, dtype=self.dtype)
       
        def block_component(c):
            block.dense(ti.ijk, leaf_block_size).place(c, offset=offset)

        block_component(self.grid_m)
        block_component(self.grid_v_in)
        block_component(self.grid_v_out)
        
        self.gravity = ti.Vector.field(dim, self.dtype, shape=())
        self.gravity[None] = gravity
        self.dt[None] = dt
        self.n_substeps[None] = round(frame_dt / dt)
    
    @ti.kernel
    def compute_F_tmp(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            self.F_tmp[p] = (ti.Matrix.identity(self.dtype, self.dim) + self.dt[None] * self.C[p, s]) @ self.F[p, s]

    @ti.kernel
    def svd(self):
        for p in range(self.n_particles[None]):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p].cast(ti.f64))

    @ti.kernel
    def project_F(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            self.F[p, s+1] = self.F_tmp[p]

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.kernel
    def p2g(self, s: ti.i32):
        ti.block_local(self.grid_m)
        ti.block_local(self.grid_v_in)
        ti.block_local(self.grid_v_out)
        for p in range(self.n_particles[None]):
            base = ti.floor(self.x[p, s] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.x[p, s] * self.inv_dx[None] - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = self.F[p, s+1]
            stress = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            
            # Simple elasticity (Neo-Hookean / Corotated)
            J = new_F.determinant()
            scale = self.lam[p] * ti.log(J) - self.mu[p]
            grad_v = self.C[p, s]
            epsilon = 0.5 * (grad_v + grad_v.transpose())
            stress = self.damping_coeff * epsilon * J + self.mu[p] * (new_F @ new_F.transpose()) + scale * ti.Matrix.identity(self.dtype, self.dim)

            stress = (-self.dt[None] * self.p_vol[None] * 4 * self.inv_dx[None]**2) * stress
            affine = stress + self.p_mass[p] * self.C[p, s]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), self.dtype) - fx) * self.dx[None]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        self.grid_v_in[base + offset] += \
                            weight * (self.p_mass[p] * self.v[p, s] + affine @ dpos)
                        self.grid_m[base + offset] += weight * self.p_mass[p]
        
    @ti.kernel
    def grid_op(self, s: ti.i32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-8 * self.dx[None] ** 3:
                v_out = self.grid_v_in[I] / self.grid_m[I] + self.dt[None] * self.gravity[None]
                # No collision here for free fall
                self.grid_v_out[I] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        ti.block_local(self.grid_m)
        ti.block_local(self.grid_v_in)
        ti.block_local(self.grid_v_out)
        for p in range(self.n_particles[None]):
            base = ti.floor(self.x[p, f] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.x[p, f] * self.inv_dx[None] - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), self.dtype) - fx
                        g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx[None]

            self.v[p, f + 1] = new_v
            self.x[p, f + 1] = self.x[p, f] + self.dt[None] * self.v[p, f + 1]
            self.C[p, f + 1] = new_C
    
    @ti.kernel
    def check_cfl(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.math.isnan(self.v[p, s]).any():
                self.cfl_satisfy[None] = 0
            if ti.abs(self.v[p, s]).max() * self.dt[None] > self.dx[None]:
                self.cfl_satisfy[None] = 0

    @ti.kernel
    def rotate_buffer(self):
        for p in range(self.n_particles[None]):
            self.x[p, 0] = self.x[p, self.cuda_chunk_size]
            self.v[p, 0] = self.v[p, self.cuda_chunk_size]
            self.C[p, 0] = self.C[p, self.cuda_chunk_size]
            self.F[p, 0] = self.F[p, self.cuda_chunk_size]

    def substep(self, s):
        local_index = s % self.cuda_chunk_size
        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd()
        self.project_F(local_index)

        self.p2g(local_index)
        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)
        
        if (local_index == self.cuda_chunk_size-1):
            self.rotate_buffer()
    
    def advance(self, f):
        for i in range(self.n_substeps[None] * f, self.n_substeps[None] * (f+1)):
            if self.cfl_satisfy[None]:
                self.substep(i)
            else:
                print(f"CFL violated at step {i}")
                break

def load_mesh_and_fill(filename, dx, scale=1.0, jitter_ratio=0.4):
    print(f"Loading mesh from {filename} for particle generation...")
    mesh = trimesh.load(filename)
    if scale != 1.0: mesh.apply_scale(scale)

    # 1. Get Surface Vertices
    surface_verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    n_verts = len(surface_verts)
    
    # 2. Fill Interior
    print("Voxelizing mesh to fill interior...")
    voxel_grid = mesh.voxelized(pitch=dx)
    voxel_grid = voxel_grid.fill()
    interior_points = voxel_grid.points.astype(np.float32)

    if jitter_ratio > 0:
        jitter = (np.random.rand(*interior_points.shape) - 0.5) * dx * jitter_ratio
        interior_points += jitter

    print(f"Mesh info: {n_verts} vertices, {len(interior_points)} interior particles.")
    all_points = np.vstack((surface_verts, interior_points))
    
    return all_points

# --- Simulation Setup ---

def run_simulation(output_dir="workspace/taichi/output_sim"):
    # Parameters
    dtype = ti.f32
    dt = 1e-4
    frame_dt = 1.0/60.0
    dx = 0.01
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 16 
    
    rho = 1000.0
    E = 1e6 # Stiff enough
    nu = 0.2
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Initialization
    mesh_path = "/root/workspace/taichi_dataset/meshes/billiard.ply" 
    temp_mesh = trimesh.load(mesh_path)
    extents = temp_mesh.bounding_box.extents
    max_extent = max(extents)
    desired_radius = 0.08
    scale_factor = (desired_radius * 2) / max_extent
    print(f"Auto-scaling mesh by factor {scale_factor}")

    base_particles = load_mesh_and_fill(mesh_path, dx/2, scale=scale_factor, jitter_ratio=0.4)
    # Center the particles
    center_offset = -np.mean(base_particles, axis=0)
    base_particles += center_offset
    
    # Place ONE ball at 2.0m height
    start_height = 2.0
    init_pos = base_particles + np.array([0, start_height, 0], dtype=np.float32)
    init_vel = np.zeros_like(init_pos)
    
    n_particles_est = len(init_pos)
    print(f"Initializing Benchmark Scenario with {n_particles_est} particles")

    # [NEW] Snapshot
    def save_snapshot(components, filename):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        p = components[::5] # Subsample
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=1, c='b', alpha=0.6) 
        # Mapped Taichi (x, z, y). Plot Z is height.
        
        # [NEW] Draw Projection Lines for CoM
        com = np.mean(components, axis=0)
        cx, cz, cy = com[0], com[2], com[1] # Remapped to Plot coords: x->x, z->y, y->z
        
        # Projection to xy plane (floor) - Drop Z to 0 (or bottom of plot?)
        # Let's project to the "walls" based on limits [-0.5, 0.5]
        
        # Line to Z-axis (Height): (cx, cz, cy) -> (0, 0, cy) ? No, that's to the axis line.
        # User said "corresponding to axes".
        # Let's draw lines to the axes planes or axes themselves.
        # Standard: Line to (cx, cz, 0) then to (cx, 0, 0) and (0, cz, 0)?
        # Let's do simply: 
        # 1. Vertical line down to ground: (cx, cz, cy) -> (cx, cz, 0)
        # 2. Line to X axis: (cx, cz, 0) -> (cx, -0.5, 0) (projection on wall) -> (cx, -0.5, 0) marker?
        # Let's just draw lines from CoM to the 3 separate axes values.
        # X-axis val: (cx, 0.5, 0) -- no, axes are at edges.
        
        # Let's try drawing to the "zero" planes if visible, or just simple drop lines.
        # Line 1: CoM to Vertical Axis (Height): (cx, cz, cy) -> (0, 0, cy) -- dashed
        # Line 2: CoM to X Axis: (cx, cz, cy) -> (cx, 0, 0) (if y/z are 0 at axis)
        
        # Let's stick to standard 3D plot style:
        # Point P(x, y, z)
        # Line P -> (x, y, zmin)
        # Line P -> (xmin, y, z)
        # Line P -> (x, ymin, z)
        
        xlim = [-0.5, 0.5]
        ylim = [-0.5, 0.5] # Plot Y (Taichi Z)
        zlim = [0, 2.5]    # Plot Z (Taichi Y)
        
        # Define Grid Walls (Back panels based on view azim=45)
        # Viewer is at (+, +). Back walls are X_min (-0.5) and Y_min (-0.5). Floor is Z_min (0).
        wall_x = -0.5
        wall_y = -0.5
        wall_z = 0.0
        
        # 1. Drop Line to Floor
        ax.plot([cx, cx], [cz, cz], [cy, wall_z], 'k--', alpha=0.5, linewidth=1)
        # Mark floor point
        ax.scatter([cx], [cz], [wall_z], s=10, c='k', marker='x', alpha=0.3)

        # 2. Line to Back Right Wall (Y=0.5)
        ax.plot([cx, cx], [cz, wall_y], [cy, cy], 'k--', alpha=0.5, linewidth=1)
        
        # 3. Line to Back Left Wall (X=-0.5)
        ax.plot([cx, wall_x], [cz, cz], [cy, cy], 'k--', alpha=0.5, linewidth=1)
        
        # Add coordinate label at CoM
        ax.text(cx, cz, cy + 0.1, f'({cx:.2f}, {cy:.2f}, {cz:.2f})', fontsize=9, color='black')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (Height)')
        ax.set_title('Free Fall Setup (t=0)')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 2.5])
        
        # [NEW] Enforce equal aspect ratio
        ax.set_box_aspect((1, 1, 2.5))
        
        # View angle
        ax.view_init(elev=20, azim=45)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved snapshot to {filename}")

    save_snapshot(init_pos, "benchmark_drop_setup.png")

    # Fields
    num_particles = ti.field(ti.i32, shape=())
    particle = ti.root.dynamic(ti.i, 2**18, particle_chunk_size)
    
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, gravity=[0, -9.8, 0], 
                       material=MPMSimulator.elasticity, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3
    
    @ti.kernel
    def init_particles_with_vel(n: int, pos_field: ti.template(), vel_field: ti.template()):
        num_particles[None] = n
        for i in range(n):
            sim.x[i, 0] = pos_field[i]
            sim.v[i, 0] = vel_field[i]
            sim.F[i, 0] = ti.Matrix.identity(dtype, 3)
            sim.C[i, 0] = ti.Matrix.zero(dtype, 3, 3)
            sim.p_mass[i] = sim.p_vol[None] * rho
            sim.mu[i] = mu
            sim.lam[i] = lam

    pos_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    vel_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    pos_field.from_numpy(init_pos)
    vel_field.from_numpy(init_vel)
    
    init_particles_with_vel(n_particles_est, pos_field, vel_field)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare CSV Logging
    csv_path = os.path.join(output_dir, "benchmark_drop_data.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "com_y", "vel_y", "theory_y", "theory_v", "error_y"])
    
    simulation_frames = 40 # 0.66 seconds, enough to fall 2m (t=sqrt(2*2/9.8)=0.63s)
    gravity = -9.8
    
    print(f"Starting simulation... Outputting to {csv_path}")
    
    for frame in range(simulation_frames):
        sim.advance(frame)
        
        # Calculate CoM and Mean Velocity
        current_pos = sim.x.to_numpy()[:num_particles[None], 0, :]
        current_vel = sim.v.to_numpy()[:num_particles[None], 0, :]
        
        com = np.mean(current_pos, axis=0)
        mean_vel = np.mean(current_vel, axis=0)
        
        # Time
        time = (frame + 1) * frame_dt 
        # Analytical Solution
        theory_y = start_height + 0.5 * gravity * time**2
        theory_v = gravity * time
        
        error_y = abs(com[1] - theory_y)
        
        # Log
        csv_writer.writerow([time, com[1], mean_vel[1], theory_y, theory_v, error_y])
        
        print(f"Frame {frame}: y={com[1]:.4f} (Exact: {theory_y:.4f}), v={mean_vel[1]:.4f}")

    csv_file.close()
    print("Benchmark completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="/root/workspace/taichi_dataset/benchmark_output", help="Output directory")
    args = parser.parse_args()
    run_simulation(args.output_dir)
