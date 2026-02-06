import taichi as ti
import numpy as np
import math
import os
import argparse
import csv

# Initialize Taichi
ti.init(arch=ti.cuda, device_memory_fraction=0.9)

@ti.data_oriented
class MPMSimulator:
    # Material types
    elasticity = 10
    
    # Surface types
    surface_sticky = 0
    surface_slip = 1
    surface_separate = 2
    surface_friction = 3 # [NEW]

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
        
        self.step_particle = self.particle.dense(ti.j, cuda_chunk_size+1) 
        
        self.x = ti.Vector.field(dim, dtype=self.dtype)
        self.v = ti.Vector.field(dim, dtype=self.dtype)
        self.C = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.F = ti.Matrix.field(dim, dim, dtype=self.dtype)
        
        self.step_particle.place(self.x, self.v, self.C, self.F)
        
        self.damping_coeff = 0 

        self.mu = ti.field(dtype=self.dtype)
        self.lam = ti.field(dtype=self.dtype)
        self.p_mass = ti.field(self.dtype)
        
        self.F_tmp = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.U = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.V = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig_out = ti.Vector.field(dim, dtype=self.dtype)

        self.particle.place(self.mu, self.lam, self.p_mass, self.F_tmp, self.U, self.V, self.sig, self.sig_out)

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
        
        # [NEW] List of colliders
        self.colliders = []

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
            if self.grid_m[I] > 1e-10 * self.dx[None] ** 3:
                v_out = self.grid_v_in[I] / self.grid_m[I] + self.dt[None] * self.gravity[None]
                
                # Apply Colliders
                for i in ti.static(range(len(self.colliders))):
                     v_out = self.colliders[i](I, v_out)
                
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

    # [NEW] Enhanced Collider Logic
    def add_surface_collider(self, point, normal, surface=surface_slip, friction=0.3):
        point = list(point)
        # Normalize normal vector
        length = math.sqrt(sum(x**2 for x in normal))
        normal = [x/length for x in normal]

        @ti.func
        def get_velocity(I, v):
            grid_pos = I.cast(self.dtype) * self.dx[None]
            offset = grid_pos - ti.Vector(point)
            n = ti.Vector(normal)
            
            # Signed distance
            dist = offset.dot(n)
            
            # Collision detection (if penetrating or very close)
            if dist <= 0: # 1e-4 tolerance
                # Decompose velocity
                v_n_val = n.dot(v)
                v_n = v_n_val * n
                v_t = v - v_n
                
                # Check if moving INTO the wall
                if v_n_val < 0:
                    v_n_new = ti.Vector.zero(self.dtype, self.dim)
                    v_t_new = v_t
                    
                    if ti.static(surface == self.surface_sticky):
                        v_t_new = ti.Vector.zero(self.dtype, self.dim)
                        
                    elif ti.static(surface == self.surface_slip):
                        # Frictionless: keep tangent velocity as is
                        pass
                        
                    elif ti.static(surface == self.surface_friction):
                        # Coulomb Friction
                        # Fn (Normal Force Proxy) ~ |v_n| / dt * mass (simplified)
                        # Actually we can just scale velocity directly at boundary for simple impulse friction
                        # Target velocity is 0, max impulse is mu * |vn_impulse|
                        
                        v_t_norm = v_t.norm()
                        if v_t_norm > 1e-6:
                            # The normal impulse required to stop penetration is -v_n_val
                            # The max tangent impulse allowed is friction * |normal_impulse|
                            # v_t_new = v_t - min(|v_t|, friction * |v_n|) * (v_t / |v_t|)
                            
                            max_tangent_impulse = friction * ti.abs(v_n_val)
                            
                            # Simple Coulomb Clamp
                            if v_t_norm <= max_tangent_impulse:
                                # Sticky (Static Friction)
                                v_t_new = ti.Vector.zero(self.dtype, self.dim)
                            else:
                                # Dynamic Friction
                                v_t_new = v_t * (1 - max_tangent_impulse / v_t_norm)
                    
                    # Combine
                    v = v_n_new + v_t_new
                    
            return v

        self.colliders.append(get_velocity)

# --- Simulation Setup ---

def run_simulation(output_dir="/root/workspace/taichi_dataset/benchmark_output", use_friction=False, friction_val=0.0):
    # Parameters
    dtype = ti.f32
    dt = 5e-5 # Smaller DT for contact stability
    frame_dt = 1.0/60.0 # 60 FPS
    dx = 0.015 # Slightly coarser for speed/stability
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 16 
    
    rho = 1000.0
    E = 2e6 # Stiff
    nu = 0.2
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # 0. Cube Particles Generation
    cube_size = 0.2
    half_size = cube_size / 2
    
    # Generate grid of particles
    samples = int(cube_size / (dx/2)) 
    x = np.linspace(-half_size, half_size, samples)
    y = np.linspace(-half_size, half_size, samples)
    z = np.linspace(-half_size, half_size, samples)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    cube_particles = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T.astype(np.float32)
    
    # Rotation for initializing ALIGNED with slope? 
    # Or just drop it on slope? Dropping is better to prove stability.
    # Let's rotate it 30 degrees to match slope initially so it slides smoothly
    theta = math.radians(30)
    c, s = math.cos(theta), math.sin(theta)
    
    # Rotation Matrix (around Z axis, assuming slope drops along X)
    # Slope is defined by normal [-sin, cos, 0]
    # So slope vector is [cos, sin, 0]
    # Let's align slope with X-Y plane
    
    # Rotate cube to match slope
    rot_mat = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # cube_particles = cube_particles @ rot_mat.T
    
    # Initial Position (Up the slope)
    # Slope plane passes through origin
    # Normal = [-sin(30), cos(30), 0] = [-0.5, 0.866, 0]
    
    # Let's place it high up on the slope
    # Slope surface height at x=-1 is y = x * tan(30)
    # Let's simplify: 
    # Plane normal n = [-sin, cos, 0]
    # Plane equation: -x*sin + y*cos = 0 => y = x * tan(theta)
    
    start_x = -1.0
    start_y = start_x * math.tan(theta) + 0.005 # Almost touching surface
    
    cube_particles += np.array([start_x, start_y, 0], dtype=np.float32)
    
    # Rotate particles to align with slope (visualniceness)
    # Rotate around CoM
    com = np.mean(cube_particles, axis=0)
    cube_particles -= com
    cube_particles = cube_particles @ rot_mat.T
    cube_particles += com
    
    init_pos = cube_particles
    init_vel = np.zeros_like(init_pos)
    
    n_particles_est = len(init_pos)
    print(f"Initializing Slope Benchmark with {n_particles_est} particles. Friction: {use_friction}")

    # [NEW] Snapshot
    def save_snapshot(particles, filename):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Particles
        p = particles[::5]
        # Map: X->X, Z->Y_plot, Y->Z_plot (Height)
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=1, c='r' if use_friction else 'b', alpha=0.6, label='Object')
        
        # Plot Slope Plane
        # y = x * tan(theta)
        # Create a grid for the plane (X, Z plane in Taichi coords, but Y varies with X)
        xs = np.linspace(-2, 1, 10)
        zs = np.linspace(-0.5, 0.5, 10)
        X, Z = np.meshgrid(xs, zs)
        Y = X * math.tan(theta)
        
        ax.plot_surface(X, Z, Y, alpha=0.3, color='gray')
        
        # [NEW] Draw Projection Lines for CoM
        com = np.mean(particles, axis=0)
        cx, cz, cy = com[0], com[2], com[1] # Map Taichi(x, z, y) -> Plot(x, y, z)
        
        # Limits
        xlim = [-2, 1]
        ylim = [-1, 1] # Plot Y (Taichi Z)
        zlim = [-1, 1] # Plot Z (Taichi Y)
        
        # Project to Floor (Z=zmin=-1)
        # ax.plot([cx, cx], [cz, cz], [zlim[0], cy], 'k--', alpha=0.4, linewidth=1)
        # Actually for slope visual, maybe just project to the slope surface? 
        # Or standard wall projections.
        
        # Define Grid Walls (Back panels based on view azim=-120 -> Quad 3 -> Back is Max/Max)
        wall_x = 1.0
        wall_y = 1.0 # Plot Y (Taichi Z)
        wall_z = -1.0
        
        # 1. To Floor
        ax.plot([cx, cx], [cz, cz], [cy, wall_z], 'k--', alpha=0.5, linewidth=1)
        ax.scatter([cx], [cz], [wall_z], s=10, c='k', marker='x', alpha=0.3)
        
        # 2. To Back Right Wall (Y=1.0)
        ax.plot([cx, cx], [cz, wall_y], [cy, cy], 'k--', alpha=0.5, linewidth=1)
        
        # 3. To Back Left Wall (X=1.0)
        ax.plot([cx, wall_x], [cz, cz], [cy, cy], 'k--', alpha=0.5, linewidth=1)
        
        # Mark the CoM
        ax.scatter([cx], [cz], [cy], s=20, c='k', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (Height)')
        ax.set_title(f'Inclined Plane Setup (Angle={math.degrees(theta):.0f}$^\circ$)')
        
        # Set realistic aspect
        ax.set_xlim([-2, 1])
        ax.set_ylim([-1, 1]) 
        ax.set_zlim([-1, 1])
        
        # [NEW] Enforce equal aspect ratio
        ax.set_box_aspect((3, 2, 2))
        
        # View angle
        ax.view_init(elev=20, azim=-120)
        
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved snapshot to {filename}")

    if friction_val == 0.0 or friction_val == 0.3: # Only save once or for key cases
        save_snapshot(init_pos, f"benchmark_slope_setup_mu_{friction_val:.1f}.png")

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
    
    # Define Slope Collider
    normal_vec = [-math.sin(theta), math.cos(theta), 0]
    
    surface_type = MPMSimulator.surface_friction if use_friction else MPMSimulator.surface_slip
    
    sim.add_surface_collider(point=[0, 0, 0], normal=normal_vec, surface=surface_type, friction=friction_val)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare CSV
    filename = f"benchmark_slope_mu_{friction_val:.1f}.csv"
    csv_path = os.path.join(output_dir, filename)
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["time", "vel_mag", "theory_v", "error_v"])
    
    simulation_frames = 60 # 1 second
    gravity_g = 9.8
    
    # Theoretical Acceleration
    # a = g * (sin - mu * cos)
    acc_theory = gravity_g * math.sin(theta)
    if use_friction:
        acc_theory -= gravity_g * friction_val * math.cos(theta)
        acc_theory = max(0, acc_theory)
        
    print(f"Starting simulation... Outputting to {csv_path}")
    print(f"Theoretical Acceleration: {acc_theory:.4f} m/s^2")
    
    for frame in range(simulation_frames):
        sim.advance(frame)
        
        current_vel = sim.v.to_numpy()[:num_particles[None], 0, :]
        mean_vel = np.mean(current_vel, axis=0)
        
        # Calculate velocity magnitude (down slope)
        vel_mag = np.linalg.norm(mean_vel)
        
        # Time
        time = (frame + 1) * frame_dt 
        theory_v = acc_theory * time
        
        error_v = abs(vel_mag - theory_v)
        
        csv_writer.writerow([time, vel_mag, theory_v, error_v])
        
        print(f"Frame {frame}: v={vel_mag:.4f} (Theory: {theory_v:.4f})")

    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="/root/workspace/taichi_dataset/benchmark_output", help="Output directory")
    parser.add_argument('--mu', type=float, default=0.0, help="Friction coefficient")
    args = parser.parse_args()
    
    run_simulation(args.output_dir, args.mu > 0, args.mu)
