
import taichi as ti
import numpy as np
import math
import os
import trimesh
import argparse
import json

# Initialize Taichi
ti.init(arch=ti.cuda, device_memory_fraction=0.9)

@ti.data_oriented
class MPMSimulator:
    # Material types
    surface_sticky = 0
    surface_slip = 1
    surface_separate = 2

    elasticity = 10
    viscous_fluid = 11
    von_mises = 12
    drucker_prager = 13

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
        
        # Increased damping to reduce jelly-like oscillation for rigid implementation
        self.damping_coeff = 15.0 

        # Material properties
        self.friction_alpha = ti.field(dtype=self.dtype, shape=())
        self.cohesion = ti.field(dtype=self.dtype, shape=())
        self.yield_stress = ti.field(dtype=self.dtype, shape=())
        self.plastic_viscosity = ti.field(dtype=self.dtype, shape=())
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

        self.analytic_collision = []
    
    @ti.func
    def smu(self, x1, x2, mu=1e-4):
        return 0.5 * ((x1 + x2) + ti.sqrt((x1 - x2) ** 2 + mu))
    
    @ti.kernel
    def compute_F_tmp(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.static(self.material==self.viscous_fluid):
                self.F_tmp[p][0,0] = (1.0 + self.dt[None] * self.C[p, s].trace()) * self.F[p, s][0,0]
            else:
                self.F_tmp[p] = (ti.Matrix.identity(self.dtype, self.dim) + self.dt[None] * self.C[p, s]) @ self.F[p, s]

    @ti.kernel
    def svd(self):
        for p in range(self.n_particles[None]):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p].cast(ti.f64))

    @ti.kernel
    def project_F(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            sig = ti.Vector([ti.max(self.sig[p][0,0], 0.05), ti.max(self.sig[p][1,1], 0.05), ti.max(self.sig[p][2,2], 0.05)])
            if ti.static(self.material == self.viscous_fluid):
                self.F[p, s+1][0,0] = ti.max(self.F_tmp[p][0,0], 0.05)
            elif ti.static(self.material == self.drucker_prager):
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                shifted_trace = trace_epsilon - self.cohesion[None] * self.dim
                if shifted_trace >= 0:
                    epsilon = ti.Vector.one(self.dtype, self.dim) * self.cohesion[None]
                else:
                    epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                    epsilon_hat_norm = self.norm(epsilon_hat)
                    delta_gamma = epsilon_hat_norm + (self.dim * self.lam[p] + 2. * self.mu[p]) / (2. * self.mu[p]) * (shifted_trace) * self.friction_alpha[None]
                    epsilon -= (ti.max(delta_gamma, 0) / epsilon_hat_norm) * epsilon_hat
                sig_out = ti.exp(epsilon)
                self.sig_out[p] = sig_out
                self.F[p, s+1] = self.U[p] @ self.make_matrix_from_diag(sig_out) @ self.V[p].transpose()
            elif ti.static(self.material == self.von_mises):
                b_trial = sig ** 2
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                s_trial = 2 * self.mu[p] * epsilon_hat
                s_trial_norm = self.norm(s_trial)
                y = s_trial_norm - ti.sqrt(2./3) * self.yield_stress[None]
                sig_out = ti.Vector.zero(self.dtype, self.dim)
                if y > 0:
                    mu_hat = self.mu[p] * b_trial.sum() / self.dim
                    s_new_norm = s_trial_norm - y / (1 + self.plastic_viscosity[None] / (2 * mu_hat * self.dt[None]))
                    s_new = (s_new_norm / s_trial_norm) * s_trial
                    H = s_new / (2 * self.mu[p]) + trace_epsilon / self.dim
                    sig_out = ti.exp(H)
                else:
                    sig_out = sig
                self.sig_out[p] = sig_out
                self.F[p, s+1] = self.U[p] @ self.make_matrix_from_diag(sig_out) @ self.V[p].transpose()
            else:
                self.F[p, s+1] = self.F_tmp[p]

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.func
    def clamp(self, a, eps=1e-6):
        if a>=0:
            a = ti.max(a, eps)
        else:
            a = ti.min(a, -eps)
        return a

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

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
            if ti.static(self.material == self.elasticity):
                J = new_F.determinant()
                scale = self.lam[p] * ti.log(J) - self.mu[p]
                grad_v = self.C[p, s]
                epsilon = 0.5 * (grad_v + grad_v.transpose())
                stress = self.damping_coeff * epsilon * J + self.mu[p] * (new_F @ new_F.transpose()) + scale * ti.Matrix.identity(self.dtype, self.dim)
            elif ti.static(self.material == self.viscous_fluid):
                J = new_F[0,0]
                kappa = .6666666666 * self.mu[p] + self.lam[p]
                stress = kappa * ti.Matrix.identity(self.dtype, self.dim) * (J - 1 / (J ** 6))
                grad_v = self.C[p, s]
                epsilon = 0.5 * (grad_v + grad_v.transpose())
                stress += self.mu[p] * epsilon * J
            else:
                log_sig = ti.log(self.sig_out[p])
                tau = 2 * self.mu[p] * log_sig + self.lam[p] * log_sig.sum()
                stress = self.U[p] @ self.make_matrix_from_diag(tau) @ self.U[p].transpose()

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
                for i in ti.static(range(len(self.analytic_collision))):
                    v_out = self.analytic_collision[i](I, v_out)
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

    def add_surface_collider(self, point, normal, surface=surface_sticky):
        point = list(point)
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        @ti.func
        def get_velocity(I, v):
            offset = I.cast(self.dtype) * self.dx[None] - ti.Vector(point)
            n = ti.Vector(normal)
            if offset.dot(n) <= 1e-6:
                if ti.static(surface == self.surface_sticky):
                    v = ti.Vector.zero(self.dtype, self.dim)
                else:
                    normal_component = n.dot(v)
                    if ti.static(surface == self.surface_slip):
                        v = v - n * normal_component
                    else:
                        v = v - n * min(normal_component, 0)
            return v

        self.analytic_collision.append(get_velocity)

# --- Simulation Setup ---

def run_simulation(output_dir="workspace/taichi_dataset/output_domino", material_type='elasticity'):
    # Parameters
    dtype = ti.f32
    dt = 2e-5 # Reduced dt to satisfy CFL with high stiffness (sound speed ~160m/s)
    frame_dt = 1/60.0 # Standard 60fps
    dx = 0.005 # Higher resolution for dominos
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 64
    
    # --- Material Setup ---
    # Default values
    friction_angle = 0.0
    cohesion = 0.0
    yield_stress = 0.0
    plastic_viscosity = 0.0

    print(f"Material Type: {material_type}")

    if material_type == 'elasticity':
        # Elasticity
        material = MPMSimulator.elasticity
        rho = 2000.0 # Heavier
        E = 5e7 # Much Stiffer to prevent bending
        nu = 0.2
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    elif material_type == 'plasticine':
        material = MPMSimulator.von_mises
        rho = 1000.0
        E = 1e4
        nu = 0.25
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        yield_stress = 1000.0
        plastic_viscosity = 0.0
    
    elif material_type == 'sand':
        material = MPMSimulator.drucker_prager
        rho = 1800.0
        E = 1e6
        nu = 0.3
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        friction_angle = 25.0 
        cohesion = 0.0
    else:
        # Fallback or other materials
        material = MPMSimulator.elasticity
        rho = 1000.0
        E = 1e5
        nu = 0.25
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    
    # --- Generate Dominos ---
    # 3 Dominos
    # 1. Rotated (Pusher)
    # 2. Standing
    # 3. Standing
    
    all_points = []
    
    # Dimensions (Width, Height, Thickness) - swapped depth/width depending on orientation
    # Let's say aligned along X axis.
    # Thickness = 0.02 (X size)
    # Height = 0.1 (Y size)
    # Width = 0.05 (Z size)
    
    d_thickness = 0.04
    d_height = 0.2
    d_width = 0.1
    spacing = 0.08 # Closer

    
    # Use trimesh to generate boxes and voxelize them
    print("Generating dominos...")
    
    def get_voxels(extents, translation, rotation_angle_z_deg=0.0):
        mesh = trimesh.creation.box(extents=extents)
        if rotation_angle_z_deg != 0:
            rot = trimesh.transformations.rotation_matrix(np.radians(rotation_angle_z_deg), [0, 0, 1])
            mesh.apply_transform(rot)
            
        mesh.apply_translation(translation)
        
        # Voxelize
        voxel = mesh.voxelized(pitch=dx)
        voxel = voxel.fill()
        points = voxel.points.astype(np.float32)
        
        # Add slight jitter
        jitter = (np.random.rand(*points.shape) - 0.5) * dx * 0.5
        points += jitter
        return points

    def get_sphere(radius, translation):
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=3)
        mesh.apply_translation(translation)
        voxel = mesh.voxelized(pitch=dx)
        voxel = voxel.fill()
        points = voxel.points.astype(np.float32)
        # Add slight jitter
        jitter = (np.random.rand(*points.shape) - 0.5) * dx * 0.5
        points += jitter
        return points

    # Domino 1: Upright
    # Start at x=0.2
    points1 = get_voxels([d_thickness, d_height, d_width], [0.2, d_height/2, 0.5])
    all_points.append(points1)
    
    # Domino 2: Upright
    points2 = get_voxels([d_thickness, d_height, d_width], [0.2 + spacing, d_height/2, 0.5])
    all_points.append(points2)
    
    # Domino 3: Upright
    points3 = get_voxels([d_thickness, d_height, d_width], [0.2 + spacing*2, d_height/2, 0.5])
    all_points.append(points3)

    # Ball
    ball_radius = 0.03
    points_ball = get_sphere(ball_radius, [0.2 - 0.1, d_height - 0.05, 0.5]) # Left of first domino, hitting top
    all_points.append(points_ball)

    init_pos = np.concatenate(all_points, axis=0)
    n_particles_est = len(init_pos)
    
    # Calculate number of ball particles for assigning initial velocity
    n_ball_particles = len(points_ball)
    n_domino_particles = n_particles_est - n_ball_particles # Ball is last added
    
    print(f"Generated {n_particles_est} particles for dominos and ball.")

    # Fields
    num_particles = ti.field(ti.i32, shape=())
    
    # Particle layout
    particle = ti.root.dynamic(ti.i, 2**20, particle_chunk_size)
    
    # Simulator
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, gravity=[0, -9.8, 0], 
                       material=material, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3 
    
    if material == MPMSimulator.drucker_prager:
        sin_phi = math.sin(math.radians(friction_angle))
        alpha = math.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        sim.friction_alpha[None] = alpha
        sim.cohesion[None] = cohesion
    elif material == MPMSimulator.von_mises:
        sim.yield_stress[None] = yield_stress
        sim.plastic_viscosity[None] = plastic_viscosity

    @ti.kernel
    def init_particles_from_field(n: int, pos_field: ti.template(), n_domino: int):
        num_particles[None] = n
        for i in range(n):
            sim.x[i, 0] = pos_field[i]
            if i >= n_domino:
                sim.v[i, 0] = ti.Vector([8.0, 0.0, 0.0]) # Ball velocity towards right
            else:
                sim.v[i, 0] = ti.Vector([0.0, 0.0, 0.0])
            sim.F[i, 0] = ti.Matrix.identity(dtype, 3)
            sim.C[i, 0] = ti.Matrix.zero(dtype, 3, 3)
            sim.p_mass[i] = sim.p_vol[None] * rho
            # Make ball heavier?
            if i >= n_domino:
                 sim.p_mass[i] *= 5.0

            sim.mu[i] = mu
            sim.lam[i] = lam

    # Upload particles
    pos_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    pos_field.from_numpy(init_pos)
    init_particles_from_field(n_particles_est, pos_field, n_domino_particles)
    
    # Add boundary (Floor)
    # Changed from surface_sticky to surface_slip to allow dominos to fall (rotate/slide)
    # surface_sticky forces velocity to 0 at the ground, making them "stuck".
    sim.add_surface_collider(point=[0, 0.0, 0], normal=[0, 1, 0], surface=MPMSimulator.surface_sticky)

    # Run loop
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata
    initial_pos_np = sim.x.to_numpy()[:num_particles[None], 0, :]
    xyz_min = initial_pos_np.min(axis=0).tolist()
    xyz_max = initial_pos_np.max(axis=0).tolist()
    
    simulation_frames = 120 # Need more frames to see them fall
    gravity_vec = [0, -9.8, 0]
    
    bc_data = {
        "ground": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0]
    }
    
    metadata = {
        "data": {
            "xyz_min": xyz_min,
            "xyz_max": xyz_max
        },
        "n_frames": simulation_frames,
        "gravity": gravity_vec,
        "bc": bc_data
    }
    
    json_output_path = os.path.join(output_dir, "metadata.json")
    with open(json_output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Starting simulation... Outputting to {output_dir}")
    for frame in range(simulation_frames): 
        sim.advance(frame)
        pos = sim.x.to_numpy()[:num_particles[None], 0, :] 
        np.save(os.path.join(output_dir, f"frame_{frame:04d}.npy"), pos)
        if frame % 10 == 0:
            print(f"Frame {frame} completed. Particles: {num_particles[None]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to elasticity as requested
    parser.add_argument('-o', '--output_dir', type=str, default="/root/workspace/taichi_dataset/particles_output/output_domino", help="Output directory")
    parser.add_argument('-m', '--material', type=str, default="elasticity", help="Material type")
    args = parser.parse_args()
    run_simulation(args.output_dir, args.material)
