# 枕头保龄球模拟：硬的物体（保龄球）撞到软的物体（枕头）
# Refactored to match simulate_billiard_mpm.py architecture (Active Vertices + Metadata Loop)

import taichi as ti
import numpy as np
import math
import os
import trimesh
import argparse
import json

# Initialize Taichi
ti.init(arch=ti.cuda, device_memory_fraction=0.6)

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

    # Removed n_mesh_vertices and mesh_x as we now treat vertices as particles directly
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
        
        self.damping_coeff = 0

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
    
    # Removed g2mesh as we use particles as vertices

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
        # Removed g2mesh()
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

# [NEW] Helper from simulate_billiard_mpm.py
def load_mesh_and_fill(filename, dx, scale=1.0, jitter_ratio=0.4):
    print(f"Loading mesh from {filename} for particle generation...")
    mesh = trimesh.load(filename)
    
    # Standard Rotation for consistency (Z-up)
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)

    if scale != 1.0:
        mesh.apply_scale(scale)

    # 1. Get Surface Vertices (These will be the first N particles)
    surface_verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    n_verts = len(surface_verts)
    
    # Get UVs
    uvs = mesh.visual.uv if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None else None
    
    # 2. Fill Interior
    print("Voxelizing mesh to fill interior...")
    try:
        voxel_grid = mesh.voxelized(pitch=dx)
        voxel_grid = voxel_grid.fill()
        interior_points = voxel_grid.points.astype(np.float32)
    except Exception as e:
        print(f"Voxelization failed, using surface only: {e}")
        interior_points = np.array([], dtype=np.float32).reshape(0,3)

    if jitter_ratio > 0 and len(interior_points) > 0:
        jitter = (np.random.rand(*interior_points.shape) - 0.5) * dx * jitter_ratio
        interior_points += jitter

    print(f"Mesh info: {n_verts} vertices, {len(interior_points)} interior particles.")
    
    # Combine: Vertices FIRST, then interior
    all_points = np.vstack((surface_verts, interior_points))
    
    return all_points, faces, n_verts, uvs

# Helper to place a mesh-based object in the scene
def place_object(base_points, base_faces, offset, existing_vert_count):
    # base_points: (N, 3) all particles
    # base_faces: (F, 3) original face indices
    # offset: position offset
    # existing_vert_count: index offset for faces
    
    # Points
    new_points = base_points + np.array(offset, dtype=np.float32)
    # Faces (only update indices)
    new_faces = base_faces + existing_vert_count # Actually, we export individually now, so faces might not need offset if exported per object?
    # Wait, billiard script adds offset to faces because it might merge them? 
    # Actually billiard script uses `base_faces` directly in export because it exports INDIVIDUAL PLYs.
    # So we don't strictly need to offset faces if we don't merge them. 
    # But let's keep faces 'local' to the object (0-indexed) for individual export.
    return new_points, base_faces

def run_simulation(output_dir="/root/workspace/taichi_dataset/particles_output/output_pillow_bowling", material_type='elasticity'):
    # Parameters
    dtype = ti.f32
    dt = 1e-4
    frame_dt = 1/60.0
    dx = 0.01
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 64
    
    # Default values
    friction_angle = 0.0
    cohesion = 0.0
    yield_stress = 0.0
    plastic_viscosity = 0.0

    # Initialize Taichi parameters based on material logic (Hardcoded for this task mainly, but supporting structure)
    # Note: We will set mu/lam per particle.
    
    # Material Props
    rho_bowling = 2000.0
    E_bowling = 1e7
    nu_bowling = 0.2
    mu_1 = E_bowling / (2 * (1 + nu_bowling))
    lam_1 = E_bowling * nu_bowling / ((1 + nu_bowling) * (1 - 2 * nu_bowling))

    rho_pillow = 500.0
    E_pillow = 1e4 # [MODIFIED] Made softer (was 5e4) to match soft material reference
    nu_pillow = 0.3
    mu_2 = E_pillow / (2 * (1 + nu_pillow))
    lam_2 = E_pillow * nu_pillow / ((1 + nu_pillow) * (1 - 2 * nu_pillow))

    # Base material for logic
    material = MPMSimulator.elasticity 
    
    # Container for all particles
    all_particles_list = []
    all_vel_list = []
    all_rho_list = []
    all_mu_list = []
    all_lam_list = []
    
    # Metadata for reconstruction
    object_metadata = []
    
    # Helper to load and add object
    def add_object(obj_name, filename, pos, vel, rho, mu, lam, scale=1.0):
        print(f"Adding object: {obj_name} from {filename}")
        pts, faces, n_verts, uvs = load_mesh_and_fill(filename, dx/2, scale=scale)
        
        # Center mesh before placing? 
        # load_mesh_and_fill applies Rotation. Trimesh usually keeps origin. 
        # We assume we place 'pos' as the new centroid or reference point.
        # Let's align centroid to 0,0,0 first.
        centroid = np.mean(pts[:n_verts], axis=0)
        pts -= centroid
        
        # Place
        pts += np.array(pos, dtype=np.float32)
        
        # Velocity
        v = np.full(pts.shape, vel, dtype=np.float32)
        
        # Material Params Arrays
        rho_arr = np.full(len(pts), rho * ((dx*0.5)**3), dtype=np.float32) # Store MASS (vol * rho)
        mu_arr = np.full(len(pts), mu, dtype=np.float32)
        lam_arr = np.full(len(pts), lam, dtype=np.float32)

        # Append
        all_particles_list.append(pts)
        all_vel_list.append(v)
        all_rho_list.append(rho_arr) # Actually this is mass
        all_mu_list.append(mu_arr)
        all_lam_list.append(lam_arr)
        
        object_metadata.append({
            'name': obj_name,
            'n_verts': n_verts,
            'n_total': len(pts),
            'faces': faces,
            'uvs': uvs
        })
    
    # 1. Bowling Ball
    bowling_path = "/root/workspace/taichi_dataset/meshes/Bowling/Bowling.ply"
    # Scale? 
    # Check bbox roughly. 
    # Let's trust the file or apply same logic as before (0.25m / current)
    # To be precise, let's load it quickly first to check scale or just use a safe factor.
    # Previous code had scale logic. I will re-implement it briefly.
    tmp = trimesh.load(bowling_path)
    ext = tmp.bounding_box.extents
    scale_1 = 0.25 / np.linalg.norm(ext) # Approx scale to 25cm
    
    # [MODIFIED] Position on ground (radius ~0.125)
    add_object(
        "bowling", bowling_path, 
        pos=[-0.5, 0.4, 0.5], vel=[5.0, -2.0, 0.0],
        rho=rho_bowling, mu=mu_1, lam=lam_1, scale=scale_1
    )
    
    # 2. Pillow
    pillow_path = "/root/workspace/taichi_dataset/meshes/Pillow/pillow.ply"
    tmp2 = trimesh.load(pillow_path)
    ext2 = tmp2.bounding_box.extents
    scale_2 = 0.5 / np.linalg.norm(ext2) # Approx scale to 50cm
    
    # [MODIFIED] Position on ground. Assuming thickness ~0.2, center at 0.1?
    # Let's put slightly higher to allow settling
    add_object(
        "pillow", pillow_path,
        pos=[0.0, 0.0, 0.5], vel=[0.0, 0.0, 0.0],
        rho=rho_pillow, mu=mu_2, lam=lam_2, scale=scale_2
    )
    
    # Finalize Init
    init_pos = np.vstack(all_particles_list)
    init_vel = np.vstack(all_vel_list)
    init_mass = np.concatenate(all_rho_list)
    init_mu = np.concatenate(all_mu_list)
    init_lam = np.concatenate(all_lam_list)
    
    n_particles_est = len(init_pos)
    print(f"Total particles: {n_particles_est}")

    # Simulator Init
    num_particles = ti.field(ti.i32, shape=())
    particle = ti.root.dynamic(ti.i, 2**19, particle_chunk_size)
    
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, gravity=[0, -9.8, 0], 
                       material=material, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3
    
    # Init Particles Kernel
    @ti.kernel
    def init_particles(n: int, pos: ti.template(), vel: ti.template(), mass: ti.template(), mu: ti.template(), lam: ti.template()):
        num_particles[None] = n
        for i in range(n):
            sim.x[i, 0] = pos[i]
            sim.v[i, 0] = vel[i]
            sim.p_mass[i] = mass[i]
            sim.mu[i] = mu[i]
            sim.lam[i] = lam[i]
            sim.F[i, 0] = ti.Matrix.identity(dtype, 3)
            sim.C[i, 0] = ti.Matrix.zero(dtype, 3, 3)

    # Convert to fields
    pos_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    vel_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    mass_field = ti.field(dtype=dtype, shape=n_particles_est)
    mu_field = ti.field(dtype=dtype, shape=n_particles_est)
    lam_field = ti.field(dtype=dtype, shape=n_particles_est)
    
    pos_field.from_numpy(init_pos)
    vel_field.from_numpy(init_vel)
    mass_field.from_numpy(init_mass)
    mu_field.from_numpy(init_mu)
    lam_field.from_numpy(init_lam)
    
    init_particles(n_particles_est, pos_field, vel_field, mass_field, mu_field, lam_field)
    
    # Boundary
    sim.add_surface_collider(point=[0, 0.0, 0], normal=[0, 1, 0], surface=MPMSimulator.surface_slip)

    # Run Loop
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata JSON
    xyz_min = init_pos.min(axis=0).tolist()
    xyz_max = init_pos.max(axis=0).tolist()
    # [MODIFIED] 30 frames
    simulation_frames = 30
    
    metadata = {
        "data": {"xyz_min": xyz_min, "xyz_max": xyz_max},
        "n_frames": simulation_frames,
        "gravity": [0, -9.8, 0],
        "bc": {"ground": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1]}
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Starting simulation... Output: {output_dir}")
    for frame in range(simulation_frames):
        sim.advance(frame)
        
        # Export
        current_pos_all = sim.x.to_numpy()[:num_particles[None], 0, :]
        
        start_idx = 0
        for obj in object_metadata: # Iterate over objects
            n_verts = obj['n_verts']
            n_total = obj['n_total']
            faces = obj['faces']
            uvs = obj['uvs']
            name = obj['name']
            
            # Extract Vertices (First N particles of the object)
            obj_verts = current_pos_all[start_idx : start_idx + n_verts]
            
            # Create Mesh
            mesh_out = trimesh.Trimesh(vertices=obj_verts, faces=faces, process=False)
            if uvs is not None:
                mesh_out.visual = trimesh.visual.TextureVisuals(uv=uvs)
            
            # Export
            # Using name_frame_XXXX.ply pattern
            out_name = f"{name}_frame_{frame:04d}.ply"
            mesh_out.export(os.path.join(output_dir, out_name))
            
            start_idx += n_total
            
        print(f"Frame {frame} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="/root/workspace/taichi_dataset/particles_output/output_pillow_bowling", help="Output directory")
    parser.add_argument('-m', '--material', type=str, default="elasticity", help="Material type")
    args = parser.parse_args()
    run_simulation(args.output_dir, args.material)
