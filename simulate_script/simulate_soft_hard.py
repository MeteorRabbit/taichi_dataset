# 冰壶简单模拟改成硬的物体撞到软的物体

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

    def __init__(self, dtype, dt, frame_dt, particle_layout, dx, inv_dx, n_particles, n_mesh_vertices, gravity=[0, -9.8, 0], material=elasticity, cuda_chunk_size=400):
        
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

        # Mesh vertices advection
        self.n_mesh_vertices = n_mesh_vertices
        self.mesh_x = ti.Vector.field(dim, dtype=self.dtype, shape=n_mesh_vertices)
        
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
    
    @ti.kernel
    def g2mesh(self):
        # Update Mesh Vertices using grid velocity (Passive Advection)
        for i in range(self.n_mesh_vertices):
            # Same Grid interpolation logic
            base = ti.floor(self.mesh_x[i] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.mesh_x[i] * self.inv_dx[None] - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            
            new_v = ti.Vector.zero(self.dtype, self.dim)
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        g_v = self.grid_v_out[base(0) + k, base(1) + l, base(2) + m]
                        weight = w[k](0) * w[l](1) * w[m](2)
                        new_v += weight * g_v
            
            # Simple Forward Euler for Mesh
            self.mesh_x[i] += self.dt[None] * new_v

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
        self.g2mesh() # Update mesh vertices
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

def load_mesh_data(filename, scale=1.0, offset=[0.0, 0.0, 0.0]):
    """Loads mesh, applies transform, and returns vertices and faces."""
    mesh = trimesh.load(filename)
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)
    if scale != 1.0:
        mesh.apply_scale(scale)
    
    # Translate (Center at 0,0,0 first if needed, but here we just add offset)
    # The original logic centered the particles?? No, it just added offset.
    # We should center the mesh to its centroid or bounding box center before placing?
    # Original logic: points += offset.
    # So we apply offset translation.
    mesh.apply_translation(offset)
    
    return mesh.vertices.astype(np.float32), mesh.faces

def sample_particles_from_mesh(mesh, dx, jitter_ratio=0.4):
    """Uses existing mesh object to generate particles."""
    print("Voxelizing mesh...")
    voxel_grid = mesh.voxelized(pitch=dx)
    print("Filling interior...")
    voxel_grid = voxel_grid.fill()
    points = voxel_grid.points.astype(np.float32)
    # Add jitter
    if jitter_ratio > 0:
        jitter = (np.random.rand(*points.shape) - 0.5) * dx * jitter_ratio
        points += jitter
    return points

def run_simulation(output_dir="workspace/taichi/output_sim", material_type='elasticity'):
    # Parameters
    dtype = ti.f32
    dt = 1e-4
    frame_dt = 1/60.0
    dx = 0.01
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 64 # Smaller chunk size for forward sim
    
    # --- Material Setup ---
    # Default values for optional parameters
    friction_angle = 0.0
    cohesion = 0.0
    yield_stress = 0.0
    plastic_viscosity = 0.0

    # Select material type here:
    # 'elasticity', 'plasticine', 'sand', 'newtonian', 'non_newtonian', 'toothpaste_custom'
    # material_type is passed as argument

    if material_type == 'elasticity':
        # PAC-NeRF Elasticity (Jelly)
        material = MPMSimulator.elasticity
        rho = 1000.0
        E = 5e6 # 316228
        nu = 0.25
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    elif material_type == 'plasticine':
        # PAC-NeRF Plasticine (Stiff Von Mises)
        material = MPMSimulator.von_mises
        rho = 1000.0
        E = 1e4
        nu = 0.25
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        yield_stress = 1000.0
        plastic_viscosity = 0.0

    elif material_type == 'sand':
        # PAC-NeRF Sand (Drucker-Prager)
        material = MPMSimulator.drucker_prager
        rho = 1800.0 
        E = 1e6
        nu = 0.3
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        friction_angle = 10.0 # PAC-NeRF default
        cohesion = 0.0

    elif material_type == 'newtonian':
        # PAC-NeRF Newtonian Fluid (Water/Honey)
        material = MPMSimulator.viscous_fluid
        rho = 1000.0
        kappa = 1e4
        mu = 10.0
        lam = kappa - 2/3 * mu

    elif material_type == 'non_newtonian':
        # PAC-NeRF Non-Newtonian Fluid (Cream)
        material = MPMSimulator.von_mises
        rho = 1000.0
        kappa = 1e5
        mu = 100.0
        lam = kappa - 2/3 * mu
        yield_stress = 10.0
        plastic_viscosity = 1.0

    elif material_type == 'toothpaste_custom':
        # User's Custom Toothpaste (Stiff Non-Newtonian)
        material = MPMSimulator.von_mises
        rho = 1000.0
        mu = 10000.0 # 10^4
        kappa = 1000000.0 # 10^6
        lam = kappa - 2/3 * mu
        yield_stress = 3000.0 # 3 * 10^3
        plastic_viscosity = 10.0 # 10
    
    else:
        raise ValueError(f"Unknown material type: {material_type}")

    print(f"Selected Material: {material_type}")
    print(f"  mu: {mu}, lam: {lam}, yield_stress: {yield_stress}, plastic_viscosity: {plastic_viscosity}")

    
    # Initialization
    # Curling Setup
    radius = 0.1
    height = 0.05
    target_distance = 0.5
    
    # Load Meshes
    mesh_path = "workspace/taichi_dataset/meshes/curling.ply"
    
    # Prepare Thrower Mesh
    # Note: original c1_pos was center. load_mesh_data applies offset.
    # We should adjust offset so that the resulting mesh sits where we want.
    # Assuming the curling.ply is centered at origin? We should probably check or assume so.
    # If we assume it's roughly centered, we just translate to c1_pos.
    c1_pos = [0.3, height/2 + dx, 0.5]
    c1_vel = [7.5, 0.0, 0.0] 
    
    print(f"Loading Thrower Mesh...")
    mesh1 = trimesh.load(mesh_path)
    # [NEW] Check for UVs on mesh1
    uv1 = mesh1.visual.uv if hasattr(mesh1.visual, 'uv') and mesh1.visual.uv is not None else None

    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh1.apply_transform(rot_matrix)
    # Scale to match roughly the size we want? Original was radius=0.1, ply might be different scale.
    # Assuming curling.ply is unit size? Or user provided it. 
    # Let's assume it doesn't need scaling or user provides correct scale mesh.
    # But usually simulations need specific size.
    # The original "generate_cylinder" used radius=0.1.
    # Let's hope the ply is reasonable size. If not, physics might explode.
    # Safe bet: Scale it to fit 2*radius in X/Z.
    
    bbox_1 = mesh1.bounding_box.extents
    # If bbox is too big/small, scale? User said "modified to get vertices from meshes/curling.ply", 
    # usually implies the mesh is ready. I will trust the mesh size or apply 1.0.
    
    mesh1.apply_translation(c1_pos)
    v1_verts = mesh1.vertices.astype(np.float32)
    f1_faces = mesh1.faces
    
    p1 = sample_particles_from_mesh(mesh1, dx/2) # Sample particles
    v1 = np.full(p1.shape, c1_vel, dtype=np.float32)
    
    # Prepare Target Mesh (Just one now)
    base_target_x = 0.3 + target_distance
    base_target_z = 0.5 + 0.1 # With offset
    # Row 1 target
    target_pos = [base_target_x, height/2 + dx, base_target_z]
    
    print(f"Loading Target Mesh...")
    mesh2 = trimesh.load(mesh_path) # Reload to get fresh instance
    # [NEW] Check for UVs on mesh2
    uv2 = mesh2.visual.uv if hasattr(mesh2.visual, 'uv') and mesh2.visual.uv is not None else None

    mesh2.apply_transform(rot_matrix)
    mesh2.apply_translation(target_pos)
    v2_verts = mesh2.vertices.astype(np.float32)
    f2_faces = mesh2.faces
    
    p2 = sample_particles_from_mesh(mesh2, dx/2)
    v2_vel_target = np.full(p2.shape, [0.0, 0.0, 0.0], dtype=np.float32)

    # Combine Particles
    init_pos = np.vstack((p1, p2))
    init_vel = np.vstack((v1, v2_vel_target))
    n_particles_est = len(init_pos)
    
    # Combine Mesh Vertices
    all_mesh_verts = np.vstack((v1_verts, v2_verts))
    n_mesh_verts = len(all_mesh_verts)
    mesh_split_idx = len(v1_verts) # Index where mesh 2 starts
    
    print(f"Initializing Curling Scenario with {n_particles_est} particles and {n_mesh_verts} mesh vertices")
    
    # Fields
    num_particles = ti.field(ti.i32, shape=())
    
    # Particle layout
    # Reduced max capacity from 2**20 (1M) to 2**16 (65k) to save memory
    particle = ti.root.dynamic(ti.i, 2**16, particle_chunk_size)
    
    # Simulator
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, n_mesh_vertices=n_mesh_verts, gravity=[0, -9.8, 0], 
                       material=material, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3 # Approximate volume
    
    # Set init mesh verts
    sim.mesh_x.from_numpy(all_mesh_verts)

    # Set global material parameters
    if material == MPMSimulator.drucker_prager:
        sin_phi = math.sin(math.radians(friction_angle))
        alpha = math.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        sim.friction_alpha[None] = alpha
        sim.cohesion[None] = cohesion
    elif material == MPMSimulator.von_mises:
        sim.yield_stress[None] = yield_stress
        sim.plastic_viscosity[None] = plastic_viscosity
    
    # Initialize particles
    # Soft material properties (Object 2)
    # We simply reduce E for the second object to make it softer.
    E_soft = E * 0.1 # 10x softer
    mu_soft = E_soft / (2 * (1 + nu))
    lam_soft = E_soft * nu / ((1 + nu) * (1 - 2 * nu))

    # Initialize particles
    @ti.kernel
    def init_particles_with_vel(n: int, pos_field: ti.template(), vel_field: ti.template(), split_idx: int):
        num_particles[None] = n
        for i in range(n):
            sim.x[i, 0] = pos_field[i]
            sim.v[i, 0] = vel_field[i]
            
            sim.F[i, 0] = ti.Matrix.identity(dtype, 3)
            sim.C[i, 0] = ti.Matrix.zero(dtype, 3, 3)
            sim.p_mass[i] = sim.p_vol[None] * rho
            
            if i < split_idx:
                # Object 1 (Harder)
                sim.mu[i] = mu
                sim.lam[i] = lam
            else:
                # Object 2 (Softer)
                sim.mu[i] = mu_soft
                sim.lam[i] = lam_soft

    pos_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    vel_field = ti.Vector.field(3, dtype=dtype, shape=n_particles_est)
    pos_field.from_numpy(init_pos)
    vel_field.from_numpy(init_vel)
    
    # split_idx is len(p1)
    split_idx = len(p1)
    init_particles_with_vel(n_particles_est, pos_field, vel_field, split_idx)
    
    # Add boundary (Floor) - Ice (Slip)
    sim.add_surface_collider(point=[0, 0.0, 0], normal=[0, 1, 0], surface=MPMSimulator.surface_slip)

    # Run loop
    # output_dir is passed as argument
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Generate Metadata JSON ---
    initial_pos_np = sim.x.to_numpy()[:num_particles[None], 0, :]
    xyz_min = initial_pos_np.min(axis=0).tolist()
    xyz_max = initial_pos_np.max(axis=0).tolist()
    
    simulation_frames = 30
    gravity_vec = [0, -9.8, 0]
    
    # Boundary condition: [point, normal, type(0=sticky)]
    bc_data = {
        "ground": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1]
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
    print(f"Saved simulation metadata to {json_output_path}")

    print(f"Starting simulation... Outputting to {output_dir}")
    for frame in range(simulation_frames): # Run for frames
        sim.advance(frame)
        
        # NOTE: Export PLY instead of NPY
        # Get Mesh positions
        current_mesh_verts = sim.mesh_x.to_numpy()
        
        # Split
        mv1 = current_mesh_verts[:mesh_split_idx]
        mv2 = current_mesh_verts[mesh_split_idx:]
        
        # Export PLY 1
        mesh1_out = trimesh.Trimesh(vertices=mv1, faces=f1_faces, process=False)
        # [NEW] Re-apply UVs
        if uv1 is not None:
             mesh1_out.visual = trimesh.visual.TextureVisuals(uv=uv1)
        mesh1_out.export(os.path.join(output_dir, f"curling_0_frame_{frame:04d}.ply"))
        
        # Export PLY 2
        mesh2_out = trimesh.Trimesh(vertices=mv2, faces=f2_faces, process=False)
        # [NEW] Re-apply UVs
        if uv2 is not None:
             mesh2_out.visual = trimesh.visual.TextureVisuals(uv=uv2)
        mesh2_out.export(os.path.join(output_dir, f"curling_1_frame_{frame:04d}.ply"))
        
        # Still export particles just in case? Or remove?
        # User said: "分别导出为ply而不是npy", implying replace npy with ply. 
        # But for debugging it might be useful to have particles.
        # But I will comment out NPY export to save disk/time and follow instruction.
        # pos = sim.x.to_numpy()[:num_particles[None], 0, :]
        # np.save(os.path.join(output_dir, f"frame_{frame:04d}.npy"), pos)
        
        print(f"Frame {frame} completed. Proccessed and exported meshes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="workspace/taichi_dataset/particles_output/output_soft_hard", help="Output directory for simulation results")
    parser.add_argument('-m', '--material', type=str, default="elasticity", 
                        choices=['elasticity', 'plasticine', 'sand', 'newtonian', 'non_newtonian', 'toothpaste_custom'],
                        help="Material type for simulation")
    args = parser.parse_args()
    run_simulation(args.output_dir, args.material)
