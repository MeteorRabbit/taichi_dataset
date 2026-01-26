# 保龄球简单模拟

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

def load_mesh_and_fill(filename, dx, scale=1.0, jitter_ratio=0.4):
    print(f"Loading mesh from {filename} for particle generation...")
    mesh = trimesh.load(filename)
    
    if scale != 1.0:
        mesh.apply_scale(scale)

    # 1. Get Surface Vertices
    # These will be the first N particles
    surface_verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    n_verts = len(surface_verts)
    
    # 2. Fill Interior
    print("Voxelizing mesh to fill interior...")
    # Use a slightly smaller pitch ensures we get particles inside
    voxel_grid = mesh.voxelized(pitch=dx)
    voxel_grid = voxel_grid.fill()
    interior_points = voxel_grid.points.astype(np.float32)

    # Remove interior points that are too close to surface vertices to prevent density explosion
    # A simple heuristic: if a point is within dx/2 of a surface vertex, drop it?
    # For simplicity in MPM, we can just merge them. The density will balance out quickly.
    # But filtering is better. Let's do a simple check or just rely on jitter.
    # To keep it fast without scipy cKDTree, we'll just skip detailed filtering for now 
    # and rely on the fact that voxelization usually aligns to grid, acting distinct from verts.
    
    if jitter_ratio > 0:
        jitter = (np.random.rand(*interior_points.shape) - 0.5) * dx * jitter_ratio
        interior_points += jitter

    print(f"Mesh info: {n_verts} vertices, {len(interior_points)} interior particles.")
    
    # Combine: Vertices FIRST, then interior
    all_points = np.vstack((surface_verts, interior_points))
    
    return all_points, faces, n_verts

# Helper to place a mesh-based object in the scene
def place_object(base_points, base_faces, n_verts, offset, existing_vert_count):
    # base_points: (N, 3) all particles (verts + interior)
    # base_faces: (F, 3) original face indices
    # n_verts: number of vertices at the beginning of base_points
    # existing_vert_count: how many vertices are already in the global mesh (vertex offset for faces)
    
    # Points
    new_points = base_points + np.array(offset, dtype=np.float32)
    
    # Faces (only update indices, don't change count)
    new_faces = base_faces + existing_vert_count
    
    return new_points, new_faces

# --- Simulation Setup ---

def run_simulation(output_dir="workspace/taichi/output_sim", material_type='elasticity'):
    # Parameters
    dtype = ti.f32
    dt = 1e-4
    frame_dt = 1/60.0
    dx = 0.01
    inv_dx = 1.0 / dx
    particle_chunk_size = 2**14
    cuda_chunk_size = 16 # Reduce chunk size to save memory (was 64)
    
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
        rho = 1800.0 # PAC-NeRF default doesn't specify rho in sand/default.py, but usually sand is heavier. 
                     # Wait, elastic/0.py has rho=1000. Let's stick to 1800 or 1000. 
                     # The user's previous code had 1800. Let's use 1800 for realism or 1000 if strictly following default.
                     # PAC-NeRF sand/default.py doesn't set rho, so it might use a global default.
                     # Let's use 1800 as it's more physically correct for sand.
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
    # Use billiard.ply
    mesh_path = "workspace/taichi_dataset/meshes/billiard.ply" # Check path if needed
    if not os.path.exists(mesh_path):
        # Fallback relative path check
        mesh_path = "meshes/billiard.ply"
        if not os.path.exists(mesh_path):
             mesh_path = "../meshes/billiard.ply"

    # Scale to match original radius=0.08 approx. 
    # Billiard PLY size is unknown, let's assume unit sphere or similar.
    # User said "radius=0.08" in previous code.
    # Let's load it once and measure/scale.
    temp_mesh = trimesh.load(mesh_path)
    # Get bounding box extents
    extents = temp_mesh.bounding_box.extents
    max_extent = max(extents)
    desired_radius = 0.08
    scale_factor = (desired_radius * 2) / max_extent
    print(f"Auto-scaling mesh by factor {scale_factor} to match diameter {desired_radius * 2}")

    # Prepare logic to reconstruct all meshes
    # We will accumulate all vertices and faces for export
    all_particles_list = [] # List of numpy arrays
    all_vel_list = []
    
    # Metadata for reconstruction
    # List of objects, each: { 'n_verts': int, 'n_total': int, 'faces': np.array }
    object_metadata = []
    
    # Load base mesh particles once
    # Note: center of mesh is assumed to be 0,0,0 after load? trimesh usually loads as is.
    # We might need to center it.
    
    base_particles, base_faces, n_verts_per_obj = load_mesh_and_fill(mesh_path, dx/2, scale=scale_factor, jitter_ratio=0.4)
    # Center the base particles to 0,0,0
    center_offset = -np.mean(base_particles[:n_verts_per_obj], axis=0) # Center based on vertices
    base_particles += center_offset
    
    
    # helper for tracking
    current_global_vert_count = 0
    global_faces = [] # List of face arrays
    
    def add_ball_to_scene(pos, vel):
        nonlocal current_global_vert_count
        
        p, f = place_object(base_particles, base_faces, n_verts_per_obj, pos, current_global_vert_count)
        v = np.full(p.shape, vel, dtype=np.float32)
        
        all_particles_list.append(p)
        all_vel_list.append(v)
        global_faces.append(f)
        
        # Track this object
        object_metadata.append({
            'type': 'ball',
            'n_verts': n_verts_per_obj,
            'n_total': len(p)
        })
        
        current_global_vert_count += n_verts_per_obj

    # Sphere 1 (Thrower)
    c1_pos = [0.3, desired_radius + dx, 0.5]
    c1_vel = [7.5, 0.0, 0.0]
    add_ball_to_scene(c1_pos, c1_vel)
    
    # Sphere 2 (Target) - 6 spheres in triangle
    target_distance = 0.5
    base_target_x = 0.3 + target_distance
    base_target_z = 0.5 + 0.1
    
    target_positions = []
    sphere_spacing = 2 * desired_radius + 0.02
    row_dx = sphere_spacing * math.sqrt(3) / 2
    row_dz = sphere_spacing / 2
    
    # Row 1
    target_positions.append([base_target_x, desired_radius + dx, base_target_z])
    # Row 2
    target_positions.append([base_target_x + row_dx, desired_radius + dx, base_target_z - row_dz])
    target_positions.append([base_target_x + row_dx, desired_radius + dx, base_target_z + row_dz])
    # Row 3
    target_positions.append([base_target_x + 2 * row_dx, desired_radius + dx, base_target_z - sphere_spacing])
    target_positions.append([base_target_x + 2 * row_dx, desired_radius + dx, base_target_z])
    target_positions.append([base_target_x + 2 * row_dx, desired_radius + dx, base_target_z + sphere_spacing])
    
    for pos in target_positions:
        add_ball_to_scene(pos, [0.0, 0.0, 0.0])

    init_pos = np.vstack(all_particles_list)
    init_vel = np.vstack(all_vel_list)
    n_particles_est = len(init_pos)
    print(f"Initializing Curling Scenario with {n_particles_est} particles")
    
    # Pre-merge faces for cheaper export loop
    all_faces_merged = np.vstack(global_faces)

    # Fields
    num_particles = ti.field(ti.i32, shape=())
    
    # Particle layout
    # Reduce max particles to 2^19 (approx 500k) to save memory, as we only use ~120k
    particle = ti.root.dynamic(ti.i, 2**19, particle_chunk_size)
    
    # Simulator
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, gravity=[0, -9.8, 0], 
                       material=material, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3 # Approximate volume
    
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
        
        # Get current positions from Taichi to Numpy
        # Note: sim.x contains ALL particles (verts + interior)
        current_pos_all = sim.x.to_numpy()[:num_particles[None], 0, :]
        
        # 1. Save standard particle NPY (optional, but good for debug)
        np.save(os.path.join(output_dir, f"frame_{frame:04d}_particles.npy"), current_pos_all)
        
        # 2. Reconstruct PLY Mesh
        # We need to extract the vertices for each object and assemble them
        all_verts_combined = []
        
        start_idx = 0
        for obj in object_metadata:
            # The vertices are the first 'n_verts' particles of this object's chunk
            n_total = obj['n_total']
            n_verts = obj['n_verts']
            
            # Extract vertices
            obj_verts = current_pos_all[start_idx : start_idx + n_verts]
            all_verts_combined.append(obj_verts)
            
            start_idx += n_total
            
        all_verts_combined = np.vstack(all_verts_combined)
        
        # Create Trimesh object
        mesh_out = trimesh.Trimesh(vertices=all_verts_combined, faces=all_faces_merged)
        
        # Export
        ply_path = os.path.join(output_dir, f"frame_{frame:04d}.ply")
        mesh_out.export(ply_path)
        
        print(f"Frame {frame} completed. Particles: {num_particles[None]}. Saved PLY to {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="workspace/taichi_dataset/particles_output/output_billiard_mpm", help="Output directory for simulation results")
    parser.add_argument('-m', '--material', type=str, default="elasticity", 
                        choices=['elasticity', 'plasticine', 'sand', 'newtonian', 'non_newtonian', 'toothpaste_custom'],
                        help="Material type for simulation")
    args = parser.parse_args()
    run_simulation(args.output_dir, args.material)
