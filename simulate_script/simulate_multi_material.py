# 地面改成沙地版本：cube模型，落到容器盛放的沙土体面上

import taichi as ti
import numpy as np
import math
import os
import trimesh
import argparse
import json

# Initialize Taichi
try:
    ti.init(arch=ti.cuda, device_memory_fraction=0.9)
except:
    pass

# Check which arch is actually running
# Fix: correct way to get current arch in older/newer taichi versions
current_arch = ti.cfg.arch
print(f"Taichi initialized on arch: {current_arch}")

if current_arch == ti.cuda:
    print("Taichi initialized with CUDA backend.")
else:
    print(f"Warning: GPU might not be available.")
    print("Optimization: Increasing voxel size and reducing particle count for CPU/Low-Memory safety.")

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
        self.friction_alpha = ti.field(dtype=self.dtype)
        self.cohesion = ti.field(dtype=self.dtype)
        self.yield_stress = ti.field(dtype=self.dtype)
        self.plastic_viscosity = ti.field(dtype=self.dtype)
        self.mu = ti.field(dtype=self.dtype)
        self.lam = ti.field(dtype=self.dtype)
        self.p_mass = ti.field(self.dtype)
        self.material_id = ti.field(ti.i32)
        
        # Temporary fields for computation
        self.F_tmp = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.U = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.V = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig = ti.Matrix.field(dim, dim, dtype=self.dtype)
        self.sig_out = ti.Vector.field(dim, dtype=self.dtype)

        self.particle.place(self.mu, self.lam, self.p_mass, self.material_id, self.friction_alpha, self.cohesion, self.yield_stress, self.plastic_viscosity, self.F_tmp, self.U, self.V, self.sig, self.sig_out)

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
            if self.material_id[p] == self.viscous_fluid:
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
            if self.material_id[p] == self.viscous_fluid:
                self.F[p, s+1][0,0] = ti.max(self.F_tmp[p][0,0], 0.05)
            elif self.material_id[p] == self.drucker_prager:
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                shifted_trace = trace_epsilon - self.cohesion[p] * self.dim
                if shifted_trace >= 0:
                    epsilon = ti.Vector.one(self.dtype, self.dim) * self.cohesion[p]
                else:
                    epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                    epsilon_hat_norm = self.norm(epsilon_hat)
                    delta_gamma = epsilon_hat_norm + (self.dim * self.lam[p] + 2. * self.mu[p]) / (2. * self.mu[p]) * (shifted_trace) * self.friction_alpha[p]
                    epsilon -= (ti.max(delta_gamma, 0) / epsilon_hat_norm) * epsilon_hat
                sig_out = ti.exp(epsilon)
                self.sig_out[p] = sig_out
                self.F[p, s+1] = self.U[p] @ self.make_matrix_from_diag(sig_out) @ self.V[p].transpose()
            elif self.material_id[p] == self.von_mises:
                b_trial = sig ** 2
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                s_trial = 2 * self.mu[p] * epsilon_hat
                s_trial_norm = self.norm(s_trial)
                y = s_trial_norm - ti.sqrt(2./3) * self.yield_stress[p]
                sig_out = ti.Vector.zero(self.dtype, self.dim)
                if y > 0:
                    mu_hat = self.mu[p] * b_trial.sum() / self.dim
                    s_new_norm = s_trial_norm - y / (1 + self.plastic_viscosity[p] / (2 * mu_hat * self.dt[None]))
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
            if self.material_id[p] == self.elasticity:
                J = new_F.determinant()
                scale = self.lam[p] * ti.log(J) - self.mu[p]
                grad_v = self.C[p, s]
                epsilon = 0.5 * (grad_v + grad_v.transpose())
                stress = self.damping_coeff * epsilon * J + self.mu[p] * (new_F @ new_F.transpose()) + scale * ti.Matrix.identity(self.dtype, self.dim)
            elif self.material_id[p] == self.viscous_fluid:
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

def load_mesh_vertices_and_faces(filename, scale=1.0, offset=[0.5, 0.5, 0.5]):
    print(f"Loading mesh from {filename}...")
    mesh = trimesh.load(filename)
    
    # Handle Scene object if returned
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Scene is empty")
        print("Scene loaded, using first geometry.")
        mesh = list(mesh.geometry.values())[0]

    # Rotate 90 degrees around X axis (to convert Z-up to Y-up)
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)
    
    if scale != 1.0:
        mesh.apply_scale(scale)
    
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces
    
    # --- Auto Centering ---
    # Calculate current centroid
    centroid = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    # Shift vertices to center around (0,0,0)
    vertices -= centroid
    print(f"Mesh centered. Shifted by {-centroid}")
    
    # Apply offset
    vertices += np.array(offset, dtype=np.float32)
    
    print(f"Loaded {len(vertices)} vertices and {len(faces)} faces.")
    return vertices, faces

def load_mesh_particles(filename, dx, scale=1.0, offset=[0.5, 0.5, 0.5], jitter_ratio=0.4):
    print(f"Loading mesh from {filename}...")
    mesh = trimesh.load(filename)
    
    # Rotate 90 degrees around X axis (to convert Z-up to Y-up)
    # Rotation matrix for -90 deg around X:
    # [1, 0, 0]
    # [0, 0, -1]
    # [0, 1, 0]
    # But trimesh uses 4x4 matrix
    rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    mesh.apply_transform(rot_matrix)
    
    if scale != 1.0:
        mesh.apply_scale(scale)
    print("Voxelizing mesh...")
    voxel_grid = mesh.voxelized(pitch=dx)
    print("Filling interior...")
    voxel_grid = voxel_grid.fill()
    points = voxel_grid.points.astype(np.float32)
    # 添加随机扰动
    if jitter_ratio > 0:
        jitter = (np.random.rand(*points.shape) - 0.5) * dx * jitter_ratio
        points += jitter
    points += np.array(offset, dtype=np.float32)
    print(f"Generated {len(points)} particles from mesh (jitter_ratio={jitter_ratio}).")
    return points

# --- Simulation Setup ---

def run_simulation(output_dir="workspace/taichi/output_sim", material_type='non_newtonian'):
    # Parameters
    dtype = ti.f32
    # Detect if we are on CPU to adjust parameters automatically
    is_cpu = (ti.cfg.arch != ti.cuda)
    
    dt = 1e-4
    frame_dt = 1/60.0
    
    # Adaptive settings
    if is_cpu:
        dx = 0.02 # Lower resolution for CPU
        particle_chunk_size = 2**12
        cuda_chunk_size = 32
        print(f"Running on CPU: Adjusted dx to {dx} to prevent OOM/Timeouts.")
    else:
        dx = 0.02 # High resolution for GPU (Adjusted to preventing OOM)
        particle_chunk_size = 2**14
        cuda_chunk_size = 64
        
    inv_dx = 1.0 / dx
    
    # --- Material Parameter Helper ---
    def get_material_props(m_type):
        # Returns: material_id, rho, mu, lam, fric, coh, yield, visc
        m_id = MPMSimulator.elasticity
        rho = 1000.0
        E = 1e5
        nu = 0.2
        mu = 0.0
        lam = 0.0
        fric = 0.0
        coh = 0.0
        yld = 0.0
        visc = 0.0
        
        if m_type == 'elasticity':
            m_id = MPMSimulator.elasticity
            rho = 2000.0 # Increased density (was 1000) to make it heavier for better impact
            E = 3e5
            nu = 0.25
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))

        elif m_type == 'plasticine':
            m_id = MPMSimulator.von_mises
            rho = 1000.0
            E = 1e4
            nu = 0.25
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            yld = 1000.0

        elif m_type == 'sand':
            m_id = MPMSimulator.drucker_prager
            rho = 1800.0
            # Adjusted for softer sand: Lower E and friction_angle
            E = 2e5   # Lowered from 1e6 to make sand softer and more splashy
            nu = 0.3
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            friction_angle = 10.0 
            sin_phi = math.sin(math.radians(friction_angle))
            fric = math.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        elif m_type == 'newtonian':
            m_id = MPMSimulator.viscous_fluid
            rho = 1000.0
            kappa = 1e4
            mu = 10.0
            lam = kappa - 2/3 * mu

        elif m_type == 'non_newtonian':
            m_id = MPMSimulator.von_mises
            rho = 1000.0
            kappa = 1e5
            mu = 100.0
            lam = kappa - 2/3 * mu
            yld = 10.0
            visc = 1.0

        elif m_type == 'toothpaste_custom':
            m_id = MPMSimulator.von_mises
            rho = 1000.0
            mu = 10000.0
            kappa = 1000000.0
            lam = kappa - 2/3 * mu
            yld = 3000.0
            visc = 10.0
            
        return m_id, rho, mu, lam, fric, coh, yld, visc

    # Get properties for the main object
    obj_props = get_material_props(material_type)
    print(f"Object Material: {material_type}")
    
    # Get properties for the ground (Sand)
    g_props = get_material_props('sand')
    print(f"Ground Material: Sand (Deep)")
    
    # Initialization Paths
    ply_path = "/root/workspace/taichi_dataset/meshes/toyduck.ply" 
    
    init_pos_obj = None
    obj_faces = None

    if ply_path and os.path.exists(ply_path):
        # Load mesh vertices directly
        # First load without offset to calculate size
        temp_verts, obj_faces = load_mesh_vertices_and_faces(ply_path, scale=1.0, offset=[0.0, 0.0, 0.0])
        
        # Calculate bounds
        min_b = temp_verts.min(axis=0)
        max_b = temp_verts.max(axis=0)
        size = max_b - min_b
        print(f"Original Object Size: {size}")
        
        # Target max size (Container is 0.6 wide, let's keep object under 0.25 to be safe)
        target_size = 0.25
        max_dim = size.max()
        scale_factor = 1.0
        if max_dim > target_size:
            scale_factor = target_size / max_dim
            print(f"Object too large. Auto-scaling by {scale_factor:.4f}")
        
        # Reload with correct scale and offset
        # Offset y to 0.3 (sand surface is 0.0, sand bottom is -0.2)
        # 0.3 is a higher drop height for better splash
        init_pos_obj, obj_faces = load_mesh_vertices_and_faces(ply_path, scale=scale_factor, offset=[0.0, 0.3, 0.0])
        print(f"Object particles (vertices): {len(init_pos_obj)}")
    else:
        # Fallback cube
        print("Ply file not found, using random cube for object")
        # Center the cube at (0, 0.4, 0) to avoid touching walls
        init_pos_obj = np.random.rand(10000, 3).astype(np.float32) * 0.2 + np.array([-0.1, 0.4, -0.1])

    # Generate Ground Particles
    # Box from [-0.3, -0.2, -0.3] to [0.3, 0.0, 0.3] (Shrunk to 1/2)
    print("Generating ground particles...")
    g_min = np.array([-0.3, -0.2, -0.3])
    g_max = np.array([0.3, 0.0, 0.3])
    
    # Target: Higher density (*4), More random
    # Previous: Box 1.2x1.2x0.2 = 0.288 m^3. Count ~290k.
    # New: Box 0.6x0.6x0.2 = 0.072 m^3 (1/4 volume).
    # Maintain ~300k particles in 1/4 volume => 4x Density.
    target_ground_particles = 300000 
    
    # Use Random Uniform Sampling to avoid regular grid patterns
    init_pos_ground = np.random.uniform(low=g_min, high=g_max, size=(target_ground_particles, 3)).astype(np.float32)

    # --- Density Correction ---
    # Since we are over-sampling (packing more particles than the grid resolution 'dx' implies),
    # we must reduce the mass per particle to maintain the correct physical density (rho).
    # Default p_vol is (dx * 0.5)^3.
    # Expected particles for this volume = Volume / p_vol
    sand_volume = (g_max[0] - g_min[0]) * (g_max[1] - g_min[1]) * (g_max[2] - g_min[2])
    p_vol_default = (dx * 0.5) ** 3
    expected_particles = sand_volume / p_vol_default
    density_scale = expected_particles / target_ground_particles
    
    print(f"Sand Density Correction: Scale Rho by {density_scale:.4f} (Count: {target_ground_particles} vs Expected: {int(expected_particles)})")
    
    # Adjust Ground Properties (rho is index 1)
    g_props_list = list(g_props)
    g_props_list[1] *= density_scale # Scale rho
    g_props = tuple(g_props_list)
    
    print(f"Ground particles: {len(init_pos_ground)}")
    
    total_particles = len(init_pos_obj) + len(init_pos_ground)
    print(f"Total particles: {total_particles}")

    # Fields
    num_particles = ti.field(ti.i32, shape=())
    
    # Particle layout - Increased size
    particle = ti.root.dynamic(ti.i, 2**23, particle_chunk_size)
    
    # Simulator
    sim = MPMSimulator(dtype=dtype, dt=dt, frame_dt=frame_dt, particle_layout=particle, 
                       dx=ti.field(dtype, shape=()), inv_dx=ti.field(dtype, shape=()), 
                       n_particles=num_particles, gravity=[0, -9.8, 0], 
                       material=None, cuda_chunk_size=cuda_chunk_size)
    
    sim.dx[None] = dx
    sim.inv_dx[None] = inv_dx
    sim.cfl_satisfy[None] = 1
    sim.p_vol[None] = (dx * 0.5) ** 3 
    
    # Initialize particles kernel
    @ti.kernel
    def add_particles(offset: int, n: int, pos: ti.template(), 
                      m_id: int, p_rho: float, p_mu: float, p_lam: float, 
                      p_fric: float, p_coh: float, p_yld: float, p_visc: float):
        for i in range(n):
            idx = offset + i
            sim.x[idx, 0] = pos[i]
            sim.v[idx, 0] = ti.Vector([0.0, 0.0, 0.0])
            sim.F[idx, 0] = ti.Matrix.identity(dtype, 3)
            sim.C[idx, 0] = ti.Matrix.zero(dtype, 3, 3)
            sim.p_mass[idx] = sim.p_vol[None] * p_rho
            sim.mu[idx] = p_mu
            sim.lam[idx] = p_lam
            sim.material_id[idx] = m_id
            
            sim.friction_alpha[idx] = p_fric
            sim.cohesion[idx] = p_coh
            sim.yield_stress[idx] = p_yld
            sim.plastic_viscosity[idx] = p_visc

    # Add Object
    pos_field_obj = ti.Vector.field(3, dtype=dtype, shape=len(init_pos_obj))
    pos_field_obj.from_numpy(init_pos_obj)
    add_particles(0, len(init_pos_obj), pos_field_obj, *obj_props)
    
    # Add Ground
    pos_field_g = ti.Vector.field(3, dtype=dtype, shape=len(init_pos_ground))
    pos_field_g.from_numpy(init_pos_ground)
    add_particles(len(init_pos_obj), len(init_pos_ground), pos_field_g, *g_props)

    num_particles[None] = total_particles
    
    # Add boundary (Bedrock)
    # The sand is from y=-0.2 to 0.0. So floor should be at -0.2
    sim.add_surface_collider(point=[0, -0.2, 0], normal=[0, 1, 0], surface=MPMSimulator.surface_sticky)

    # Add walls ("Pool" boundary) to contain sand
    # Sand range is x=[-0.3, 0.3], z=[-0.3, 0.3] (Shrunk to 1/2 size)
    wall_padding = 0.0 
    
    # Left Wall (x = -0.3)
    sim.add_surface_collider(point=[-0.3 - wall_padding, 0, 0], normal=[1, 0, 0], surface=MPMSimulator.surface_slip)
    # Right Wall (x = 0.3)
    sim.add_surface_collider(point=[0.3 + wall_padding, 0, 0], normal=[-1, 0, 0], surface=MPMSimulator.surface_slip)
    # Back Wall (z = -0.3)
    sim.add_surface_collider(point=[0, 0, -0.3 - wall_padding], normal=[0, 0, 1], surface=MPMSimulator.surface_slip)
    # Front Wall (z = 0.3)
    sim.add_surface_collider(point=[0, 0, 0.3 + wall_padding], normal=[0, 0, -1], surface=MPMSimulator.surface_slip)

    # Run loop
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Generate Metadata JSON ---
    initial_pos_np = sim.x.to_numpy()[:num_particles[None], 0, :]
    xyz_min = initial_pos_np.min(axis=0).tolist()
    xyz_max = initial_pos_np.max(axis=0).tolist()
    
    simulation_frames = 30
    gravity_vec = [0, -9.8, 0]
    
    bc_data = {
        "ground": [[0.0, -0.2, 0.0], [0.0, 1.0, 0.0], 0],
        "wall_left": [[-0.3, 0.0, 0.0], [1.0, 0.0, 0.0], 0],
        "wall_right": [[0.3, 0.0, 0.0], [-1.0, 0.0, 0.0], 0],
        "wall_back": [[0.0, 0.0, -0.3], [0.0, 0.0, 1.0], 0],
        "wall_front": [[0.0, 0.0, 0.3], [0.0, 0.0, -1.0], 0]
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
    n_obj = len(init_pos_obj)
    for frame in range(simulation_frames): 
        sim.advance(frame)
        
        pos = sim.x.to_numpy()[:num_particles[None], 0, :] 
        
        # Split and save
        pos_obj = pos[:n_obj]
        pos_ground = pos[n_obj:]
        
        # Save Object as PLY mesh if faces exist, else NPY
        if obj_faces is not None:
             mesh = trimesh.Trimesh(vertices=pos_obj, faces=obj_faces)
             mesh.export(os.path.join(output_dir, f"frame_{frame:04d}_obj.ply"))
        else:
             np.save(os.path.join(output_dir, f"frame_{frame:04d}_obj.npy"), pos_obj)

        # Save Ground as NPY (Point Cloud)
        np.save(os.path.join(output_dir, f"frame_{frame:04d}_ground.npy"), pos_ground)

        print(f"Frame {frame} completed. Particles: {num_particles[None]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="workspace/taichi_dataset/particles_output/output_multi_material", help="Output directory for simulation results")
    parser.add_argument('-m', '--material', type=str, default="elasticity", 
                        choices=['elasticity', 'plasticine', 'sand', 'newtonian', 'non_newtonian', 'toothpaste_custom'],
                        help="Material type for simulation")
    args = parser.parse_args()
    run_simulation(args.output_dir, args.material)
