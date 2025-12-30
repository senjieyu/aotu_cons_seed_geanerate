# -*- coding: utf-8 -*-
import os
import numpy as np
from ase.build import fcc111, stack
from ase.io.vasp import write_vasp
from ase.atoms import Atoms

# ==============================================================================
# Part 1: 核心工具函数
# ==============================================================================

def get_d111(a):
    """根据晶格常数计算 FCC(111) 面间距"""
    return a / np.sqrt(3)

def fix_pbc_vertical_layers(atoms, d_111):
    """
    【核心修复】强制修正 Z 轴周期性，防止边界重叠。
    """
    # 1. 获取层数信息
    z_coords = atoms.positions[:, 2]
    layers_unique = np.unique(np.round(z_coords, decimals=3))
    n_layers = len(layers_unique)
    
    # 2. 强制垂直盒子
    total_height = n_layers * d_111
    cell = atoms.get_cell()
    new_cell = np.array([cell[0], cell[1], [0, 0, total_height]])
    atoms.set_cell(new_cell, scale_atoms=False)
    
    # 3. 均匀化 Z 坐标 (消除数值误差)
    indices = np.argsort(z_coords)
    atoms_per_layer = len(atoms) // n_layers
    pos = atoms.positions
    for i in range(n_layers):
        layer_indices = indices[i*atoms_per_layer : (i+1)*atoms_per_layer]
        pos[layer_indices, 2] = i * d_111
    atoms.set_positions(pos)
    
    # 4. 自动修剪
    if n_layers % 3 != 0:
        print(f"  [Auto-Fix] Layer count {n_layers} not divisible by 3. Trimming top layer...")
        del atoms[indices[-atoms_per_layer:]]
        new_height = (n_layers - 1) * d_111
        cell = atoms.get_cell()
        cell[2, 2] = new_height
        atoms.set_cell(cell, scale_atoms=False)
        print(f"  [Auto-Fix] New height: {new_height:.3f}, Layers: {n_layers-1}")
    
    return atoms

def build_base_bulk(element, a, size):
    """构建基础体相 Slab (无真空)"""
    d111 = get_d111(a)
    atoms = fcc111(symbol=element, size=size, a=a, vacuum=0.0, orthogonal=False)
    atoms.pbc = [True, True, True]
    atoms = fix_pbc_vertical_layers(atoms, d111)
    return atoms

def get_inplane_vectors(element, a):
    """获取面内滑移矢量 a1, a2"""
    tmp = build_base_bulk(element, a, size=(1,1,3))
    cell = tmp.get_cell()
    return np.array(cell[0]), np.array(cell[1])

def split_upper_lower(atoms):
    """Z轴分层：返回 upper 部分的 mask"""
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    half = len(order) // 2
    # half 之后的是上半部分
    upper_mask = np.zeros(len(z), dtype=bool)
    upper_mask[order[half:]] = True
    return upper_mask  # 【修正】只返回一个值

def add_perturbation(atoms, amp, seed):
    """添加随机微扰"""
    if amp <= 0: return atoms
    rng = np.random.default_rng(seed)
    pert = atoms.copy()
    disp = (rng.random((len(pert), 3)) - 0.5) * 2.0 * amp
    pert.positions += disp
    return pert

def write_structure_file(atoms, name, output_dir, relax_dir, element, need_relax_sd=False):
    """统一写文件函数"""
    path = os.path.join(output_dir, f"{name}.vasp")
    # 【修正】direct=True 会使用分数坐标。ASE write_vasp 默认可能会 wrap。
    # 如果我们要完全控制，最好手动写，或者信任 ASE。
    # ASE 的 write_vasp 默认 behavior 比较复杂。
    # 但我们这里更关注下面的手动写入部分 (relax_dir)。
    write_vasp(path, atoms, direct=True, sort=True, vasp5=True)
    
    if need_relax_sd:
        sd = np.zeros((len(atoms), 3), dtype=bool)
        if "twin" in name:
            sd[:] = True
        else:
            upper_mask = split_upper_lower(atoms)
            sd[upper_mask, 2] = True
            
        path_relax = os.path.join(relax_dir, f"{name}.vasp")
        with open(path_relax, "w") as f:
            f.write(f"{element}\n1.0\n")
            cell = atoms.get_cell()
            for v in cell: f.write(f"  {v[0]:16.12f}  {v[1]:16.12f}  {v[2]:16.12f}\n")
            f.write(f" {element}\n {len(atoms)}\nSelective dynamics\nDirect\n")
            # 【修正】wrap=False，保留原子原始位置，防止视觉上的跳变
            scaled = atoms.get_scaled_positions(wrap=False)
            for i in range(len(atoms)):
                x,y,z = scaled[i]
                flags = ["T" if sd[i,j] else "F" for j in range(3)]
                f.write(f"  {x:16.12f}  {y:16.12f}  {z:16.12f}   {'  '.join(flags)}\n")

# ==============================================================================
# Part 2: 六大功能模块函数 (修正了调用方式)
# ==============================================================================

def generate_1d_main(base_atoms, a1, n_points, n_perturb, amp, dirs, elm):
    """1. 生成 1D 主滑移路径 (沿 a1)"""
    print(f"--- Generating 1D Main (Points: {n_points}) ---")
    # 【修正】这里不再解包，直接接收一个返回值
    upper = split_upper_lower(base_atoms)
    
    for i, frac in enumerate(np.linspace(0, 1.0, n_points)):
        atoms = base_atoms.copy()
        atoms.positions[upper] += frac * a1
        
        name = f"1D_main_{i:03d}_frac{frac:.4f}"
        # 【修正】基础结构写入 relax 目录时，开启 SD 标志 (FFT)，且不加微扰
        write_structure_file(atoms, name, dirs['out'], dirs['relax'], elm, need_relax_sd=True)
        
        for k in range(n_perturb):
            pert = add_perturbation(atoms, amp, seed=i*100+k)
            # 微扰结构通常只放在 out 目录用于 MD，或者 relax 目录作为备选
            # 这里保持原样，或者根据需求调整。用户只强调了“relax中的文件”不要变（指不要微扰）
            write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], elm, need_relax_sd=False)

def generate_1d_sec(base_atoms, a2, n_points, n_perturb, amp, dirs, elm):
    """2. 生成 1D 次滑移路径 (沿 a2)"""
    print(f"--- Generating 1D Secondary (Points: {n_points}) ---")
    upper = split_upper_lower(base_atoms) # 【修正】
    
    for i, frac in enumerate(np.linspace(0, 1.0, n_points)):
        atoms = base_atoms.copy()
        atoms.positions[upper] += frac * a2
        
        name = f"1D_sec_{i:03d}_frac{frac:.4f}"
        # 【修正】开启 SD
        write_structure_file(atoms, name, dirs['out'], dirs['relax'], elm, need_relax_sd=True)
        
        for k in range(n_perturb):
            pert = add_perturbation(atoms, amp, seed=1000+i*100+k)
            write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], elm, need_relax_sd=False)

def generate_2d_surface(base_atoms, a1, a2, grid_n, n_perturb, amp, dirs, elm):
    """3. 生成 2D Gamma Surface"""
    print(f"--- Generating 2D Surface (Grid: {grid_n}x{grid_n}) ---")
    upper = split_upper_lower(base_atoms) # 【修正】
    fracs = np.linspace(0, 1.0, grid_n)
    
    for ix, fx in enumerate(fracs):
        for iy, fy in enumerate(fracs):
            atoms = base_atoms.copy()
            atoms.positions[upper] += fx * a1 + fy * a2
            
            name = f"2D_{ix:02d}_{iy:02d}"
            # 【修正】开启 SD
            write_structure_file(atoms, name, dirs['out'], dirs['relax'], elm, need_relax_sd=True)
            
            for k in range(n_perturb):
                pert = add_perturbation(atoms, amp, seed=2000 + ix*100 + iy + k)
                write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], elm, need_relax_sd=False)

def generate_key_points(base_atoms, a1, fracs, n_perturb, amp, dirs, elm):
    """4. 生成关键点 (带弛豫推荐)"""
    print(f"--- Generating Key Points (Count: {len(fracs)}) ---")
    upper = split_upper_lower(base_atoms) # 【修正】
    
    for i, frac in enumerate(fracs):
        atoms = base_atoms.copy()
        atoms.positions[upper] += frac * a1
        
        name = f"normal_relax_key_{i:02d}_frac{frac:.3f}"
        write_structure_file(atoms, name, dirs['out'], dirs['relax'], elm, need_relax_sd=True)
        
        for k in range(n_perturb):
            pert = add_perturbation(atoms, amp, seed=3000 + i*100 + k)
            write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], elm)

def generate_strain(base_atoms, a1, strains, fracs, n_perturb, amp, dirs, elm):
    """5. 生成应变结构 (体积变化 + 滑移)"""
    print(f"--- Generating Strain Structures (Strains: {strains}) ---")
    upper = split_upper_lower(base_atoms) # 【修正】
    cell0 = base_atoms.get_cell()
    
    for eps in strains:
        scale = 1.0 + eps
        
        for frac in fracs:
            atoms = base_atoms.copy()
            atoms.positions[upper] += frac * a1
            atoms.positions *= scale
            atoms.set_cell(cell0 * scale, scale_atoms=False)
            
            name = f"strain_eps{eps:.3f}_frac{frac:.3f}"
            write_structure_file(atoms, name, dirs['out'], dirs['relax'], elm, need_relax_sd=True)
            
            for k in range(n_perturb):
                pert = add_perturbation(atoms, amp, seed=4000 + int((eps+0.5)*1000) + k)
                write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], elm)

def generate_twin(element, a, size, n_perturb, amp, dirs):
    """6. 生成孪晶结构 (Twin)"""
    print(f"--- Generating Twin Structure ---")
    d111 = get_d111(a)
    nx, ny, nlayers = size
    half_size = (nx, ny, max(6, nlayers//2))
    
    slab = fcc111(symbol=element, size=half_size, a=a, vacuum=0.0, orthogonal=False)
    
    twin = slab.copy()
    pos = twin.positions 
    z_mid = (pos[:,2].max() + pos[:,2].min()) / 2 
    pos[:, 2] = 2*z_mid - pos[:, 2] 
    twin.set_positions(pos) 
    
    atoms = stack(slab, twin, axis=2, distance=d111) 
    atoms.pbc = [True, True, True] 
    
    # 自动修复边界 
    atoms = fix_pbc_vertical_layers(atoms, d111) 
    
    name = "twin_CTB_base" 
    write_structure_file(atoms, name, dirs['out'], dirs['relax'], element, need_relax_sd=True) 
    
    for k in range(n_perturb): 
        pert = add_perturbation(atoms, amp, seed=5000+k) 
        write_structure_file(pert, f"{name}_pert{k:02d}", dirs['out'], dirs['relax'], element) 

# ==============================================================================
# Part 3: Main 配置与执行区
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. 全局配置 ---
    config = {
        "element": "Cu",
        "lattice_a": 3.615,
        "size_111": (3, 3, 12),
        "perturb_amp": 0.0, # 【修改】禁用微扰
        "output_root": r"C:/Users/senji/Desktop/AOTU_cons\test_file",
    }

    # --- 2. 各个功能的特定参数 ---
    
    task_1d_main = {
        "enable": True,
        "n_points": 41,
        "n_perturb": 0, # 【修改】禁用微扰
    }
    
    task_1d_sec = {
        "enable": True,
        "n_points": 21,
        "n_perturb": 0, # 【修改】禁用微扰
    }
    
    task_2d = {
        "enable": True,
        "grid_n": 8,
        "n_perturb": 0, # 【修改】禁用微扰
    }
    
    task_key_points = {
        "enable": True,
        "fractions": [0.0, 1/6, 1/3, 1/2, 2/3, 1.0],
        "n_perturb": 0, # 【修改】禁用微扰
    }
    
    task_strain = {
        "enable": True,
        "strains": [-0.02, -0.01, 0.01, 0.02],
        "key_fracs": [0.0, 1/3, 2/3],
        "n_perturb": 0, # 【修改】禁用微扰
    }
    
    task_twin = {
        "enable": True,
        "n_perturb": 0, # 【修改】禁用微扰
    }

    # ========================== 执行逻辑 ==========================
    
    dirs = {
        'out': config['output_root'],
        'relax': os.path.join(config['output_root'], "to_relax")
    }
    os.makedirs(dirs['out'], exist_ok=True)
    os.makedirs(dirs['relax'], exist_ok=True)
    
    print(f"Start generating structures for {config['element']}...")
    print(f"Output Directory: {config['output_root']}")

    base_atoms = build_base_bulk(config['element'], config['lattice_a'], config['size_111'])
    a1, a2 = get_inplane_vectors(config['element'], config['lattice_a'])
    
    if task_1d_main['enable']:
        generate_1d_main(base_atoms, a1, task_1d_main['n_points'], task_1d_main['n_perturb'], 
                         config['perturb_amp'], dirs, config['element'])

    if task_1d_sec['enable']:
        generate_1d_sec(base_atoms, a2, task_1d_sec['n_points'], task_1d_sec['n_perturb'], 
                        config['perturb_amp'], dirs, config['element'])

    if task_2d['enable']:
        generate_2d_surface(base_atoms, a1, a2, task_2d['grid_n'], task_2d['n_perturb'], 
                            config['perturb_amp'], dirs, config['element'])

    if task_key_points['enable']:
        generate_key_points(base_atoms, a1, task_key_points['fractions'], task_key_points['n_perturb'], 
                            config['perturb_amp'], dirs, config['element'])

    if task_strain['enable']:
        generate_strain(base_atoms, a1, task_strain['strains'], task_strain['key_fracs'], 
                        task_strain['n_perturb'], config['perturb_amp'], dirs, config['element'])

    if task_twin['enable']:
        generate_twin(config['element'], config['lattice_a'], config['size_111'], 
                      task_twin['n_perturb'], config['perturb_amp'], dirs)

    print("All tasks finished successfully.")
