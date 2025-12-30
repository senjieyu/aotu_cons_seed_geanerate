# -*- coding: utf-8 -*-
# model_construct.py
import os
import numpy as np

from ase import Atoms
from ase.build import fcc111, bcc110, hcp0001
from ase.io.vasp import write_vasp
from ase.neighborlist import NeighborList


class AtomUtils:
    """原子操作通用工具集"""

    @staticmethod
    def fix_pbc_vertical(atoms, vacuum=0.0):
        """修复垂直方向周期性，基于当前原子高度重设 Cell"""
        z_coords = atoms.positions[:, 2]
        if len(z_coords) == 0: return atoms

        unique_layers = np.unique(np.round(z_coords, 3))
        n_layers = len(unique_layers)

        if n_layers > 1:
            d_spacing = np.mean(np.diff(unique_layers))
        else:
            d_spacing = 2.0

        total_height = max(n_layers, 1) * d_spacing + vacuum
        cell = atoms.get_cell()
        new_cell = np.array([cell[0], cell[1], [0, 0, total_height]])
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.pbc = [True, True, True]
        return atoms

    @staticmethod
    def split_upper_lower_mask(atoms):
        """
        Z轴分层：返回 upper 部分的 mask
        【修正】完全仿照用户提供的 split_upper_lower
        """
        z = atoms.positions[:, 2]
        order = np.argsort(z)
        half = len(order) // 2
        # half 之后的是上半部分
        upper_mask = np.zeros(len(z), dtype=bool)
        upper_mask[order[half:]] = True
        return upper_mask

    @staticmethod
    def fix_pbc_vertical_layers(atoms, d_111):
        """
        【核心修复】强制修正 Z 轴周期性，防止边界重叠。
        仿照用户提供的 fix_pbc_vertical_layers
        """
        # 1. 获取层数信息
        z_coords = atoms.positions[:, 2]
        # round to 3 decimals to group layers
        layers_unique = np.unique(np.round(z_coords, decimals=3))
        n_layers = len(layers_unique)
        
        # 2. 强制垂直盒子
        total_height = n_layers * d_111
        cell = atoms.get_cell()
        # Keep original xy, reset z vector to strictly vertical
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
            # Remove top layer atoms
            # indices are sorted by Z. Top layer is the last block.
            to_delete = indices[-atoms_per_layer:]
            del atoms[to_delete]
            
            new_height = (n_layers - 1) * d_111
            cell = atoms.get_cell()
            cell[2, 2] = new_height
            atoms.set_cell(cell, scale_atoms=False)
            print(f"  [Auto-Fix] New height: {new_height:.3f}, Layers: {n_layers-1}")
            
        return atoms

    # Keep fix_pbc_vertical_robust as an alias or remove it
    fix_pbc_vertical_robust = fix_pbc_vertical_layers

    @staticmethod
    def add_perturbation(atoms, amp, seed):
        if amp <= 0: return atoms
        rng = np.random.default_rng(seed)
        pert = atoms.copy()
        disp = (rng.random((len(pert), 3)) - 0.5) * 2.0 * amp
        pert.positions += disp
        return pert

    @staticmethod
    def remove_overlaps(atoms, cutoff=0.8):
        n = len(atoms)
        if n == 0: return atoms
        nl = NeighborList([cutoff] * n, bothways=True, self_interaction=False)
        nl.update(atoms)
        to_delete = set()
        for i in range(n):
            if i in to_delete: continue
            idxs, offsets = nl.get_neighbors(i)
            for j, off in zip(idxs, offsets):
                if j <= i or j in to_delete: continue
                rij = atoms.positions[j] + np.dot(off, atoms.cell) - atoms.positions[i]
                if np.linalg.norm(rij) < cutoff: to_delete.add(j)
        if to_delete: del atoms[sorted(to_delete)]
        return atoms

    @staticmethod
    def write_vasp_file(atoms, filename, output_dir, relax_dir=None, element="X", need_sd=False, sd_mode="slab_xyz"):
        """
        sd_mode:
          - 'slab_xyz': Fix top/bottom layers (FFF), relax middle (TTT). Good for stable surfaces/twins.
          - 'gsfe_z': Fix X/Y for ALL atoms (FFT or FFF), only relax Z. Good for Energy Curves.
        """
        full_path = os.path.join(output_dir, f"{filename}.vasp")
        write_vasp(full_path, atoms, direct=True, sort=True, vasp5=True)

        if need_sd and relax_dir:
            sd_path = os.path.join(relax_dir, f"{filename}.vasp")
            sd_flags = np.ones((len(atoms), 3), dtype=bool)
            
            if sd_mode == "gsfe_z":
                # Strategy: Match GSFE script logic
                # Fix X and Y for ALL atoms. 
                # Relax Z for UPPER atoms only.
                upper_mask = AtomUtils.split_upper_lower_mask(atoms)
                sd_flags[:] = False # Reset to F F F
                sd_flags[upper_mask, 2] = True # Set upper Z to T
                
            else: # 'slab_xyz' (Default)
                # T T T
                sd_flags[:] = True
                
                # If we want to fix boundaries (like for surface slab), we can do it.
                # But user asked for "Full Relax TTT".
                # Usually for slab we fix center layers to mimic bulk, relax surface.
                # Or fix bottom, relax top.
                # But here for "Twin Full Relax", let's relax everything (except maybe cell shape ISIF=2/3).
                # But user specifically said "TTT".
                # Let's assume TTT for all atoms.
                pass

            with open(sd_path, "w") as f:
                f.write("Generated generic\n1.0\n")
                cell = atoms.get_cell()
                for v in cell: f.write(f"  {v[0]:12.8f}  {v[1]:12.8f}  {v[2]:12.8f}\n")
                f.write(f" {element}\n {len(atoms)}\nSelective dynamics\nDirect\n")
                scaled = atoms.get_scaled_positions(wrap=True)
                for i in range(len(atoms)):
                    s = scaled[i]
                    flg = " ".join(["T" if x else "F" for x in sd_flags[i]])
                    f.write(f"  {s[0]:12.8f}  {s[1]:12.8f}  {s[2]:12.8f} {flg}\n")


class StructureFactory:

    @staticmethod
    def _get_vectors_by_symmetry(atoms, lattice_type, size):
        cell = atoms.get_cell()
        a1_prim = np.array(cell[0]) / size[0]
        a2_prim = np.array(cell[1]) / size[1]
        info = {}
        if lattice_type == "FCC":
            v_main, info['main_label'] = a1_prim, "[1-10]"
            v_sec, info['sec_label'] = a2_prim, "[11-2]"
        elif lattice_type == "BCC":
            v_main, info['main_label'] = a1_prim, "[1-11]"
            v_sec, info['sec_label'] = a2_prim, "[001]"
        elif lattice_type == "HCP":
            v_main, info['main_label'] = a1_prim, "<11-20>"
            v_sec, info['sec_label'] = a2_prim, "<10-10>"
        else:
            v_main, v_sec = a1_prim, a2_prim
        return v_main, v_sec, info

    @staticmethod
    def generate_fcc_stacking_sequence(layer_counts_list):
        """生成 ABC 序列索引 (0, 1, 2)"""
        current_idx = 0
        direction = 1  # 1=ABC, -1=CBA
        sequence = []
        for n_layers in layer_counts_list:
            if n_layers <= 0: continue
            for _ in range(n_layers):
                sequence.append(current_idx)
                current_idx = (current_idx + direction) % 3
            # Switch direction for twin boundary
            direction *= -1
            # Adjust current_idx to maintain continuity at the boundary
            # The last placed atom was at `last_idx`.
            # The next atom should be a reflection.
            # E.g. A B C (current is A) -> Direction switch -> B A C
            # If sequence ends at C (idx 2), direction 1. Next would be A (0).
            # Switch to -1. Next should be B (1).
            # Logic:
            # We just placed `n_layers`. The last one was `sequence[-1]`.
            # The loop updated `current_idx` to the *next* potential spot in current direction.
            # But we want to reflect.
            # Let's trace:
            # Seq: 0(A), 1(B), 2(C). current_idx becomes 0. direction 1.
            # End of region. direction becomes -1.
            # We want next to be 1(B).
            # current_idx (0) - 1 (old dir) = 2 (C) -> Last placed.
            # Last placed (2) + new dir (-1) = 1 (B). Correct.
            
            # Re-calculate correct start for next segment:
            last_placed = (current_idx - (-direction)) % 3  # undo the last advance with OLD direction
            current_idx = (last_placed + direction) % 3     # advance with NEW direction
            
        return sequence

    @staticmethod
    def apply_slip_1d(atoms, vec, frac, mode='main'):
        """
        Task 1/2: Apply 1D Slip
        mode='main': vec is full slip vector.
        mode='sec': vec is partial slip vector.
        """
        # Determine slip vector
        slip_vec = vec * frac
        
        # Split upper/lower
        upper_mask = AtomUtils.split_upper_lower_mask(atoms)
        
        # Apply shift
        atoms.positions[upper_mask] += slip_vec
        
        # Wrap
        atoms.wrap()
        
        return atoms

    @staticmethod
    def apply_slip_2d(atoms, v1, v2, fx, fy):
        """
        Task 3: Apply 2D Slip (Gamma Surface)
        shift = fx * v1 + fy * v2
        """
        slip_vec = fx * v1 + fy * v2
        
        upper_mask = AtomUtils.split_upper_lower_mask(atoms)
        atoms.positions[upper_mask] += slip_vec
        atoms.wrap()
        
        return atoms

    @staticmethod
    def build_fcc_by_stacking_sequence(element, a, size_xy, sequence, vacuum=0.0):
        """根据序列构建完美FCC"""
        # Base layer (111)
        layer_base = fcc111(symbol=element, size=(size_xy[0], size_xy[1], 1), a=a, vacuum=0.0, orthogonal=False)
        cell = layer_base.get_cell()
        
        # In-plane vectors
        v1 = cell[0] / size_xy[0]
        v2 = cell[1] / size_xy[1]
        
        # Inter-layer spacing and shift
        d111 = float(a) / np.sqrt(3.0)
        # Shift vector for stacking (A->B->C) is 1/3(v1+v2) usually, or along diagonal.
        # For orthogonal=False fcc111, cell[0]=(a,0,0) (no), it's hexagonal.
        # Check ASE fcc111 documentation or behavior.
        # Usually fcc111 returns a hexagonal cell.
        # Shift A->B is (2/3, 1/3) or (1/3, 1/3) depending on basis.
        # Let's rely on geometric relation: 
        # A at (0,0), B at (r + s)/3 ? No.
        # Let's use the standard shift vector for hexagonal close packed layers.
        # Vector A->B.
        # For standard ASE fcc111:
        # P0 = (0,0,0). 
        # P1 (Layer B) should be at specific shift.
        # The shift vector `shift_vec` is such that 3*shift_vec ~ lattice vector.
        shift_vec = (v1 + v2) / 3.0 # This is a good approximation for standard hex cell.

        positions = []
        numbers = []
        ref_pos = layer_base.get_positions()
        ref_nums = layer_base.get_atomic_numbers()

        for z_idx, stack_idx in enumerate(sequence):
            z_height = z_idx * d111
            # stack_idx is 0, 1, 2. 
            # 0->0 shift, 1->1 shift, 2->2 shift.
            xy_shift = shift_vec * stack_idx
            
            current_layer_pos = ref_pos + xy_shift
            current_layer_pos[:, 2] = z_height
            
            positions.append(current_layer_pos)
            numbers.append(ref_nums)

        all_pos = np.vstack(positions)
        all_nums = np.concatenate(numbers)
        total_height = len(sequence) * d111 + vacuum
        new_cell = np.array([cell[0], cell[1], [0, 0, total_height]])

        atoms = Atoms(positions=all_pos, numbers=all_nums, cell=new_cell, pbc=[True, True, True])
        atoms.wrap()
        return atoms
    @staticmethod
    def build_fcc(element, a, size, vacuum=0.0):
        # 【修正】用户确认脚本不是矩形，所以使用 orthogonal=False (菱形/六方原胞)
        # 这与用户的 GSFE 脚本一致。
        atoms = fcc111(symbol=element, size=size, a=a, vacuum=0.0, orthogonal=False)
        
        # Calculate d111
        d111 = float(a) / np.sqrt(3.0)
        # Use user's fix logic
        atoms = AtomUtils.fix_pbc_vertical_layers(atoms, d111)
        
        # Add vacuum if needed
        if vacuum > 0.0:
            c = atoms.get_cell()
            c[2, 2] += vacuum
            atoms.set_cell(c, scale_atoms=False)
            atoms.pbc = [True, True, True]
            
        atoms.info.update({'struct_type': 'FCC', 'element': element, 'lattice_a': float(a), 'size': tuple(size),
                           'vacuum': float(vacuum)})
        v1, v2, info = StructureFactory._get_vectors_by_symmetry(atoms, "FCC", size)
        return atoms, v1, v2, info

    @staticmethod
    def build_bcc(element, a, size, vacuum=0.0):
        atoms = bcc110(symbol=element, size=size, a=a, vacuum=0.0, orthogonal=False)
        atoms = AtomUtils.fix_pbc_vertical(atoms, vacuum=vacuum)
        atoms.info.update({'struct_type': 'BCC', 'element': element, 'lattice_a': float(a), 'size': tuple(size),
                           'vacuum': float(vacuum)})
        v1, v2, info = StructureFactory._get_vectors_by_symmetry(atoms, "BCC", size)
        return atoms, v1, v2, info

    @staticmethod
    def build_hcp(element, a, c, size, vacuum=0.0):
        atoms = hcp0001(symbol=element, size=size, a=a, c=c, vacuum=0.0, orthogonal=False)
        atoms = AtomUtils.fix_pbc_vertical(atoms, vacuum=vacuum)
        atoms.info.update({'struct_type': 'HCP', 'element': element, 'lattice_a': float(a), 'lattice_c': float(c),
                           'size': tuple(size), 'vacuum': float(vacuum)})
        v1, v2, info = StructureFactory._get_vectors_by_symmetry(atoms, "HCP", size)
        return atoms, v1, v2, info

    @staticmethod
    def build_intersecting_twin(base_atoms, plane_miller, width_A, center_frac=0.5):
        """
        构建交叉/多级孪晶
        :param base_atoms: 基础原子结构
        :param plane_miller: 孪晶面的米勒指数 (h, k, l)，例如 (1, -1, 1)
        :param width_A: 孪晶板条的宽度 (Angstrom)
        :param center_frac: 孪晶中心在盒子中的位置 (0.0-1.0)
        """
        # 1. 确定晶体取向矩阵
        # 假设 base_atoms 是标准 fcc111 构建的，其 Z 轴 // [111]
        # X 轴 // [1-10], Y 轴 // [11-2] (六角坐标系转换来的)
        # 我们需要将 plane_miller (在晶体坐标系下) 转换为 笛卡尔坐标系下的法向量 n
        
        # 标准 FCC 晶格矢量 (笛卡尔系)
        # a1 = a/2 [0, 1, 1], a2 = a/2 [1, 0, 1], a3 = a/2 [1, 1, 0] (这是原胞)
        # 但我们用的是 fcc111 切出来的超胞。
        # 简单起见，利用 ASE 的 Miller 指数工具或者几何关系。
        # 对于 FCC，(hkl) 面的法向在倒空间是 [hkl]。在实空间笛卡尔系下，如果是立方晶系，方向 [hkl] 垂直于 (hkl) 面。
        # 关键是找到当前 atoms 的坐标系与晶体立方轴 [100], [010], [001] 的关系。
        
        # 假设输入的 base_atoms 是由 build_fcc 生成的，Z//[111]。
        # 我们可以试探性地定义常见的交叉孪晶面。
        # 主孪晶面是 (111) (Z平面)。
        # 常见的交叉孪晶面是 (11-1), (1-11), (-111)。
        # 它们与 (111) 的夹角都是 ~70.5 度。
        
        # 让我们计算 (1 -1 1) 在当前坐标系下的法向量。
        # 已知：Z_axis (0,0,1) 对应 [1 1 1] (归一化后)。
        # 我们需要一个旋转矩阵 R，使得 R * [1 1 1] = [0 0 1]。
        # 实际上，我们可以直接构造法向量 n。
        # 两个向量 [1 1 1] 和 [1 -1 1] 的夹角余弦 = 1/3。
        # 在当前坐标系中，Z 是 [1 1 1]。我们需要一个向量 n，使得 n . Z = cos(70.53) = 1/3。
        # 且 n 应该在 X-Z 或 Y-Z 平面内，或者任意旋转。
        # 我们可以简单地定义 n = (sin(theta), 0, cos(theta))，其中 cos(theta)=1/3。
        # sin(theta) = sqrt(1 - 1/9) = sqrt(8)/3 = 2sqrt(2)/3。
        
        # 定义法向量 n
        cost = 1.0/3.0
        sint = np.sqrt(8.0)/3.0
        # 任意选一个方位角 phi，比如 0 (在 XZ 平面)
        # n = [sint, 0, cost]
        # 注意：这只是几何上的一个满足夹角的方向。对应具体的 (1-11) 需要看 X/Y 的具体取向。
        # 但对于构建“一般的交叉孪晶模型”，只要夹角对，物理上就是等价的对称面。
        normal = np.array([sint, 0.0, cost])
        
        # 2. 定义板条区域
        # 中心点 C
        cell = base_atoms.get_cell()
        center = np.dot(center_frac, np.sum(cell, axis=0)) # Body center if 0.5
        # 或者简单的盒子中心
        center = np.sum(cell, axis=0) * 0.5
        
        # 筛选原子
        # 距离 d = (r - center) . n
        # 条件：-w/2 < d < w/2
        pos = base_atoms.get_positions()
        # 考虑 PBC：需要最小镜像距离。
        # 简化：假设盒子足够大，且我们只处理中心附近的板条，不跨越边界。
        # 或者直接计算距离。
        
        rel_pos = pos - center
        dists = np.dot(rel_pos, normal)
        
        mask = np.abs(dists) < (width_A / 2.0)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            print("Warning: No atoms found in the twin band region.")
            return base_atoms

    @staticmethod
    def build_wedge_twin(base_atoms, angle_deg=70.53):
        """
        构建楔形/V形孪晶 (Wedge Twin)
        策略：定义两个相交平面，内部填充孪晶点阵，外部填充基体点阵。
        并执行边界修复。
        """
        # 1. 定义几何
        # 我们希望 V 形开口向上或向某侧。
        # 假设 V 形由两个 {111} 面组成。
        # 面 1: (11-1) -> 法线 n1 (倾斜)
        # 面 2: (-111) -> 法线 n2 (倾斜)
        # 或者简单点：
        # 面 1: Z 轴 (111) -> n1 = [0,0,1]
        # 面 2: 倾斜 70.5 度 -> n2 = [0, sin, cos]
        # 这样 V 形夹角是 (180 - 70.5) 或 70.5。
        # 图片显示的是一个 V 形切口。我们假设夹角是 70.5 度（两个 111 面的夹角）。
        
        # 定义两个法向量，关于 Z 轴对称分布，这样 V 形正对上方。
        # 半角 = 70.53 / 2 = 35.26 度
        # n1 = [-sin(35), 0, cos(35)]
        # n2 = [ sin(35), 0, cos(35)]
        # 这种几何在 FCC 中对应 (-1 1 1) 和 (1 1 1) ? 
        # 在 fcc111 (Z//111) 体系中，其他 {111} 面法线与 Z 的夹角都是 70.5 度。
        # 所以我们应该用：
        # n1: 倾斜 70.5 度 (右侧)
        # n2: 倾斜 70.5 度 (左侧, 或者是另一个方向)
        # 实际上，V 形孪晶通常是由两个共轭孪晶面围成的。
        
        # 简化模型：
        # 定义 V 形区域： |y| < z * tan(alpha) ?
        # 让我们用两个平面：
        # Plane 1: n1 . (r - center) < 0
        # Plane 2: n2 . (r - center) < 0
        # Intersection is the Wedge.
        
        # 角度设置：
        theta = np.radians(70.53 / 2.0)
        # 让我们在 YZ 平面构建 V 形
        n1 = np.array([0.0,  np.cos(theta), np.sin(theta)]) # 指向右上方? No, dot product.
        # 线方程: z = |y| * cot(theta).
        # Wedge region: z > |y| * cot(theta) (开口向上)
        # 或者 z < ...
        
        # 为了匹配图片中的 V 形（开口向下，像个屋顶，或者开口向上像个谷）：
        # 图片里的红色箭头指的是 V 形边界。
        # 我们构建一个开口向上的 V 形孪晶区。
        
        # 2. 准备晶格
        cell = base_atoms.get_cell()
        center = np.sum(cell, axis=0) * 0.5
        pos_orig = base_atoms.get_positions()
        
        # 寻找最近原子作为 Pivot (V 形顶点)
        dists = np.linalg.norm(pos_orig - center, axis=1)
        pivot = pos_orig[np.argmin(dists)]
        
        # 旋转矩阵 (绕 X 轴旋转 180 度? 或者绕 <111> 旋转 60 度?)
        # 孪晶取向应该是基体绕某个 <111> 轴旋转 60 度。
        # 但是 V 形区域内的晶格取向必须与两个边界都形成孪晶关系吗？
        # 几何上，如果只有一种孪晶取向，它只能与一个面形成共格孪晶界。
        # 与另一个面通常形成非共格晶界 (Incoherent TB)。
        # 除非是五重孪晶等多晶结构。
        # 假设这里是单孪晶取向：
        # 基体 (Matrix) 和 孪晶 (Twin)。
        # 界面 1 是共格孪晶界 (CTB)。
        # 界面 2 是非共格孪晶界 (ITB) 或普通晶界。
        
        # 我们选择绕 n1 (假设 n1 是 (111) 面法线) 旋转 60 度。
        # 这样 n1 界面是完美的。n2 界面是复杂的。
        
        # 修正：在 FCC111 体系中，Z 是 [111]。
        # 我们让 n1 = [0,0,1] (Z轴)。
        # 这样水平界面是完美的。
        # V 形的一个边是水平的？图片里是倒 V 形 (Chevron)。两个边都是斜的。
        # 这意味着两个边都是倾斜的 {111} 面。
        # 这种结构通常叫 "Stacking Fault Tetrahedron" 的一部分，或者是 Lomer-Cottrell 锁。
        # 无论如何，我们构建一个几何 V 形，内部填充旋转 60 度的晶格。
        # 旋转轴：选择其中一个面的法线，或者 Z 轴。
        # 如果选 Z 轴旋转，那么倾斜面都是非共格的。
        # 让我们选 Z 轴旋转 (标准孪晶取向)，然后切一个 V 形。
        
        n_rot = np.array([0.0, 0.0, 1.0])
        th = np.radians(60.0)
        c_ = np.cos(th); s_ = np.sin(th)
        R = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]) # 绕 Z 轴简单的旋转矩阵
        # Wait, build_fcc Z is [111], X/Y are in plane.
        # 绕 Z 旋转 60 度就是绕 [111] 旋转。正确。
        
        # 生成 Grid A (Base) 和 Grid B (Twin)
        pos_A = pos_orig
        pos_B = np.dot(pos_A - pivot, R.T) + pivot
        
        # 3. 定义 V 形区域掩码
        # 开口向上的 V 形: z > |y - cy| * slope + cz
        # Slope 对应 70.5 度夹角。
        # tan(90 - 35.26) = cot(35.26) ~ sqrt(2) = 1.414
        slope = np.sqrt(2.0)
        
        # 坐标相对于 Pivot
        v = pos_A - pivot # 使用 Grid A 的坐标来定义空间区域
        # 区域：V 形内部
        # 假设 V 形沿 X 轴延伸 (柱状)，在 YZ 平面看是 V 形。
        mask_wedge = (v[:, 2] > np.abs(v[:, 1]) * slope) 
        
        # 4. 填充
        final_pos = []
        
        # 区域外：基体
        final_pos.append(pos_A[~mask_wedge])
        
        # 区域内：孪晶
        # 必须判断旋转后的原子 pos_B 是否在区域内
        v_B = pos_B - pivot
        mask_wedge_B = (v_B[:, 2] > np.abs(v_B[:, 1]) * slope)
        final_pos.append(pos_B[mask_wedge_B])
        
        # 5. 边界修复 (自动补全)
        all_pos = np.vstack(final_pos)
        
        # 检查空隙 (Void)
        # 构建 NeighborList 或简单的距离矩阵
        # 这是一个 O(N^2) 操作，但 N~300 很快。
        # 寻找“本该有原子但没有”的地方。
        # 简单策略：遍历 Grid A 中在边界附近的原子。
        # 如果一个原子既不在 Final 列表中（被切掉了），
        # 且它周围 (r < 2.0 A) 没有 Final 中的原子（即没有被 Twin 原子替代），
        # 那么它就是一个空洞，需要补回来。
        
        # Grid A 中被切掉的原子
        removed_A = pos_A[mask_wedge]
        
        # Grid B 中被切掉的原子 (不在 Wedge 内的)
        removed_B = pos_B[~mask_wedge_B]
        
        # 候选补全原子 = removed_A + removed_B
        candidates = np.vstack([removed_A, removed_B])
        
        # 验证候选原子是否与现存原子重叠
        # 如果不重叠，则补入。
        from ase.geometry import get_distances
        
        # 现有原子
        current_atoms = Atoms(positions=all_pos, cell=cell, pbc=True)
        
        # 筛选候选者
        to_add = []
        cutoff = 1.8 # 最小允许间距
        
        # 批量计算距离比较慢，逐个检查候选者
        # 优化：只检查边界附近的。
        
        for cand in candidates:
            # 检查 cand 是否在 V 形边界附近
            # v = cand - pivot
            # dist_to_boundary = abs(z - |y|*slope)
            # if dist_to_boundary > 2.0: continue
            
            # 检查与现有原子的距离
             # 考虑 PBC
             # 使用 ase.geometry.get_distances 来计算单个点到一组点的距离
             # p1: shape (1, 3), p2: shape (N, 3)
             # returns: vector, distance^2 (or distance)
             # get_distances(p1, p2, cell, pbc) -> (D_vec, D_sq) (D_vec is vector matrix)
             
             D_vec, D_sq = get_distances(cand.reshape(1,3), all_pos, cell=cell, pbc=True)
             
             # D_sq is (1, N)
             min_dist = np.sqrt(np.min(D_sq))
             
             if min_dist > cutoff:
                 to_add.append(cand)
                 # 同时也把这个新原子加入 all_pos，防止后续候选者与它重叠
                 all_pos = np.vstack([all_pos, cand])
        
        # 6. 重建
        symbol = base_atoms.get_chemical_symbols()[0]
        final_atoms = Atoms(symbol * len(all_pos), positions=all_pos, cell=cell, pbc=True)
        
        # 最后再做一次 remove_overlaps 以防万一
        final_atoms = AtomUtils.remove_overlaps(final_atoms, cutoff=1.5)
        
        return final_atoms
        """
        构建多重平行交叉孪晶 (改进版：原子锁定旋转)
        :param base_atoms: 基础原子结构 (假定为 FCC111 取向)
        :param plane_miller: 孪晶面指数 (在正交坐标系下的近似方向)
        :param width_A: 孪晶板条宽度
        :param spacing_A: 间距
        :param num_twins: 数量
        """
        # 1. 确定旋转轴和法向量
        # 在 FCC111 坐标系中 (Z // [111]):
        # 主滑移面是 XY 平面。
        # 交叉孪晶面 (Conjugate Twin) 通常与 Z 轴成 ~70.53 度。
        # 我们可以选择法向量 n 在 YZ 平面上倾斜。
        # n = [0, sin(70.53), cos(70.53)] = [0, 2sqrt(2)/3, 1/3]
        
        # 这种取向对应于 (11-1) 面（如果 X 是 [1-10]）。
        cos_theta = 1.0/3.0
        sin_theta = np.sqrt(8.0)/3.0
        # 定义孪晶面的法向量 (也是旋转轴)
        # 注意：ASE 的 fcc111 构建出来的 X 轴通常是 [1-10], Y 是 [11-2] (或反之，视 orthogonal 参数)
        # 无论如何，我们只要定义一个几何上正确的倾斜面即可。
        # 旋转轴 n 必须是晶体学上的 <111> 方向。
        # 在 fcc111 坐标系下，Z 是 [111]。
        # 其他 <111> 方向（如 [-1 1 1]）在这个坐标系下的分量是关键。
        # 简单起见，我们直接使用几何构造的 n，并假设它对应某个 <111>。
        normal = np.array([0.0, sin_theta, cos_theta]) # 在 YZ 平面倾斜
        
        # 2. 准备参数
        pos = base_atoms.get_positions()
        cell = base_atoms.get_cell()
        center_of_box = np.sum(cell, axis=0) * 0.5
        
        # 定义板条偏移
        offsets = (np.arange(num_twins) - (num_twins - 1) / 2.0) * spacing_A
        
        # 3. 执行构建 (Cut and Rotate)
        # 为了保证原子匹配，我们采用如下策略：
        # 遍历每个板条：
        #   找到板条中心的某个原子 -> 作为旋转锚点 (Anchor)
        #   选出板条区域内的所有原子
        #   绕 Anchor 旋转 60 度 (绕 normal 轴)
        
        # 预计算所有原子到中心的投影距离 d = r . n
        # 但要注意，这里的 n 是相对于盒子中心的。
        # 为了简单，我们在盒子中心建立参考系。
        rel_pos = pos - center_of_box
        dists = np.dot(rel_pos, normal)
        
        new_positions = pos.copy()
        
        # 旋转矩阵构建 (绕 normal 旋转 60 度)
        # Angle = 60 deg = pi/3
        # R = I + sin(t) [n]x + (1-cos(t)) [n]x^2
        th = np.pi / 3.0
        c_rot = np.cos(th)
        s_rot = np.sin(th)
        nx, ny, nz = normal
        
        # Rodrigues rotation formula matrix
        K = np.array([
            [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0]
        ])
        R = np.eye(3) + s_rot * K + (1 - c_rot) * (K @ K)
        
        # 记录被修改的原子索引，用于后续清理
        modified_indices = []
        
        for off in offsets:
            # 定义当前板条区域
            # d_local = dists - off
            mask = np.abs(dists - off) < (width_A / 2.0)
            idxs = np.where(mask)[0]
            
            if len(idxs) == 0: continue
            
            # **关键步骤**：寻找锚点
            # 在选中的原子中，找一个最接近该板条几何中心 (center_of_box + off*n) 的原子
            # 这样可以确保旋转轴穿过或是非常接近一个原子位置
            # 从而保持晶格点阵的重合性
            
            # 板条理想中心
            ideal_center = center_of_box + normal * off
            # 计算选中原子到理想中心的距离平方
            d_sq = np.sum((pos[idxs] - ideal_center)**2, axis=1)
            pivot_idx = idxs[np.argmin(d_sq)] # 锚点原子索引
            pivot_pos = pos[pivot_idx]
            
            # 对区域内原子进行旋转
            # v = r - pivot
            # v_new = R @ v
            # r_new = v_new + pivot
            
            subset_pos = pos[idxs]
            v = subset_pos - pivot_pos
            # 矩阵乘法: (N,3) @ (3,3).T -> (N,3)
            v_new = np.dot(v, R.T)
            new_positions[idxs] = v_new + pivot_pos
            
            modified_indices.extend(idxs)
            
        # 更新位置
        base_atoms.positions = new_positions
        
        # 4. 冲突处理
        # 旋转后，边界处可能会有原子靠得太近。
        # 使用较大的 cutoff 清理重叠，因为孪晶界处的原子间距不应小于最近邻距离 (~2.55A for Cu)
        # 设置 cutoff = 2.0 A (安全距离)
        # 注意：这种原位旋转可能会导致边界产生空隙或重叠。
        # 更严格的方法是生成全尺寸孪晶再剪切，但锚点旋转法在保持局部对称性上通常表现不错。
        
        # 必须处理 PBC 带来的问题吗？
        # 如果板条不穿过 PBC 边界，没问题。如果穿过，可能会断裂。
        # 假设盒子足够大，板条在内部。
        
        base_atoms = AtomUtils.remove_overlaps(base_atoms, cutoff=1.8)
        
        return base_atoms
            
        # 3. 旋转原子 (Twin Operation)
        # 绕法向量 n 旋转 180 度。
        # 旋转中心：center + n * dist (投影点)？
        # 不，应该是关于“孪晶中心面”的镜像。
        # 镜像操作：r' = r - 2 * ((r - center) . n) * n
        # 这会将原子翻转到板条的另一侧？
        # 不，我们想要的是改变晶格取向，而不是把原子搬运到别处。
        # 孪晶操作通常是绕孪晶轴（即法线 n）旋转 60度 或 180度。
        # FCC 孪晶：绕 [111] (即 n) 旋转 60 度 (ABC -> ACB)。
        # 或者镜像反映。
        # 如果我们做原位旋转，需要以每个原子自身的某个格点为中心？不，是以整体区域为中心。
        
        # 让我们尝试绕 n 轴旋转 180 度 (相当于中心对称+镜像)。
        # 对于 FCC，绕 <111> 旋转 60 度是孪晶。旋转 180 度也是孪晶关系（但在某些轴系下）。
        # 图片说："绕 [111] 轴旋转 60° 或 180°"。
        # 这里的 [111] 指的是**板状区域的法线**。
        # 所以我们绕 n 轴旋转 180 度。
        
        # 构建旋转矩阵 (Angle Axis)
        # R(n, theta)
        # Rodrigues formula
        # v_rot = v cos t + (n x v) sin t + n (n . v) (1 - cos t)
        # theta = 180 deg -> cos=-1, sin=0
        # v_rot = -v + 2 n (n . v)
        # 这实际上就是关于 n 的镜像 (如果是 v_rot = v - 2 n (n.v) 是关于面反射)。
        # Wait, v_rot = -v + 2n(n.v) 是关于直线的 180 度旋转。
        
        # 选择旋转中心：板条的几何中心 `center`。
        sel_pos = pos[indices]
        v = sel_pos - center
        
        # 应用旋转 (180度)
        # v_new = -v + 2 * n * (v . n)
        # 向量化计算
        v_dot_n = np.dot(v, normal) # shape (N,)
        v_dot_n = v_dot_n[:, np.newaxis] # shape (N, 1)
        v_new = -v + 2 * normal * v_dot_n
        
        new_pos_abs = v_new + center
        
        # 更新位置
        base_atoms.positions[indices] = new_pos_abs
        
        # 4. 清理重叠
        # 旋转后，边界处的原子可能与未旋转的基体原子重叠。
        # 必须清理。
        base_atoms = AtomUtils.remove_overlaps(base_atoms, cutoff=1.5) # Cu bond ~ 2.5, cutoff < 2.5. 1.5 is safe.
        
        return base_atoms
