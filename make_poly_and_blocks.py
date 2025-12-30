# -*- coding: utf-8 -*-
# make_poly_and_blocks.py
"""
make_poly_and_blocks.py

功能：
1) Atomsk 生成（可选）孪晶晶胞
2) Voronoi 多晶
3) 输出整体多晶（VASP）
4) 在多晶中稳定切出 100~300 原子的 block

特性：
- 使用临时目录，避免 Atomsk overwrite 交互
- 原子数预估，防炸
- 3x3x3 影像超胞切块（解决 PBC 跨边界）
- 自适应放大窗口，保证 block 能切出来
- 写 VASP 兼容旧版 ASE（无 direct 参数也可用）
"""

import os
import shutil
import subprocess
import tempfile
import numpy as np

from ase.io import read as ase_read
from ase.atoms import Atoms

# 兼容不同版本 ASE 的 VASP writer
try:
    from ase.io.vasp import write_vasp as _write_vasp
except Exception:
    _write_vasp = None


# ============================================================
# 基础工具
# ============================================================
def find_atomsk():
    """查找系统中的 atomsk 可执行文件"""
    atomsk = shutil.which("atomsk") or shutil.which("atomsk.exe")
    if atomsk is None:
        raise RuntimeError("找不到 atomsk，请加入 PATH")
    return atomsk


def run_cmd(cmd, cwd=None):
    """
    运行命令行指令

    Args:
        cmd: 命令列表
        cwd: 工作目录
    """
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"[CMD FAILED]\nCMD: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def ensure_dir(d):
    """确保目录存在，若不存在则创建"""
    os.makedirs(d, exist_ok=True)


def write_vasp_compat(atoms: Atoms, path: str):
    """
    兼容旧版 ASE 的 VASP 文件写入方法

    Args:
        atoms: ASE Atoms 对象
        path: 输出文件路径
    """
    atoms.wrap()

    if _write_vasp is None:
        # 最后兜底：用 ase.io.write
        from ase.io import write
        try:
            write(path, atoms, format="vasp", vasp5=True)
        except TypeError:
            write(path, atoms, format="vasp")
        return

    # 尝试不同签名
    try:
        _write_vasp(path, atoms, vasp5=True, sort=False)
        return
    except TypeError:
        pass

    try:
        _write_vasp(path, atoms, vasp5=True)
        return
    except TypeError:
        pass

    _write_vasp(path, atoms)


def add_perturb(atoms, amp, seed):
    """
    对原子位置添加随机扰动

    Args:
        atoms: ASE Atoms 对象
        amp: 扰动幅度
        seed: 随机种子

    Returns:
        添加扰动后的 Atoms 对象
    """
    if amp <= 0:
        return atoms
    rng = np.random.default_rng(seed)
    a = atoms.copy()
    a.positions += (rng.random((len(a), 3)) - 0.5) * 2.0 * amp
    a.wrap()
    return a


# ============================================================
# 原子数估算（防炸）
# ============================================================
def estimate_atoms_fcc(a, box):
    """
    估计 FCC 结构中的原子数

    Args:
        a: 晶格常数
        box: 盒子尺寸 [x, y, z]

    Returns:
        估计的原子数
    """
    vol = float(box[0]) * float(box[1]) * float(box[2])
    return int(vol * (4.0 / a ** 3))


# ============================================================
# FCC 孪晶晶胞（镜像 + merge）
# ============================================================
def build_fcc_twin_unit(atomsk, element, a, orient, dup, tmpdir):
    """
    构建 FCC 孪晶单元

    Args:
        atomsk: atomsk 可执行文件路径
        element: 元素符号
        a: 晶格常数
        orient: 晶体取向 [o1, o2, o3]
        dup: 重复次数 [nx, ny, nz]
        tmpdir: 临时目录路径

    Returns:
        孪晶单元文件路径
    """
    o1, o2, o3 = orient
    nx, ny, nz = dup

    cell = os.path.join(tmpdir, "cell.xsf")
    mirror = os.path.join(tmpdir, "mirror.xsf")
    final = os.path.join(tmpdir, "twin_unit.xsf")

    run_cmd([
        atomsk, "--create", "fcc", str(a), element,
        "orient", o1, o2, o3,
        "-duplicate", str(nx), str(ny), str(nz),
        cell
    ], cwd=tmpdir)

    run_cmd([atomsk, cell, "-mirror", "0", "Y", "-wrap", mirror], cwd=tmpdir)
    run_cmd([atomsk, "--merge", "Y", "2", cell, mirror, final], cwd=tmpdir)

    return final


# ============================================================
# Voronoi 多晶
# ============================================================
def write_poly_txt(tmpdir, box, ngrains):
    """
    写入多晶配置文件

    Args:
        tmpdir: 临时目录路径
        box: 盒子尺寸 [x, y, z]
        ngrains: 晶粒数量

    Returns:
        配置文件路径
    """
    path = os.path.join(tmpdir, "poly.txt")
    with open(path, "w") as f:
        f.write(f"box {box[0]} {box[1]} {box[2]}\n")
        f.write(f"random {int(ngrains)}\n")
    return path


def build_polycrystal(atomsk, seed_xsf, poly_txt, tmpdir):
    """
    构建多晶结构

    Args:
        atomsk: atomsk 可执行文件路径
        seed_xsf: 种子晶胞文件路径
        poly_txt: 多晶配置文件路径
        tmpdir: 临时目录路径

    Returns:
        多晶结构文件路径
    """
    out = os.path.join(tmpdir, "poly.lmp")
    run_cmd([atomsk, "--polycrystal", seed_xsf, poly_txt, out, "-wrap"], cwd=tmpdir)
    return out


# ============================================================
# 高命中切块（3x3x3 影像 + 自适应窗口）
# ============================================================
def crop_blocks(atoms, out_dir, n_blocks, atom_range,
                win_min, win_max, seed, perturb_amp):
    """
    从多晶结构中切割出小块

    Args:
        atoms: ASE Atoms 对象（多晶结构）
        out_dir: 输出目录
        n_blocks: 需要切割的块数
        atom_range: 原子数范围 [min, max]
        win_min: 窗口最小尺寸
        win_max: 窗口最大尺寸
        seed: 随机种子
        perturb_amp: 扰动幅度

    Returns:
        成功保存的块数
    """
    blocks_dir = os.path.join(out_dir, "Poly_blocks")
    ensure_dir(blocks_dir)

    rng = np.random.default_rng(seed)

    nmin, nmax = atom_range
    wmin = np.array(win_min, float)
    wmax = np.array(win_max, float)

    cell = atoms.cell
    L = atoms.cell.lengths()
    base_pos = atoms.positions
    Z = atoms.get_atomic_numbers()

    # 3x3x3 影像（解决窗口跨边界）
    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shifts.append(i * cell[0] + j * cell[1] + k * cell[2])
    shifts = np.array(shifts)

    img_pos = np.vstack([base_pos + s for s in shifts])
    img_Z = np.hstack([Z for _ in shifts])

    saved = 0
    attempts = 0

    max_attempts = n_blocks * 2000
    expand_every = 200
    expand_factor = 1.15

    while saved < n_blocks and attempts < max_attempts:
        attempts += 1

        center = rng.random(3) * L
        win = wmin + rng.random(3) * (wmax - wmin)

        lo = center - 0.5 * win
        hi = center + 0.5 * win

        mask = np.all((img_pos >= lo) & (img_pos <= hi), axis=1)
        pos_sel = img_pos[mask]
        Z_sel = img_Z[mask]

        if nmin <= len(pos_sel) <= nmax:
            pos_local = pos_sel - lo
            sub = Atoms(
                numbers=Z_sel,
                positions=pos_local,
                cell=np.diag(win),
                pbc=[True, True, True]
            )
            sub.wrap()

            if perturb_amp > 0:
                sub = add_perturb(sub, perturb_amp, seed + saved)

            fn = os.path.join(blocks_dir, f"block_{saved:03d}_N{len(sub)}.vasp")
            write_vasp_compat(sub, fn)
            saved += 1

        # 若长时间一个都没切到，自动放大窗口
        if attempts % expand_every == 0 and saved == 0:
            wmin *= expand_factor
            wmax *= expand_factor

    if saved == 0:
        print("  [WARN] 没切出 block：请增大 block_window_max 或放宽 block_atom_range")

    return saved


# ============================================================
# main.py 调用入口
# ============================================================
def run_from_config(cfg, out_dir):
    """
    根据配置运行整个流程

    Args:
        cfg: 配置字典
        out_dir: 输出目录

    Returns:
        包含结果信息的字典
    """
    ensure_dir(out_dir)
    atomsk = find_atomsk()

    element = cfg["element"]
    a = float(cfg["lattice_a"])

    box = cfg["poly_box"]
    est = estimate_atoms_fcc(a, box)

    if est > cfg.get("max_poly_atoms", 300000):
        raise RuntimeError(
            f"poly_box={box} Å 预计 {est} 原子，太大了，请调小 poly_box 或提高 max_poly_atoms"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) seed（孪晶晶胞）
        seed_xsf = build_fcc_twin_unit(
            atomsk, element, a,
            orient=cfg["twin_orient"],
            dup=cfg["twin_duplicate"],
            tmpdir=tmpdir
        )

        # 2) poly
        poly_txt = write_poly_txt(tmpdir, box, cfg["poly_grains"])
        poly_lmp = build_polycrystal(atomsk, seed_xsf, poly_txt, tmpdir)

        # 3) read full poly (Lammps data)
        # 用 atom_style 避免 FutureWarning
        atoms = ase_read(poly_lmp, format="lammps-data", atom_style="atomic")
        atoms.set_chemical_symbols([element] * len(atoms))
        atoms.pbc = [True, True, True]
        atoms.wrap()

        # 4) write full vasp
        full_vasp = os.path.join(out_dir, "Polycrystal_full.vasp")
        write_vasp_compat(atoms, full_vasp)

        # 5) crop blocks
        saved = crop_blocks(
            atoms,
            out_dir,
            cfg["n_blocks"],
            cfg["block_atom_range"],
            cfg["block_window_min"],
            cfg["block_window_max"],
            cfg["seed"],
            cfg.get("block_perturb_amp", 0.0)
        )

        return {
            "full_atoms": len(atoms),
            "blocks_saved": saved,
            "blocks_dir": os.path.join(out_dir, "Poly_blocks"),
            "full_vasp": full_vasp,
        }
