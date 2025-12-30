# -*- coding: utf-8 -*-
# basic.py
import os
from ase.build import bulk
from ase.io import write as ase_write
from ase.io import read as ase_read


def build_supercell(structure: str, element: str, lattice_a: float, lattice_c=None, supercell=(3, 3, 3)):
    """
    生成标准 bulk 超胞：
    - FCC/BCC 使用 cubic conventional cell（FCC=4原子，BCC=2原子）
    - HCP 使用 hex cell（原子数取决于 ASE 默认 cell）
    """
    st = structure.upper()
    if st == "FCC":
        atoms = bulk(element, "fcc", a=float(lattice_a), cubic=True)
    elif st == "BCC":
        atoms = bulk(element, "bcc", a=float(lattice_a), cubic=True)
    elif st == "HCP":
        if lattice_c is None:
            lattice_c = 1.633 * float(lattice_a)
        atoms = bulk(element, "hcp", a=float(lattice_a), c=float(lattice_c))
    else:
        raise ValueError(f"Unknown structure: {structure} (use FCC/BCC/HCP)")

    atoms = atoms.repeat(tuple(supercell))
    return atoms


def remove_one_atom(atoms, vacancy_index: int):
    """
    只删除一个原子（从当前 atoms 中），不做任何额外操作
    """
    n0 = len(atoms)
    if n0 == 0:
        raise RuntimeError("Empty atoms object, cannot remove vacancy.")

    idx = int(vacancy_index)
    if idx < 0 or idx >= n0:
        raise IndexError(f"vacancy_index out of range: {idx}, n={n0}")

    del atoms[idx]

    # 强制保证只删了一个
    if len(atoms) != n0 - 1:
        raise RuntimeError(f"Vacancy removal failed: before={n0}, after={len(atoms)}")

    return atoms


def write_vasp_and_verify(atoms, out_path: str):
    """
    写 POSCAR 并立刻读回校验原子数
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 不加 sort / direct 这类容易版本不兼容的参数，保持最稳
    ase_write(out_path, atoms, format="vasp")

    # 读回校验
    back = ase_read(out_path)
    if len(back) != len(atoms):
        raise RuntimeError(f"[I/O Verify Failed] write={len(atoms)} readback={len(back)} : {out_path}")

    return out_path


def run_from_config(cfg: dict):
    """
    cfg:
      structure, element, lattice_a, lattice_c
      supercell
      vacancy_index
      output_dir
      name_prefix
      verbose
    """
    structure = cfg["structure"]
    element = cfg["element"]
    lattice_a = float(cfg["lattice_a"])
    lattice_c = cfg.get("lattice_c", None)
    supercell = tuple(cfg.get("supercell", (3, 3, 3)))

    vacancy_index = int(cfg.get("vacancy_index", 0))
    output_dir = cfg.get("output_dir", os.getcwd())
    name_prefix = cfg.get("name_prefix", "Basic")
    verbose = bool(cfg.get("verbose", True))

    # 1) build supercell
    atoms = build_supercell(structure, element, lattice_a, lattice_c, supercell)
    n_before = len(atoms)

    # 2) remove exactly one atom
    atoms = remove_one_atom(atoms, vacancy_index)
    n_after = len(atoms)

    # 3) filename includes Natoms to avoid opening wrong file
    fname = f"{name_prefix}_{structure.upper()}_{element}_sc{supercell[0]}x{supercell[1]}x{supercell[2]}_vac{vacancy_index}_N{n_after}"
    out_path = os.path.join(output_dir, f"{fname}.vasp")

    # 4) write + readback verify
    write_vasp_and_verify(atoms, out_path)

    if verbose:
        print(f"[Basic] structure={structure.upper()} element={element} a={lattice_a} supercell={supercell}")
        print(f"[Basic] atoms before={n_before}, after vacancy={n_after} (removed index={vacancy_index})")
        print(f"[Basic] written: {out_path}")

    return {
        "n_before": n_before,
        "n_after": n_after,
        "vacancy_index": vacancy_index,
        "vasp_path": out_path,
    }
