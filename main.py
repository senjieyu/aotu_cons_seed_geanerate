# -*- coding: utf-8 -*-
# main.py
import os
import time
import threading
from datetime import datetime
from contextlib import contextmanager
import numpy as np

import model_construct as mc
import make_poly_and_blocks as mpb
import basic as basicgen
import cons_batch_generate as batch

# ======================================================================
# CONFIGURATION
# ======================================================================
PROJECT_NAME = "AOTU_run"
RUN_NAME = None

MONITOR = {"enable_heartbeat": True, "heartbeat_sec": 10}
STRUCTURE_TYPE = "FCC"

MAT_CONFIG = {
    "element": "Cu",
    "lattice_a": 3.615,
    "lattice_c": None, # Restore lattice_c
    # size: (x, y, z) in unit cells
    # orthogonal=False (Rhombic) does not require even Y.
    # Revert to original user setting (3, 3, 12)
    "size": (3, 3, 12),
    "amp": 0.0, # Disable perturbation globally
    "vacuum": 0.0, 
}

TASKS = {
    "1d_main": {"enable": True, "n_points": 21, "n_perturb": 0},
    "1d_sec": {"enable": True, "n_points": 21, "n_perturb": 0},
    "2d_surf": {"enable": True, "grid_n": 6, "n_perturb": 0},
    "key_pts": {"enable": True, "fractions": [0.0, 1.0], "n_perturb": 0},
    
    # New task for systematic twin generation
    "twin_systematic": {"enable": True},
    
    # Intersecting twin
    "twin_intersecting": {"enable": False},
    
    # Cut from Existing
    "cut_complex": {"enable": True},
}

POLY_BLOCK_CONFIG = {"enable": False, "element": MAT_CONFIG["element"], "structure": STRUCTURE_TYPE,
                     "lattice_a": MAT_CONFIG["lattice_a"], "use_twin_unit": True,
                     "twin_orient": ("[11-2]", "[111]", "[-110]"), "twin_duplicate": (1, 3, 1),
                     "poly_box": (10, 10, 10), "poly_grains": 2, "max_poly_atoms": 300000,
                     "block_atom_range": (100, 300), "block_window_min": (12, 12, 12), "block_window_max": (20, 20, 20),
                     "n_blocks": 40, "seed": 42, "block_perturb_amp": 0.0}
BASIC_GEN_CONFIG = {"enable": False, "structure": STRUCTURE_TYPE, "element": MAT_CONFIG["element"],
                    "lattice_a": MAT_CONFIG["lattice_a"], "lattice_c": MAT_CONFIG["lattice_c"], "supercell": (3, 3, 3),
                    "vacancy_index": 1, "output_dir": None, "name_prefix": "Basic", "verbose": True}


# ======================================================================
# LOGGING & UTILS
# ======================================================================
class DualLogger:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        print(f"[Log] Log file: {filepath}")

    def write(self, msg):
        print(msg)
        self.file.write(msg + '\n')
        self.file.flush()

    def close(self): self.file.close()


GLOBAL_LOGGER = None


def log(msg):
    if GLOBAL_LOGGER:
        GLOBAL_LOGGER.write(msg)
    else:
        print(msg)


def log_structure_details(name, atoms, desc, params=None):
    if GLOBAL_LOGGER is None: return
    cell = atoms.get_cell()
    msg = [f"[Struct] {name}", f"  > Desc: {desc}", f"  > Atoms: {len(atoms)}",
           f"  > Cell: {np.diag(cell)}"]
    if params: msg.append(f"  > Params: {params}")
    msg.append("-" * 30)
    log("\n".join(msg))


def _now_tag(): return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p


class Heartbeat:
    def __init__(self, name, heartbeat_sec=10, enable=True):
        self.name, self.sec, self.enable = name, heartbeat_sec, enable
        self._stop, self._thread = threading.Event(), None

    def start(self):
        if self.enable: self._stop.clear(); self._thread = threading.Thread(target=self._run,
                                                                            daemon=True); self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.sec): log(f"    ... {self.name} ...")

    def stop(self):
        if self.enable: self._stop.set(); self._thread.join(1.0)


@contextmanager
def stage(name, heartbeat=False):
    log(f"\n>>> [START] {name}")
    hb = Heartbeat(name, enable=heartbeat).start()
    try:
        yield
    finally:
        hb.stop(); log(f">>> [DONE ] {name}")


def print_job_info(struct, conf, tasks, dirs):
    log("=" * 60 + f"\nJOB DASHBOARD: {struct} {conf['element']}\n" + "=" * 60)
    log(f"Output Dir: {dirs['out_root']}")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    try:
        ROOT = os.getcwd()
        OUT = _ensure_dir(os.path.join(ROOT, "outputs", PROJECT_NAME, RUN_NAME or _now_tag()))
        GLOBAL_LOGGER = DualLogger(os.path.join(OUT, "run.log"))

        # 目录准备
        D_BASE = _ensure_dir(os.path.join(OUT, "00_base_model"))
        D_BATCH = _ensure_dir(os.path.join(OUT, "10_batch"))
        # 【关键修复】这里显式创建 relax 文件夹
        D_RELAX = _ensure_dir(os.path.join(D_BATCH, "relax"))

        D_POLY = _ensure_dir(os.path.join(OUT, "20_poly_block"))
        D_BASIC = _ensure_dir(os.path.join(OUT, "30_basic"))

        # 注入配置
        BASIC_GEN_CONFIG["output_dir"] = D_BASIC
        for k in TASKS: TASKS[k].update(
            {"amp": MAT_CONFIG["amp"], "element": MAT_CONFIG["element"], "struct_type": STRUCTURE_TYPE})

        # 1. Base Slab
        with stage("Build Base"):
            bf = mc.StructureFactory()
            if STRUCTURE_TYPE == "FCC":
                base, v1, v2, inf = bf.build_fcc(MAT_CONFIG["element"], MAT_CONFIG["lattice_a"], MAT_CONFIG["size"],
                                                 MAT_CONFIG["vacuum"])
            else:
                raise ValueError("Only FCC is fully supported for this complex twin demo")

            mc.AtomUtils.write_vasp_file(base, f"Base_{STRUCTURE_TYPE}", D_BASE)
            log_structure_details("Base", base, "Initial Matrix")

        # 定义目录字典供 batch 调用
        dirs = {"out_root": OUT, "out": D_BATCH, "relax": D_RELAX}
        print_job_info(STRUCTURE_TYPE, MAT_CONFIG, TASKS, dirs)

        # 2. General Batch (1D/2D)
        with stage("Batch Gen", heartbeat=True):
            # 2. 1D Main Path
            # ---------------------------------------------------------------------------------------------
            if TASKS["1d_main"]["enable"]:
                batch.generate_1d_main(base, v1, MAT_CONFIG, dirs, log_structure_details)

            # 3. 1D Secondary Path
            # ---------------------------------------------------------------------------------------------
            if TASKS["1d_sec"]["enable"]:
                batch.generate_1d_sec(base, v2, MAT_CONFIG, dirs, log_structure_details)

            # 4. 2D Surface
            # ---------------------------------------------------------------------------------------------
            if TASKS["2d_surf"]["enable"]:
                batch.generate_2d_surface(base, v1, v2, MAT_CONFIG, dirs, log_structure_details)
            
            # 5. Key Points
            # ---------------------------------------------------------------------------------------------
            if TASKS["key_pts"]["enable"]:
                batch.generate_key_points(base, v1, MAT_CONFIG, dirs, log_structure_details)

        # 3. Twin Systematic
        with stage("Twin Systematic", heartbeat=True):
            if TASKS.get("twin_systematic", {}).get("enable"):
                # Use new function name generate_twin
                batch.generate_twin(MAT_CONFIG, dirs, log_structure_details)

            if TASKS.get("twin_intersecting", {}).get("enable"):
                # Not ported yet, keeping disabled
                pass

            if TASKS.get("cut_complex", {}).get("enable"):
                batch.generate_cut_from_existing(dirs, MAT_CONFIG, log_structure_details)

        # 4. Poly & Basic (Optional Collect)
        if POLY_BLOCK_CONFIG.get("enable"):
            pass  # 如果需要，可在此添加多晶加载逻辑

        log(f"Done. Logs: {GLOBAL_LOGGER.file.name}")

    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        if GLOBAL_LOGGER: GLOBAL_LOGGER.close()
