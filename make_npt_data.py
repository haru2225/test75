"""NPT (true pressure control) data generation for test75. For each
(base structure, target pressure) combination, runs LAMMPS `fix npt` to
let the box volume relax to the target pressure at 300 K, then extracts
the Si-only CG trajectory in the same format as test70-74's data_*/
directories. This is the prerequisite for a genuinely pressure-aware
model (test73/74 only ever used NVT / fixed volume -- no real pressure
control at all).

Usage note: unlike test70-74's NVT scripts, each dataset here is also
labelled with its ACHIEVED mean pressure (from LAMMPS thermo output, not
just the target -- the barostat relaxes toward but does not exactly hit
the target instantaneously) in manifest.npz, so train_and_export.py (once
extended to use P as well as T -- not yet done in this script, see
README.md) has a real, measured P to condition on per dataset.
"""
import subprocess
from pathlib import Path

import numpy as np

POTENTIALS_DIR = Path(
    "/Users/harutokono/ScoreMD/toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics"
    "/.venv/lib/python3.10/site-packages/lammps/share/lammps/potentials"
)
OUT_ROOT = Path(__file__).resolve().parent
EQ_STEPS = 5000
PROD_STEPS = 10000
DUMP_EVERY = 50

# (base_structure_data_file, base_name, list of target pressures in GPa)
# Add coesite here once a verified structure file is available (see
# README.md's TODO -- NOT fabricated from memory in this session due to
# the risk of an incorrect, unstable structure).
JOBS = [
    (Path("/Users/harutokono/ScoreMD/md/silica_beta_cristobalite_init.data"), "cristobalite",
     [0.0, 10.0, 20.0]),
    (Path(__file__).resolve().parent.parent / "test74" / "stishovite_work" / "stishovite_init.data",
     "stishovite", [0.0, 20.0, 40.0, 60.0]),
]


def run_one(data_file, base_name, p_gpa):
    p_bar = p_gpa * 10000.0
    name = f"npt_{base_name}_{p_gpa:.0f}GPa"
    work = Path(__file__).resolve().parent / "npt_work" / name
    work.mkdir(parents=True, exist_ok=True)
    in_path = work / "run.in"
    in_path.write_text(f"""
units metal
boundary p p p
atom_style atomic
read_data {data_file}
mass 1 28.0855
mass 2 15.999
pair_style vashishta
pair_coeff * * {POTENTIALS_DIR}/SiO.1990.vashishta Si O
velocity all create 300.0 42 mom yes rot yes dist gaussian
fix eqnpt all npt temp 300.0 300.0 0.1 iso {p_bar} {p_bar} 1.0
thermo 500
thermo_style custom step temp press vol density
run {EQ_STEPS}
unfix eqnpt
fix prodnpt all npt temp 300.0 300.0 0.1 iso {p_bar} {p_bar} 1.0
dump traj all custom {DUMP_EVERY} {name}.lammpstrj id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
""")
    result = subprocess.run(["lmp_serial", "-in", str(in_path)], cwd=str(work),
                             capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        print(f"{name}: LAMMPS FAILED: {result.stderr[-1000:]}")
        return None
    thermo_lines = [l for l in result.stdout.splitlines() if l.strip() and l.strip()[0].isdigit()]
    mean_press_bar = np.mean([float(l.split()[2]) for l in thermo_lines[-20:]]) if thermo_lines else float("nan")
    print(f"{name}: done, mean press (prod) = {mean_press_bar / 10000.0:.2f} GPa")

    dump_path = work / f"{name}.lammpstrj"
    text = dump_path.read_text()
    blocks = text.split("ITEM: TIMESTEP")[1:]
    frames, cell = [], None
    for block in blocks:
        lines = block.strip().splitlines()
        n_atoms = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
        bounds_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: BOX BOUNDS"))
        bounds = [lines[bounds_idx + 1 + i].split() for i in range(3)]
        lengths = np.array([float(hi) - float(lo) for lo, hi in bounds])
        if cell is None:
            cell = lengths
        atoms_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
        header = lines[atoms_idx].split()[2:]
        id_col, type_col = header.index("id"), header.index("type")
        x_col, y_col, z_col = header.index("xu"), header.index("yu"), header.index("zu")
        rows = [lines[atoms_idx + 1 + i].split() for i in range(n_atoms)]
        rows.sort(key=lambda r: int(r[id_col]))
        types = np.array([int(r[type_col]) for r in rows])
        pos = np.array([[float(r[x_col]), float(r[y_col]), float(r[z_col])] for r in rows])
        frames.append(pos[types == 1])
    si_frames = np.stack(frames, axis=0)

    out_dir = OUT_ROOT / f"data_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "positions.npy", (si_frames / 10.0).astype(np.float32))
    np.savez(out_dir / "manifest.npz", cell_nm=(cell / 10.0).astype(np.float32),
              target_pressure_gpa=p_gpa, achieved_pressure_gpa=mean_press_bar / 10000.0,
              source=name)
    print(f"  Wrote {out_dir}: {si_frames.shape[0]} frames, {si_frames.shape[1]} Si atoms "
          f"(NOTE: cell is the time-AVERAGE-ish final box; NPT means volume fluctuates "
          f"frame-to-frame, unlike the fixed cell used by all of test70-74's NVT data)")


def main():
    for data_file, base_name, pressures in JOBS:
        for p_gpa in pressures:
            run_one(data_file, base_name, p_gpa)


if __name__ == "__main__":
    main()
