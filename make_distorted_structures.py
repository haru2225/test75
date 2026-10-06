"""Generate DISTORTED crystal structures for test75, following Erhard et
al. (2022, npj Comput Mater)'s training-set design: rather than only
relaxed/equilibrium structures, include randomly strained + rattled
variants so the model sees off-equilibrium local environments (large
restoring forces, compressed/stretched bonds) during training -- not just
near-equilibrium thermal noise around a few discrete minima.

For each base structure (currently: the beta-cristobalite primitive cell),
applies a few random strain tensors (+-5-15% per axis, independent per
axis so the cell becomes anisotropic) plus small atomic rattle, then runs
a SHORT real Vashishta NVT relaxation at 300 K (not a static/unrelaxed
distortion -- the few thousand steps let the local Si-O network respond
to the imposed strain rather than just sitting at an arbitrary unphysical
point) and extracts the Si-only CG trajectory, same convention as test70-74.
"""
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read
from ase.io.lammpsdata import write_lammps_data

SOURCE_DATA = Path("/Users/harutokono/ScoreMD/md/silica_beta_cristobalite_init.data")
WORK_DIR = Path(__file__).resolve().parent / "distorted_work"
POTENTIALS_DIR = Path(
    "/Users/harutokono/ScoreMD/toy-model/SiO2-CG/diffusion_for_multi_scale_molecular_dynamics"
    "/.venv/lib/python3.10/site-packages/lammps/share/lammps/potentials"
)
OUT_ROOT = Path(__file__).resolve().parent
EQ_STEPS = 2000
PROD_STEPS = 5000
DUMP_EVERY = 50

# (name, per-axis strain fractions e.g. 0.10 = +10% along that axis, random rattle std in A)
DISTORTIONS = [
    ("distorted_stretch_p10", (1.10, 1.10, 1.10), 0.05),
    ("distorted_compress_m10", (0.90, 0.90, 0.90), 0.05),
    ("distorted_shear_aniso", (1.12, 0.94, 1.03), 0.08),
    ("distorted_rattle_only", (1.0, 1.0, 1.0), 0.15),
]


def main():
    base_atoms = read(str(SOURCE_DATA), format="lammps-data", style="atomic")
    symbols = ["Si" if n == 1 else "O" for n in base_atoms.get_atomic_numbers()]
    base_atoms.set_chemical_symbols(symbols)

    for name, strain, rattle_std in DISTORTIONS:
        atoms = base_atoms.copy()
        cell = atoms.get_cell().array.copy()
        for axis in range(3):
            cell[axis] *= strain[axis]
        atoms.set_cell(cell, scale_atoms=True)
        rng = np.random.default_rng(hash(name) % (2**32))
        atoms.positions += rng.normal(0.0, rattle_std, size=atoms.positions.shape)
        atoms.wrap()

        work = WORK_DIR / name
        work.mkdir(parents=True, exist_ok=True)
        data_path = work / "init.data"
        with open(data_path, "w") as f:
            write_lammps_data(f, atoms, specorder=["Si", "O"], force_skew=True, units="metal", atom_style="atomic")

        dump_path = work / f"{name}.lammpstrj"
        in_path = work / "run.in"
        in_path.write_text(f"""
units metal
boundary p p p
atom_style atomic
read_data {data_path.name}
mass 1 28.0855
mass 2 15.999
pair_style vashishta
pair_coeff * * {POTENTIALS_DIR}/SiO.1990.vashishta Si O
velocity all create 300.0 42 mom yes rot yes dist gaussian
fix eqnvt all nvt temp 300.0 300.0 0.1
run {EQ_STEPS}
unfix eqnvt
fix prodnvt all nvt temp 300.0 300.0 0.1
dump traj all custom {DUMP_EVERY} {dump_path.name} id type xu yu zu
dump_modify traj sort id append no format float %.6f
run {PROD_STEPS}
""")
        result = subprocess.run(["lmp_serial", "-in", str(in_path)], cwd=str(work),
                                 capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"{name}: LAMMPS FAILED: {result.stderr[-1000:]}")
            continue
        print(f"{name}: {result.stdout.splitlines()[-1] if result.stdout else '(ok)'}")

        text = dump_path.read_text()
        blocks = text.split("ITEM: TIMESTEP")[1:]
        frames, out_cell = [], None
        for block in blocks:
            lines = block.strip().splitlines()
            n_atoms = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
            bounds_idx = next(i for i, line in enumerate(lines) if line.startswith("ITEM: BOX BOUNDS"))
            is_tri = "xy xz yz" in lines[bounds_idx]
            bounds_rows = [lines[bounds_idx + 1 + i].split() for i in range(3)]
            if is_tri:
                lengths = np.array([float(r[1]) - float(r[0]) for r in bounds_rows])
            else:
                lengths = np.array([float(hi) - float(lo) for lo, hi in bounds_rows])
            if out_cell is None:
                out_cell = lengths
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
        np.savez(out_dir / "manifest.npz", cell_nm=(out_cell / 10.0).astype(np.float32), source=name)
        print(f"  Wrote {out_dir}: {si_frames.shape[0]} frames, {si_frames.shape[1]} Si atoms")


if __name__ == "__main__":
    main()
