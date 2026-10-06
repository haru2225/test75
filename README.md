# test75: distorted structures + NPT pressure exploration (SiO2 CG force field)

Extends test74 (T-conditioned, 8 real local conditions: crystal, 4
intermediate temperatures, melt, melt-quenched glass, stishovite) following
the training-set design philosophy of Erhard et al. 2022 (npj Comput
Mater 8:90, the SiO2 GAP paper): rather than only equilibrium structures at
a few discrete state points, cover the local potential-energy surface more
broadly with distorted/strained structures and real NPT (pressure-
controlled) exploration.

## What is implemented and verified in this session

- **`make_distorted_structures.py`**: generates 4 distorted variants of
  the beta-cristobalite structure (uniform +10%/-10% volumetric strain,
  anisotropic shear-like strain, and rattle-only) by applying the strain
  to the cell + Gaussian rattle to positions, then relaxing with a real
  short Vashishta NVT run (so the network sees how the real potential
  responds to an imposed distortion, not an arbitrary unrelaxed snapshot).
- **`make_npt_data.py`**: runs real LAMMPS `fix npt` (true barostatted
  pressure control, first use of NPT anywhere in this project's history --
  test69-74 only ever used NVT) on cristobalite and stishovite at several
  target pressures, and records the ACHIEVED mean pressure (not just the
  target) in each dataset's manifest.npz.
- `Singularity.test75.def`, `run_test75.pbs`: same supercomputer pattern
  as test73/74.

**Neither script has been run in this session** (time ran out) -- they
are implemented and should work based on the same patterns validated in
test70-74, but have NOT been smoke-tested. Run them first and check the
output before trusting the generated data.

## Known limitation: NPT data has a frame-varying cell, but the training
## pipeline assumes one fixed cell per dataset

test70-74's `DenoiserMPNN`/`periodic_radius_graph` and the whole
training loop load ONE cell tensor per dataset and reuse it for every
frame. Under NPT, the box volume genuinely fluctuates frame-to-frame (that
is the entire point of a barostat) -- `make_npt_data.py` currently just
saves the FINAL frame's cell as an approximation, which is wrong for any
frame where the box differs meaningfully from that final value. Before
training on NPT data for real, the pipeline needs either:
  (a) a per-frame cell tensor (requires passing cell into the training
      loop's batch sampling, not just once at dataset-load time), or
  (b) restricting to NPT trajectories where the box has already
      converged/stabilized (e.g. discard the first N frames, check the
      box-length standard deviation is small enough to treat as ~fixed).
This has NOT been fixed in this session -- flagging it explicitly rather
than silently training on the wrong cell.

## Known gap: coesite is NOT included

coesite (the ~2-10 GPa SiO2 polymorph, C2/c, Z=16, 48 atoms/cell) was
explicitly requested but is NOT implemented here. Building it correctly
requires its full set of Wyckoff positions (4 distinct Si sites + 8
distinct O sites in the asymmetric unit); this was not attempted from
memory because an incorrect coordinate would produce an unstable
structure and silently corrupt downstream results. To add it properly:
pull a verified coesite CIF (e.g. from the American Mineralogist
Crystal Structure Database, Materials Project, or a cited paper's SI) and
adapt `make_distorted_structures.py`'s / test73's build pattern.

## Running

```bash
# locally (smoke test / small-scale):
python make_distorted_structures.py   # ~few minutes, needs lmp_serial
python make_npt_data.py                # ~10-20 min, needs lmp_serial, fixes the cell caveat above before real use

# supercomputer:
singularity build --force --fakeroot test75-pytorch-2.5.0-cu124.sif Singularity.test75.def
qsub run_test75.pbs
```

`train_and_export.py` is copied unmodified from test74 (T-only
conditioning, no P input yet) -- extending `DATASETS` to include the new
distorted/NPT data, and extending `DenoiserMPNN`'s conditioning to take P
as well as T (same pattern as the (P,T) version sketched earlier in
test74's git history, before it was simplified to T-only), is the next
piece of actual work, not yet done.

## Result: bigger network + 18 datasets did NOT fix the melt-quench over-disordering

Trained with HIDDEN_DIM=128, N_LAYERS=4 (4x test74's capacity) on all 18
datasets (8 base + 4 distorted + 6 NPT). Final losses were lower than
test74's across the board, but the melt->quench->glass-formation
experiment (melt_quench_glass_formation.py, same protocol as test74)
gave essentially IDENTICAL final-state disorder: nn_std ~0.29-0.33 A
after quenching and holding at 300K, vs the real glass reference's
~0.08 A -- no improvement over test74's ~0.28-0.43 A.

Conclusion: neither "network too small" nor "T/structural coverage too
narrow" was the dominant bottleneck. The denoising (fixed-sigma score
matching) training objective itself likely has a structural accuracy
ceiling here. The recommended next step is switching to DIRECT force
supervision (regress against the real Vashishta forces, extractable from
LAMMPS for every training frame -- see test70's compute_true_force.py for
the extraction method already implemented) instead of denoising -- this
is also how the cited Erhard et al. GAP potential and essentially all
mainstream ML interatomic potentials are actually trained, rather than via
score-matching.
