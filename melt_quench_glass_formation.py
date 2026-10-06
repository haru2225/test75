"""The actual end goal: can test74's CG force field, starting from the
CRYSTAL structure, be melted (heated to T=3000K, both actual thermostat
and told condition) and then quenched (cooled back to T=300K) using its
OWN learned dynamics, to form a glass -- not just represent crystal/
melt/glass as separately-trained static targets, but reproduce the
physical melt->quench->glass PROCESS end to end.

Protocol:
  phase 1 (melt):   T_actual = T_told = 3000 K, N_MELT_STEPS steps
  phase 2 (quench): T_actual = T_told ramped linearly 3000K -> 300K, N_QUENCH_STEPS steps
  phase 3 (hold):   T_actual = T_told = 300 K, N_HOLD_STEPS steps

Success metric: does the FINAL structure's nn_std (disorder) resemble the
real glass reference (~0.08 A) rather than the real crystal reference
(~0.014 A), and does it NOT simply relax back to the crystal structure?
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_and_export import DenoiserMPNN  # noqa: E402

CKPT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"
KB_EV_PER_K = 8.617333e-5
MASS_AMU = 60.0843
DT_PS = 0.001
GAMMA_INV_PS = 1.0
METAL_UNITS_CONVERSION = 9648.533

N_MELT_STEPS = 1000
N_QUENCH_STEPS = 2000
N_HOLD_STEPS = 1000
STRIDE = 100


def nn_dist_stats(pos, cell):
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= np.round(diff / cell) * cell
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return nn.mean(), nn.min(), nn.std()


def step_once(model, sigma, cell, x, v, t_k):
    kbT_ev = KB_EV_PER_K * t_k
    gamma_per_ps = 1.0 / GAMMA_INV_PS
    alpha = np.exp(-gamma_per_ps * DT_PS)
    f_scale = (1 - alpha) / gamma_per_ps
    condition = torch.tensor([[t_k / 1000.0]], dtype=torch.float32)
    with torch.no_grad():
        raw = model(x.unsqueeze(0), cell, condition)[0]
    f = kbT_ev * raw / (sigma ** 2)
    noise = torch.randn_like(x)
    accel = f / MASS_AMU * METAL_UNITS_CONVERSION
    thermal_var = kbT_ev * (1 - alpha ** 2) / MASS_AMU * METAL_UNITS_CONVERSION
    v_new = alpha * v + f_scale * accel + np.sqrt(thermal_var) * noise
    x_new = x + DT_PS * v_new
    x_new = x_new - cell * torch.round(x_new / cell)
    return x_new, v_new


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = DenoiserMPNN(ckpt["hidden_dim"], ckpt["n_layers"], ckpt["cutoff_angstrom"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sigma = ckpt["sigma_angstrom"]
    cell_np = ckpt["cell_by_name"]["crystal_300K"]
    cell = torch.tensor(cell_np, dtype=torch.float32)

    x0_np = np.load(Path(__file__).resolve().parent / "data_crystal_300K" / "positions.npy").astype(np.float32)[0] * 10.0
    x = torch.tensor(x0_np, dtype=torch.float32)
    v = torch.zeros_like(x)
    torch.manual_seed(0)

    nn0 = nn_dist_stats(x0_np, cell_np)
    print(f"Start (crystal): nn_mean={nn0[0]:.4f} A  nn_min={nn0[1]:.4f} A  nn_std={nn0[2]:.4f} A")

    print(f"\n=== Phase 1: MELT (T=3000K held, {N_MELT_STEPS} steps) ===")
    for step in range(1, N_MELT_STEPS + 1):
        x, v = step_once(model, sigma, cell, x, v, 3000.0)
        if step % STRIDE == 0:
            nn_mean, nn_min, nn_std = nn_dist_stats(x.numpy(), cell_np)
            print(f"  step {step:5d}  nn_mean={nn_mean:.4f} A  nn_min={nn_min:.4f} A  nn_std={nn_std:.4f} A")

    print(f"\n=== Phase 2: QUENCH (3000K -> 300K linear ramp, {N_QUENCH_STEPS} steps) ===")
    t_ramp = np.linspace(3000.0, 300.0, N_QUENCH_STEPS)
    for step in range(1, N_QUENCH_STEPS + 1):
        x, v = step_once(model, sigma, cell, x, v, float(t_ramp[step - 1]))
        if step % STRIDE == 0:
            nn_mean, nn_min, nn_std = nn_dist_stats(x.numpy(), cell_np)
            print(f"  step {step:5d}  T={t_ramp[step-1]:.0f}K  nn_mean={nn_mean:.4f} A  "
                  f"nn_min={nn_min:.4f} A  nn_std={nn_std:.4f} A")

    print(f"\n=== Phase 3: HOLD (T=300K held, {N_HOLD_STEPS} steps) ===")
    for step in range(1, N_HOLD_STEPS + 1):
        x, v = step_once(model, sigma, cell, x, v, 300.0)
        if step % STRIDE == 0:
            nn_mean, nn_min, nn_std = nn_dist_stats(x.numpy(), cell_np)
            print(f"  step {step:5d}  nn_mean={nn_mean:.4f} A  nn_min={nn_min:.4f} A  nn_std={nn_std:.4f} A")

    print("\n=== Reference values (from real MD, for comparison) ===")
    print("Real crystal (300K): nn_std ~ 0.014 A (sharp, ordered)")
    print("Real glass (300K, melt-quenched): nn_std ~ 0.08 A (disordered)")


if __name__ == "__main__":
    main()
