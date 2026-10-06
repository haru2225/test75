"""test74: a single fixed-sigma denoising force field trained across
SEVERAL real, genuinely distinct conditions -- crystal (cristobalite,
300 K), melt (3000 K), a melt-quenched glass (300 K), and stishovite (a
high-pressure SiO2 polymorph, 300 K) -- all generated locally (see
make_small_multi_condition_data.py and make_stishovite_data.py). The
network (DenoiserMPNN) has no atom-count-dependent parameters, so one set
of weights can represent all regimes if trained on all of them.

Conditioning is on TEMPERATURE ONLY (not pressure): all datasets so far
were generated via NVT (no barostat), so there is no real, controlled
pressure signal to condition on -- adding a P input would just be an
unused, misleading extra dimension. T is also the physically dominant
driver of which basin (crystal/melt/glass) the system sits in, so
conditioning on T alone is a reasonable, honest simplification given the
data generation method actually used.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parent

# Each dataset: (name, nominal temperature in K for conditioning, relative
# sampling probability). Add a new tuple here once a new condition's
# trajectory + manifest have been generated under data_<name>/.
DATASETS = [
    ("crystal_300K", 300.0, 0.10),
    ("intermediate_1000K", 1000.0, 0.07),
    ("intermediate_1500K", 1500.0, 0.07),
    ("intermediate_2000K", 2000.0, 0.07),
    ("intermediate_2500K", 2500.0, 0.07),
    ("melt_3000K", 3000.0, 0.10),
    ("glass_300K", 300.0, 0.10),
    ("stishovite_300K", 300.0, 0.08),
    # Distorted structures (all relaxed at 300K) -- Erhard et al.-style
    # off-equilibrium coverage, new in test75.
    ("distorted_stretch_p10", 300.0, 0.06),
    ("distorted_compress_m10", 300.0, 0.06),
    ("distorted_shear_aniso", 300.0, 0.06),
    ("distorted_rattle_only", 300.0, 0.06),
    # NPT pressure-varied structures (all at 300K, varying P -- note: P is
    # NOT part of the conditioning input yet, only T; these are included
    # as additional 300K structural diversity for now. Also note the
    # frame-fixed-cell caveat documented in README.md).
    ("npt_cristobalite_0GPa", 300.0, 0.03),
    ("npt_cristobalite_10GPa", 300.0, 0.03),
    ("npt_cristobalite_20GPa", 300.0, 0.03),
    ("npt_stishovite_0GPa", 300.0, 0.03),
    ("npt_stishovite_20GPa", 300.0, 0.03),
    ("npt_stishovite_40GPa", 300.0, 0.03),
]

T_NORM_SCALE = 1000.0  # K

DEVICE = torch.device(os.environ.get("TEST73_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
OUT_DIR = REPO_DIR / "output"

SIGMA_ANGSTROM = 0.15
CUTOFF_ANGSTROM = 5.5  # smallest cell here is the 192-atom crystal/melt/glass
# box (~13.57 A); must stay comfortably under half that.
HIDDEN_DIM = int(os.environ.get("TEST73_HIDDEN_DIM", 128))  # bigger than
# test74's 64 -- test74's melt-quench experiment showed the small network's
# residual error (train loss plateaued ~0.01-0.015, never near zero) is the
# likely bottleneck for quench accuracy, not T-coverage (adding
# intermediate-T data alone did not fix the over-disordering).
N_LAYERS = int(os.environ.get("TEST73_N_LAYERS", 4))
BATCH_SIZE = int(os.environ.get("TEST73_BATCH_SIZE", 16))
N_STEPS = int(os.environ.get("TEST73_N_STEPS", 15000))
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 1.0
SPIKE_ROLLBACK_FACTOR = 3.0


def load_positions_angstrom(name):
    positions_nm = np.load(REPO_DIR / f"data_{name}" / "positions.npy")
    manifest = np.load(REPO_DIR / f"data_{name}" / "manifest.npz")
    cell_angstrom = manifest["cell_nm"].astype(np.float32) * 10.0
    return positions_nm.astype(np.float32) * 10.0, cell_angstrom


def periodic_radius_graph(pos: torch.Tensor, cell: torch.Tensor, cutoff: float):
    diff = pos[None, :, :] - pos[:, None, :]
    diff = diff - cell * torch.round(diff / cell)
    dist = diff.norm(dim=-1)
    mask = (dist < cutoff) & (dist > 1e-6)
    src, dst = torch.nonzero(mask, as_tuple=True)
    edge_vec = diff[src, dst]
    return torch.stack([src, dst], dim=0), edge_vec


class DenoiserMPNN(nn.Module):
    """test74: extends test70-73's DenoiserMPNN with TEMPERATURE
    conditioning. Every atom's initial embedding is h0 +
    condition_mlp([T_norm]) instead of a single fixed learned vector."""

    def __init__(self, hidden_dim: int, n_layers: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.h0 = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.condition_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            ) for _ in range(n_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(n_layers)
        ])
        self.force_head_scalar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, pos: torch.Tensor, cell: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """condition: (B, 1) tensor of T_norm, one value per sample."""
        batch_size, n_atoms, _ = pos.shape
        cond_embed = self.condition_mlp(condition)  # (B, hidden_dim)
        outputs = []
        for b in range(batch_size):
            edge_index, edge_vec = periodic_radius_graph(pos[b], cell, self.cutoff)
            src, dst = edge_index
            edge_len = edge_vec.norm(dim=-1, keepdim=True)
            edge_dir = edge_vec / edge_len.clamp_min(1e-6)

            h = (self.h0 + cond_embed[b]).unsqueeze(0).expand(n_atoms, -1).contiguous()
            for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
                m_input = torch.cat([h[src], h[dst], edge_len], dim=-1)
                m = message_mlp(m_input)
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, m)
                h = h + update_mlp(torch.cat([h, agg], dim=-1))

            m_input = torch.cat([h[src], h[dst], edge_len], dim=-1)
            edge_scalar = self.message_mlps[-1](m_input)
            edge_force_scalar = self.force_head_scalar(edge_scalar)
            per_atom_force = torch.zeros(n_atoms, 3, device=pos.device, dtype=pos.dtype)
            per_atom_force.index_add_(0, dst, edge_force_scalar * edge_dir)
            outputs.append(per_atom_force)
        return torch.stack(outputs, dim=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {}
    probs = []
    cell_by_name = {}
    for name, temp_k, prob in DATASETS:
        pos, cell = load_positions_angstrom(name)
        loaded[name] = (torch.tensor(pos, dtype=torch.float32, device=DEVICE),
                         torch.tensor(cell, dtype=torch.float32, device=DEVICE))
        cell_by_name[name] = cell
        probs.append(prob)
        print(f"{name}: {pos.shape[0]} frames, {pos.shape[1]} atoms, cell {cell}, T={temp_k} K")
    probs = np.array(probs) / np.sum(probs)
    print(f"Using device: {DEVICE}")

    torch.manual_seed(0)
    model = DenoiserMPNN(HIDDEN_DIM, N_LAYERS, CUTOFF_ANGSTROM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    losses_by_kind = {name: [] for name, _, _ in DATASETS}
    n_rollbacks = 0
    last_good_state = None
    t_start = time.time()
    step = 1
    while step <= N_STEPS:
        name, temp_k, _ = DATASETS[np.random.choice(len(DATASETS), p=probs)]
        positions_t, cell_t = loaded[name]
        frame_idx = torch.randint(0, positions_t.shape[0], (BATCH_SIZE,))
        r = positions_t[frame_idx]
        condition = torch.full((BATCH_SIZE, 1), temp_k / T_NORM_SCALE, dtype=torch.float32, device=DEVICE)

        noise = torch.randn_like(r) * SIGMA_ANGSTROM
        r_noisy = r + noise
        target = -noise

        prediction = model(r_noisy, cell_t, condition)
        loss = ((prediction - target) ** 2).mean()
        loss_value = float(loss.detach())

        if len(losses) >= 20:
            baseline = np.mean(losses[-20:])
            if loss_value > SPIKE_ROLLBACK_FACTOR * baseline and last_good_state is not None:
                model.load_state_dict(last_good_state[0])
                optimizer.load_state_dict(last_good_state[1])
                n_rollbacks += 1
                print(f"step {step:5d}: loss {loss_value:.6f} > {SPIKE_ROLLBACK_FACTOR}x "
                      f"baseline {baseline:.6f} -- rolling back (rollback #{n_rollbacks})", flush=True)
                continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        losses.append(loss_value)
        losses_by_kind[name].append(loss_value)
        if step % 20 == 0:
            last_good_state = (
                {k: v.clone() for k, v in model.state_dict().items()},
                optimizer.state_dict(), step,
            )
        if step % 200 == 0 or step == 1:
            summary = "  ".join(
                f"loss_{n}={np.mean(losses_by_kind[n][-50:]) if losses_by_kind[n] else float('nan'):.5f}"
                for n, _, _ in DATASETS
            )
            elapsed = time.time() - t_start
            print(f"step {step:6d}/{N_STEPS}  {summary}  elapsed={elapsed:.1f}s  rollbacks={n_rollbacks}", flush=True)
            torch.save(
                {"model_state": model.state_dict(), "sigma_angstrom": SIGMA_ANGSTROM,
                 "cutoff_angstrom": CUTOFF_ANGSTROM, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
                 "cell_by_name": cell_by_name, "temp_by_name": {n: t for n, t, _ in DATASETS},
                 "t_norm_scale": T_NORM_SCALE, "step": step},
                OUT_DIR / "checkpoint.pt",
            )
        step += 1

    with open(OUT_DIR / "loss_history.json", "w") as f:
        json.dump(losses_by_kind, f)
    print(f"Done. Final checkpoint: {OUT_DIR / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
