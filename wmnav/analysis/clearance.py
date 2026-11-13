"""Clearance actually flown: per episode, the minimum distance between the drone
and the nearest obstacle surface along the executed trajectory.

Input: ``trajectories.pt`` written by ``wmnav.evaluate record_trajectories=true``
(or a legacy replay buffer logged with ``log_obstacles``), which stores the
obstacle positions, bounds and target next to every step.

    python -m wmnav.analysis.clearance --buffers WM1_L1=runs/.../trajectories.pt ... [--csv out.csv]

Geometry: the panels (``big_panel.urdf``, 0.1 x 1.2 x 50 m, axis-aligned, taller
than the corridor) make the problem 2-D in (x, y); walls bound the corridor.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

PANEL_HALF_EXTENT_XY = (0.05, 0.6)
N_WALL_ASSETS = 6           # walls are assets 0-5 (keep_in_env assets are loaded first)
PARKED_THRESHOLD = -100.0   # unused asset slots are parked far away
ROBOT_POS_SLICE = slice(13, 16)


def load_trajectories(path):
    """(episode id, steps) pairs from a ``trajectories.pt`` file or a legacy replay buffer."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    if not isinstance(data, dict):
        raise ValueError(f"{path}: unexpected format {type(data)}")
    return data.items()


def episode_arrays(traj, trim=True):
    """(positions, panel centres, bounds_min, bounds_max) of one episode, or None.

    The layout is static within an episode. In PPO mode the env re-randomises
    before the final frame of a finished episode is recorded, so a trailing frame
    with a different layout is trimmed; an episode whose layout changes earlier
    straddles a reset and is skipped.
    """
    required = ("robot_state", "obstacle_position", "env_bounds_min", "env_bounds_max")
    missing = [k for k in required if k not in traj]
    if missing:
        raise KeyError(f"trajectory is missing {missing}; evaluate with record_trajectories=true")
    pos = traj["robot_state"][:, ROBOT_POS_SLICE].numpy().astype(np.float64)
    obst = traj["obstacle_position"].numpy().astype(np.float64)
    bmin = traj["env_bounds_min"].numpy().astype(np.float64)
    bmax = traj["env_bounds_max"].numpy().astype(np.float64)
    if trim and len(obst) > 1:
        drift = np.abs(obst[1:, N_WALL_ASSETS:, :2] - obst[:-1, N_WALL_ASSETS:, :2]).max(axis=(1, 2))
        changes = np.nonzero(drift > 1e-3)[0]
        if len(changes):
            if changes[0] != len(drift) - 1:
                return None
            pos, obst, bmin, bmax = pos[:-1], obst[:-1], bmin[:-1], bmax[:-1]
    if len(pos) == 0:
        return None
    panels = obst[0, N_WALL_ASSETS:, :2]
    panels = panels[panels[:, 0] > PARKED_THRESHOLD]
    return pos, panels, bmin[0], bmax[0]


def clearance_analytic(pos_xy, panels_xy, bmin, bmax):
    """Exact distance from each point to the nearest panel or wall."""
    hx, hy = PANEL_HALF_EXTENT_XY
    if len(panels_xy):
        dx = np.maximum(np.abs(pos_xy[:, None, 0] - panels_xy[None, :, 0]) - hx, 0.0)
        dy = np.maximum(np.abs(pos_xy[:, None, 1] - panels_xy[None, :, 1]) - hy, 0.0)
        d_panels = np.hypot(dx, dy).min(axis=1)
    else:
        d_panels = np.full(len(pos_xy), np.inf)
    d_walls = np.minimum(np.minimum(pos_xy[:, 0] - bmin[0], bmax[0] - pos_xy[:, 0]),
                         np.minimum(pos_xy[:, 1] - bmin[1], bmax[1] - pos_xy[:, 1]))
    return np.minimum(d_panels, d_walls)


def clearance_grid(pos_xy, panels_xy, bmin, bmax, res):
    """The same via an occupancy grid and a Euclidean distance transform."""
    from scipy.ndimage import distance_transform_edt
    hx, hy = PANEL_HALF_EXTENT_XY
    nx = max(int(math.ceil((bmax[0] - bmin[0]) / res)), 1)
    ny = max(int(math.ceil((bmax[1] - bmin[1]) / res)), 1)
    gx, gy = np.meshgrid(bmin[0] + (np.arange(nx) + 0.5) * res, bmin[1] + (np.arange(ny) + 0.5) * res, indexing="ij")
    occ = np.zeros((nx, ny), dtype=bool)
    for cx, cy in panels_xy:
        occ |= (np.abs(gx - cx) <= hx) & (np.abs(gy - cy) <= hy)
    padded = np.zeros((nx + 2, ny + 2), dtype=bool)  # border left occupied: walls bound the field
    padded[1:-1, 1:-1] = ~occ
    field = distance_transform_edt(padded)[1:-1, 1:-1] * res
    ix = np.clip(((pos_xy[:, 0] - bmin[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((pos_xy[:, 1] - bmin[1]) / res).astype(int), 0, ny - 1)
    return field[ix, iy]


def summarize(mins, threshold):
    """Distribution of per-episode minimum clearances (m): mean, median, low percentiles and
    the share of episodes that came closer than ``threshold`` and than 1 m.
    """
    mins = np.asarray(mins, dtype=np.float64)
    return {
        "episodes": len(mins), "mean": float(np.mean(mins)), "median": float(np.median(mins)),
        "p05": float(np.percentile(mins, 5)), "p10": float(np.percentile(mins, 10)), "min": float(np.min(mins)),
        f"frac_below_{threshold:g}": float(np.mean(mins < threshold)), "frac_below_1m": float(np.mean(mins < 1.0)),
    }


def main():
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffers", nargs="+", required=True, help="LABEL=path/to/trajectories.pt")
    ap.add_argument("--threshold", type=float, default=0.67, help="clearance to compare against [m]")
    ap.add_argument("--grid", action="store_true", help="rasterised distance field instead of exact boxes")
    ap.add_argument("--res", type=float, default=0.05, help="grid resolution [m] for --grid")
    ap.add_argument("--no-trim", action="store_true", help="keep frames whose layout does not match the episode")
    ap.add_argument("--csv", type=Path, help="per-episode minima")
    args = ap.parse_args()

    rows, table = [], {}
    for spec in args.buffers:
        label, path = spec.split("=", 1)
        mins, skipped = [], 0
        for _id, traj in load_trajectories(path):
            ep = episode_arrays(traj, trim=not args.no_trim)
            if ep is None:
                skipped += 1
                continue
            pos, panels, bmin, bmax = ep
            c = (clearance_grid(pos[:, :2], panels, bmin, bmax, args.res) if args.grid
                 else clearance_analytic(pos[:, :2], panels, bmin, bmax))
            mins.append(float(c.min()))
            rows.append((label, _id, len(pos), float(c.min()), float(np.median(c))))
        if mins:
            table[label] = dict(summarize(mins, args.threshold), skipped=skipped)

    if table:
        cols = list(next(iter(table.values())))
        print(f"{'run':<12}" + "".join(f"{c:>16}" for c in cols))
        for label, stats in table.items():
            cells = (f"{stats[c]:>16.3f}" if isinstance(stats[c], float) else f"{stats[c]:>16}" for c in cols)
            print(f"{label:<12}" + "".join(cells))
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "episode_id", "steps", "min_clearance", "median_clearance"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
