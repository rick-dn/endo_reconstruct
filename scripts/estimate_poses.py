"""Estimate a camera trajectory from depth with frame-to-frame ICP and evaluate it against GT.

Example:
    python scripts/estimate_poses.py --trajectory O1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from endo_reconstruct.datasets.simcol import SimColSequence
from endo_reconstruct.evaluation.trajectory import ate, path_length, rpe
from endo_reconstruct.odometry.icp import ICPConfig, run_odometry


def plot_trajectories(gt: np.ndarray, est: np.ndarray, out: Path, title: str) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*gt.T, label="ground truth", lw=2)
    ax.plot(*est.T, label="ICP (depth only)", lw=1.5)
    ax.scatter(*gt[0], c="k", s=20, label="start")
    ax.set_xlabel("x (cm)"), ax.set_ylabel("y (cm)"), ax.set_zlabel("z (cm)")
    ax.legend()
    ax.set_title(title)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.linalg.norm(est - gt, axis=1) * 10)
    ax2.set_xlabel("frame"), ax2.set_ylabel("position error (mm)")
    ax2.set_title("Absolute position error")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="data/simcol3d/SyntheticColon_III")
    p.add_argument("--trajectory", default="O1")
    p.add_argument("--frames", type=int, default=None, help="use only the first N frames")
    p.add_argument("--voxel", type=float, default=ICPConfig.voxel)
    p.add_argument("--max-corr-dist", type=float, default=ICPConfig.max_corr_dist)
    p.add_argument("--no-constant-velocity", action="store_true")
    p.add_argument("--out", default="outputs/poses")
    args = p.parse_args()

    seq = SimColSequence(args.scene, args.trajectory)
    n = min(args.frames or len(seq), len(seq))
    cfg = ICPConfig(voxel=args.voxel, max_corr_dist=args.max_corr_dist,
                    constant_velocity=not args.no_constant_velocity)

    with tqdm(total=n - 1, desc=f"ICP {args.trajectory}") as bar:
        result = run_odometry((seq.depth(i) for i in range(n)), seq.intrinsics, cfg, progress=bar.update)

    gt = seq.poses[:n]
    a = ate(result.poses, gt, align="first")
    a_se3 = ate(result.poses, gt, align="se3")
    r = rpe(result.poses, gt)
    gt_len = path_length(gt)

    metrics = {
        "trajectory": args.trajectory,
        "frames": n,
        "gt_path_length_cm": gt_len,
        "est_path_length_cm": path_length(result.poses),
        "ate_rmse_cm": a["rmse"],
        "ate_final_cm": a["final"],
        "final_drift_pct": 100 * a["final"] / gt_len,
        "ate_se3_rmse_cm": a_se3["rmse"],
        "rpe_trans_rmse_mm": r["trans_rmse"] * 10,
        "rpe_rot_rmse_deg": r["rot_rmse_deg"],
        "icp_fitness_mean": float(result.fitness.mean()),
        "icp_fitness_min": float(result.fitness.min()),
        "config": vars(cfg),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.scene).name}_{args.trajectory}_icp_gtdepth"
    np.save(out / f"{stem}.npy", result.poses)
    (out / f"{stem}.json").write_text(json.dumps(metrics, indent=2))
    plot_trajectories(gt[:, :3, 3], a["aligned"], out / f"{stem}.png", f"{args.trajectory}: ICP vs GT")

    for key, val in metrics.items():
        if key != "config":
            print(f"{key:22s} {val:.4f}" if isinstance(val, float) else f"{key:22s} {val}")
    print(f"saved to {out}/{stem}.*")


if __name__ == "__main__":
    main()
