"""Trajectory error metrics (ATE, RPE) for camera-to-world pose sequences."""

from __future__ import annotations

import numpy as np


def align_to_first(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Express the estimate in the GT world frame by matching the first pose."""
    return gt[0] @ np.linalg.inv(est[0]) @ est


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> tuple[np.ndarray, float]:
    """Similarity transform (4x4 rigid part, scale) minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / xs.var(0).sum() if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T, s


def ate(est: np.ndarray, gt: np.ndarray, align: str = "first") -> dict:
    """Absolute trajectory error on camera positions.

    align: "first" (anchor at frame 0), "se3" or "sim3" (best-fit over the whole trajectory).
    """
    if align == "first":
        est_w = align_to_first(est, gt)[:, :3, 3]
        scale = 1.0
    else:
        T, scale = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale=align == "sim3")
        est_w = scale * est[:, :3, 3] @ T[:3, :3].T + T[:3, 3]
    err = np.linalg.norm(est_w - gt[:, :3, 3], axis=1)
    return {"rmse": float(np.sqrt((err**2).mean())), "max": float(err.max()), "final": float(err[-1]),
            "scale": float(scale), "errors": err, "aligned": est_w}


def rpe(est: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """Relative pose error over a fixed frame gap: translation (same units) and rotation (deg)."""
    t_err, r_err = [], []
    for i in range(len(gt) - delta):
        d_gt = np.linalg.inv(gt[i]) @ gt[i + delta]
        d_est = np.linalg.inv(est[i]) @ est[i + delta]
        e = np.linalg.inv(d_gt) @ d_est
        t_err.append(np.linalg.norm(e[:3, 3]))
        r_err.append(np.degrees(np.arccos(np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1))))
    t_err, r_err = np.array(t_err), np.array(r_err)
    return {"trans_rmse": float(np.sqrt((t_err**2).mean())), "rot_rmse_deg": float(np.sqrt((r_err**2).mean()))}


def path_length(poses: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())
