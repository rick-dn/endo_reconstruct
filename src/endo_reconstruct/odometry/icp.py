"""Frame-to-frame point-to-plane ICP odometry from depth maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from endo_reconstruct.datasets.simcol import Intrinsics
from endo_reconstruct.geometry import depth_to_pcd


@dataclass
class ICPConfig:
    voxel: float = 0.1  # cm; downsampling before ICP
    normal_radius: float = 0.4  # cm
    max_corr_dist: float = 0.3  # cm; per-frame motion is ~0.1 cm
    max_iter: int = 50
    constant_velocity: bool = True  # initialise with the previous relative motion


@dataclass
class ICPResult:
    poses: np.ndarray  # (N, 4, 4) camera-to-world, frame 0 = identity
    fitness: np.ndarray  # (N-1,) inlier ratio per pair
    rmse: np.ndarray  # (N-1,) inlier RMSE per pair, cm


def _prepare(depth: np.ndarray, k: Intrinsics, cfg: ICPConfig) -> o3d.geometry.PointCloud:
    pcd = depth_to_pcd(depth, k, voxel=cfg.voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.normal_radius, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


def register_pair(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init: np.ndarray,
    cfg: ICPConfig,
) -> o3d.pipelines.registration.RegistrationResult:
    """Transform taking source-frame points into the target frame."""
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        cfg.max_corr_dist,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.max_iter),
    )


def run_odometry(depths, k: Intrinsics, cfg: ICPConfig = ICPConfig(), progress=None) -> ICPResult:
    """Chain frame-to-frame ICP over an iterable of depth maps.

    Each new frame (source) is registered to the previous one (target); the result is the
    relative pose of the new camera in the previous camera's frame.
    """
    depths = iter(depths)
    prev = _prepare(next(depths), k, cfg)
    poses, fitness, rmse = [np.eye(4)], [], []
    rel = np.eye(4)

    for depth in depths:
        cur = _prepare(depth, k, cfg)
        result = register_pair(cur, prev, rel if cfg.constant_velocity else np.eye(4), cfg)
        rel = result.transformation
        poses.append(poses[-1] @ rel)
        fitness.append(result.fitness)
        rmse.append(result.inlier_rmse)
        prev = cur
        if progress:
            progress()

    return ICPResult(np.array(poses), np.array(fitness), np.array(rmse))
