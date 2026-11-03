"""Depth back-projection and point-cloud helpers."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from endo_reconstruct.datasets.simcol import Intrinsics


def backproject(depth: np.ndarray, k: Intrinsics, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to camera-frame points.

    Returns (points (N, 3), pixel indices (N,) into the flattened strided image) for depth > 0.
    """
    v, u = np.mgrid[0 : k.height : stride, 0 : k.width : stride]
    z = depth[::stride, ::stride]
    pts = np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], -1).reshape(-1, 3)
    valid = np.flatnonzero(z.reshape(-1) > 0)
    return pts[valid], valid


def to_pcd(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    return pcd


def depth_to_pcd(
    depth: np.ndarray,
    k: Intrinsics,
    rgb: np.ndarray | None = None,
    stride: int = 1,
    voxel: float | None = None,
) -> o3d.geometry.PointCloud:
    """Camera-frame point cloud from a depth map, optionally coloured and voxel-downsampled."""
    pts, idx = backproject(depth, k, stride)
    colors = rgb[::stride, ::stride].reshape(-1, 3)[idx] if rgb is not None else None
    pcd = to_pcd(pts, colors)
    return pcd.voxel_down_sample(voxel) if voxel else pcd
