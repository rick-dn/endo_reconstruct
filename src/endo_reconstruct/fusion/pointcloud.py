"""Fuse per-frame depth maps into a single world-frame point cloud using camera poses."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from endo_reconstruct.datasets.simcol import SimColSequence
from endo_reconstruct.geometry import depth_to_pcd


def fuse_point_cloud(
    seq: SimColSequence,
    poses: np.ndarray,
    every: int = 5,
    stride: int = 2,
    voxel: float = 0.05,
    max_depth: float = 8.0,
) -> o3d.geometry.PointCloud:
    """Back-project every `every`-th frame with its camera-to-world pose and merge.

    max_depth (cm) drops far points, which are sparse and least accurate.
    """
    fused = o3d.geometry.PointCloud()
    for i in range(0, len(poses), every):
        depth = seq.depth(i)
        depth[depth > max_depth] = 0
        pcd = depth_to_pcd(depth, seq.intrinsics, rgb=seq.rgb(i), stride=stride)
        fused += pcd.transform(poses[i])
    return fused.voxel_down_sample(voxel)
