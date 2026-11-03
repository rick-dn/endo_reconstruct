"""TSDF fusion of depth maps into a surface mesh.

Uses Open3D's UniformTSDFVolume: in Open3D 0.20 the hashed volumes (ScalableTSDFVolume and the
tensor VoxelBlockGrid) return empty meshes even for a flat plane, so the volume is a fixed cube
sized from a coarse point-cloud fusion of the same frames.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from endo_reconstruct.datasets.simcol import SimColSequence
from endo_reconstruct.fusion.pointcloud import fuse_point_cloud


def fuse_tsdf(
    seq: SimColSequence,
    poses: np.ndarray,
    every: int = 2,
    voxel: float = 0.05,
    trunc: float = 0.25,
    max_depth: float = 8.0,
    max_resolution: int = 512,
) -> o3d.geometry.TriangleMesh:
    """Integrate every `every`-th frame into a TSDF volume and extract a coloured mesh.

    Units are cm: voxel is the target TSDF cell size (coarsened if the cube would exceed
    max_resolution cells per side), trunc the truncation distance.
    """
    coarse = fuse_point_cloud(seq, poses, every=20, stride=8, voxel=0.2, max_depth=max_depth)
    bbox = coarse.get_axis_aligned_bounding_box()
    margin = 2 * trunc
    length = float(bbox.get_extent().max()) + 2 * margin
    resolution = min(max_resolution, int(np.ceil(length / voxel)))
    origin = (bbox.get_center() - length / 2).reshape(3, 1)

    k = seq.intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(k.width, k.height, k.fx, k.fy, k.cx, k.cy)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length,
        resolution=resolution,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=origin,
    )
    for i in range(0, len(poses), every):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(seq.rgb(i))),
            o3d.geometry.Image(seq.depth(i)),
            depth_scale=1.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh
