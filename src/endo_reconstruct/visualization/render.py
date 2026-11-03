"""Headless rendering of point clouds and meshes to images with Open3D's offscreen renderer.

Open3D 0.20's Filament renderer is flaky offscreen: it silently returns black images above
~250k points per scene, and occasionally returns a frame with large pure-black regions that
re-rendering does not fix. Scenes are therefore capped in size, and a frame with black regions
is rebuilt in a fresh renderer (the background is white, so pure black never occurs legitimately).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

MAX_RENDER_POINTS = 200_000
MAX_RENDER_TRIANGLES = 200_000
_MAX_ATTEMPTS = 5
_MAX_BLACK_FRACTION = 0.005

View = tuple[np.ndarray, np.ndarray, np.ndarray]  # (center, eye, up)


def principal_view(points: np.ndarray, elevation_deg: float = 0.0, azimuth_deg: float = 0.0) -> View:
    """Camera looking at the cloud's broad side: along its least-variance axis.

    Elevation/azimuth rotate the eye around the center.
    """
    center = points.mean(0)
    _, _, Vt = np.linalg.svd(points - center, full_matrices=False)
    long_axis, mid_axis, short_axis = Vt
    extent = np.ptp((points - center) @ Vt.T, axis=0).max()

    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.cos(el) * (np.cos(az) * short_axis + np.sin(az) * long_axis) + np.sin(el) * mid_axis
    eye = center + direction * extent * 1.3
    return center, eye, mid_axis


_renderers: dict[tuple[int, int], rendering.OffscreenRenderer] = {}


def _renderer(size: tuple[int, int], fresh: bool = False) -> rendering.OffscreenRenderer:
    if fresh or size not in _renderers:
        r = rendering.OffscreenRenderer(*size)
        r.scene.view.set_post_processing(False)  # tone mapping washes out unlit colours
        _renderers[size] = r
    return _renderers[size]


def _capture(renderer: rendering.OffscreenRenderer, max_frames: int = 20) -> np.ndarray:
    """Render until two consecutive frames match (the first frames after a scene change can be
    black or partially drawn). Returns a copy: the image buffer belongs to the renderer."""
    prev = np.array(renderer.render_to_image())
    for _ in range(max_frames):
        img = np.array(renderer.render_to_image())
        if np.array_equal(img, prev):
            return img
        prev = img
    return prev


def _render(
    build: Callable[[rendering.Open3DScene], None],
    view: View,
    size: tuple[int, int],
    fov: float,
    background,
) -> np.ndarray:
    """Build a scene with `build`, render it, and retry in a fresh renderer if the frame is corrupt."""
    center, eye, up = view
    img = None
    for attempt in range(_MAX_ATTEMPTS):
        renderer = _renderer(size, fresh=attempt > 0)
        renderer.scene.clear_geometry()
        renderer.scene.set_background(np.array(background))
        build(renderer.scene)
        renderer.setup_camera(fov, center, eye, up)
        img = _capture(renderer)
        if (img.sum(-1) == 0).mean() < _MAX_BLACK_FRACTION:
            return img
    return img


def render_clouds(
    clouds: list[o3d.geometry.PointCloud],
    view: View,
    size: tuple[int, int] = (1200, 900),
    point_size: float = 2.0,
    fov: float = 50.0,
    background=(1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Render point clouds (using their own colours). Returns RGB uint8."""
    total = sum(len(c.points) for c in clouds)
    keep = min(1.0, MAX_RENDER_POINTS / max(total, 1))
    clouds = [c.random_down_sample(keep) if keep < 1 else c for c in clouds]

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    def build(scene):
        for i, cloud in enumerate(clouds):
            scene.add_geometry(f"cloud{i}", cloud, mat)

    return _render(build, view, size, fov, background)


def render_mesh(
    mesh: o3d.geometry.TriangleMesh,
    view: View,
    size: tuple[int, int] = (1200, 900),
    fov: float = 50.0,
    background=(1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Render a vertex-coloured mesh with a light from the camera direction. Returns RGB uint8."""
    if len(mesh.triangles) > MAX_RENDER_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(MAX_RENDER_TRIANGLES)
    else:
        mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.compute_vertex_normals()

    # TSDF normals face the cameras that observed the surface (inside the colon); when viewed from
    # outside that leaves the visible side unlit, so turn every normal towards the viewer.
    center, eye, _ = view
    v, n = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    flip = np.einsum("ij,ij->i", n, eye - v) < 0
    n[flip] *= -1
    mesh.vertex_normals = o3d.utility.Vector3dVector(n)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    def build(scene):
        scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, center - eye)
        scene.add_geometry("mesh", mesh, mat)

    return _render(build, view, size, fov, background)
