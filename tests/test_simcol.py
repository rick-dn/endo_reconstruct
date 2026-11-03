from pathlib import Path

import numpy as np
import pytest

from endo_reconstruct.datasets.simcol import SimColSequence

SCENE = Path(__file__).parents[1] / "data/simcol3d/SyntheticColon_III"
pytestmark = pytest.mark.skipif(not SCENE.is_dir(), reason="SimCol3D Colon III not downloaded")


@pytest.fixture(scope="module")
def seq():
    return SimColSequence(SCENE, "O1")


def test_counts(seq):
    assert len(seq) == 601
    assert seq.poses.shape == (601, 4, 4)


def test_frame_formats(seq):
    rgb, depth = seq.rgb(0), seq.depth(0)
    assert rgb.shape == (475, 475, 3) and rgb.dtype == np.uint8
    assert depth.shape == (475, 475) and depth.dtype == np.float32
    assert 0 < depth.max() <= 20.0


def test_intrinsics(seq):
    k = seq.intrinsics
    assert k.fx == pytest.approx(227.60416) and k.fy == pytest.approx(227.60416)
    assert k.cx == pytest.approx(237.5) and k.cy == pytest.approx(237.5)


def test_poses_are_rigid(seq):
    R = seq.poses[:, :3, :3]
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), R.shape), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_gt_depth_reprojects_consistently(seq):
    """Warp frame 100 into frame 120 with GT poses; depths should agree to below a millimetre."""
    k = seq.intrinsics
    i, j = 100, 120
    v, u = np.mgrid[: k.height, : k.width]
    z = seq.depth(i)
    p = np.stack([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z], -1).reshape(-1, 3)
    T = np.linalg.inv(seq.pose(j)) @ seq.pose(i)
    p = p @ T[:3, :3].T + T[:3, 3]
    uj = np.round(p[:, 0] / p[:, 2] * k.fx + k.cx).astype(int)
    vj = np.round(p[:, 1] / p[:, 2] * k.fy + k.cy).astype(int)
    ok = (p[:, 2] > 0) & (uj >= 0) & (uj < k.width) & (vj >= 0) & (vj < k.height)
    err_cm = np.abs(seq.depth(j)[vj[ok], uj[ok]] - p[ok, 2])
    assert ok.mean() > 0.2
    assert np.median(err_cm) < 0.05
