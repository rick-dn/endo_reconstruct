"""Loader for the SimCol3D synthetic colonoscopy dataset.

Dataset layout (one scene folder per anatomy):

    SyntheticColon_III/
        cam.txt
        Frames_O1/FrameBuffer_0000.png, Depth_0000.png, ...
        SavedPosition_O1.txt            tx ty tz (cm), one line per frame
        SavedRotationQuaternion_O1.txt  qx qy qz qw, one line per frame

Conventions used throughout this package:
    - depth is z-depth in centimetres, float32
    - poses are 4x4 camera-to-world matrices in a right-handed frame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Depth PNGs store an 8-bit value in the high byte of a uint16; [0, 1] maps to [0, 20] cm.
DEPTH_MAX_CM = 20.0
DEPTH_SCALE = DEPTH_MAX_CM / (256.0 * 255.0)

# Unity is left-handed; flipping y on both sides converts poses to right-handed.
_FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def read_intrinsics(cam_file: Path, width: int = 475, height: int = 475) -> Intrinsics:
    """Read cam.txt.

    The file reads "227.60416 0 227.60416 0 237.5 237.5 0 0 1", which is not a row-major K:
    taken literally it gives cx=227.6, fy=237.5 (as the official visualisation script does).
    Depth reprojection with GT poses shows the true values are fx=fy=227.6, cx=cy=237.5
    (median error halves at a 20-frame gap), so the fields are read as fx, _, fy, _, cx, cy.
    """
    vals = np.loadtxt(cam_file).ravel()
    return Intrinsics(fx=vals[0], fy=vals[2], cx=vals[4], cy=vals[5], width=width, height=height)


def read_poses(position_file: Path, rotation_file: Path) -> np.ndarray:
    """Read SimCol poses as (N, 4, 4) right-handed camera-to-world matrices."""
    t = np.loadtxt(position_file).reshape(-1, 3)
    q = np.loadtxt(rotation_file).reshape(-1, 4)
    if len(t) != len(q):
        raise ValueError(f"{len(t)} positions but {len(q)} rotations")

    poses = np.tile(np.eye(4), (len(t), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    poses[:, :3, 3] = t
    return _FLIP_Y @ poses @ _FLIP_Y


class SimColSequence:
    """One SimCol trajectory, e.g. SimColSequence("data/simcol3d/SyntheticColon_III", "O1")."""

    def __init__(self, scene_dir: str | Path, trajectory: str):
        self.scene_dir = Path(scene_dir)
        self.trajectory = trajectory
        self.frames_dir = self.scene_dir / f"Frames_{trajectory}"
        if not self.frames_dir.is_dir():
            raise FileNotFoundError(self.frames_dir)

        self.rgb_files = sorted(self.frames_dir.glob("FrameBuffer_*.png"))
        self.depth_files = sorted(self.frames_dir.glob("Depth_*.png"))
        if len(self.rgb_files) != len(self.depth_files):
            raise ValueError(f"{len(self.rgb_files)} RGB frames but {len(self.depth_files)} depth maps")

        self.intrinsics = read_intrinsics(self.scene_dir / "cam.txt")
        self.poses = read_poses(
            self.scene_dir / f"SavedPosition_{trajectory}.txt",
            self.scene_dir / f"SavedRotationQuaternion_{trajectory}.txt",
        )
        if len(self.poses) != len(self):
            raise ValueError(f"{len(self.poses)} poses but {len(self)} frames")

    def __len__(self) -> int:
        return len(self.rgb_files)

    def rgb(self, i: int) -> np.ndarray:
        """RGB image as (H, W, 3) uint8."""
        img = cv2.imread(str(self.rgb_files[i]), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        return (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img

    def depth(self, i: int) -> np.ndarray:
        """Depth as (H, W) float32 in centimetres."""
        raw = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) * DEPTH_SCALE

    def pose(self, i: int) -> np.ndarray:
        """Ground-truth camera-to-world pose (4x4). Use for evaluation only."""
        return self.poses[i]
