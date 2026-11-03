"""Fuse depth maps into a colon point cloud using poses estimated by ICP, and render it.

Also fuses the same frames with GT poses, so the two reconstructions can be compared.

Example:
    python scripts/estimate_poses.py --trajectory O1   # writes outputs/poses/..._O1_icp_gtdepth.npy
    python scripts/reconstruct.py --trajectory O1
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from endo_reconstruct.datasets.simcol import SimColSequence
from endo_reconstruct.evaluation.trajectory import align_to_first
from endo_reconstruct.fusion.pointcloud import fuse_point_cloud
from endo_reconstruct.fusion.tsdf import fuse_tsdf
from endo_reconstruct.visualization.render import principal_view, render_clouds, render_mesh

ICP_COLOR = (0.95, 0.45, 0.1)
GT_COLOR = (0.15, 0.4, 0.85)


def painted(pcd: o3d.geometry.PointCloud, color) -> o3d.geometry.PointCloud:
    out = copy.deepcopy(pcd)
    out.paint_uniform_color(color)
    return out


def trajectory_line(poses: np.ndarray, color) -> o3d.geometry.PointCloud:
    """Camera path as a dense line of points (line sets ignore point size in the renderer)."""
    pos = poses[:, :3, 3]
    pts = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(pos[:-1], pos[1:])])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color(color)
    return pcd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="data/simcol3d/SyntheticColon_III")
    p.add_argument("--trajectory", default="O1")
    p.add_argument("--poses", default=None, help="estimated poses .npy (default: output of estimate_poses.py)")
    p.add_argument("--every", type=int, default=5, help="fuse every Nth frame")
    p.add_argument("--voxel", type=float, default=0.05, help="cm")
    p.add_argument("--tsdf-every", type=int, default=2, help="integrate every Nth frame into the mesh")
    p.add_argument("--no-mesh", action="store_true", help="skip TSDF mesh fusion")
    p.add_argument("--out", default="outputs/reconstruction")
    args = p.parse_args()

    seq = SimColSequence(args.scene, args.trajectory)
    stem = f"{Path(args.scene).name}_{args.trajectory}"
    est = np.load(args.poses or f"outputs/poses/{stem}_icp_gtdepth.npy")
    gt = seq.poses[: len(est)]
    # ICP poses start at identity; put them in the GT world frame (anchored at frame 0 only).
    est = align_to_first(est, gt)

    print("fusing with ICP poses...")
    cloud_icp = fuse_point_cloud(seq, est, every=args.every, voxel=args.voxel)
    print("fusing with GT poses...")
    cloud_gt = fuse_point_cloud(seq, gt, every=args.every, voxel=args.voxel)
    print(f"points: ICP {len(cloud_icp.points):,}  GT {len(cloud_gt.points):,}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out / f"{stem}_icp.ply"), cloud_icp)
    o3d.io.write_point_cloud(str(out / f"{stem}_gt.ply"), cloud_gt)

    pts = np.asarray(cloud_gt.points)
    views = {"front": principal_view(pts), "oblique": principal_view(pts, elevation_deg=35, azimuth_deg=40)}
    path_icp, path_gt = trajectory_line(est, ICP_COLOR), trajectory_line(gt, GT_COLOR)

    panels = []
    for name, view in views.items():
        panels.append((f"ICP poses, RGB texture ({name})", render_clouds([cloud_icp], view)))
        panels.append((f"GT (blue) vs ICP (orange) ({name})",
                       render_clouds([painted(cloud_gt, GT_COLOR), painted(cloud_icp, ICP_COLOR)], view, point_size=1.5)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.suptitle(f"SimCol3D {args.trajectory}: colon reconstructed from depth + ICP poses "
                 f"({len(range(0, len(est), args.every))} frames fused)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_overview.png", dpi=100)
    plt.close(fig)

    # Camera paths alone, over a faint GT cloud, for the trajectory comparison.
    faint = painted(cloud_gt.voxel_down_sample(args.voxel * 4), (0.85, 0.85, 0.85))
    img = render_clouds([faint, path_gt, path_icp], views["front"], point_size=3.0)
    plt.imsave(out / f"{stem}_paths.png", img)

    if not args.no_mesh:
        meshes = {}
        for name, poses in (("ICP poses", est), ("GT poses", gt)):
            print(f"TSDF fusion with {name}...")
            meshes[name] = fuse_tsdf(seq, poses, every=args.tsdf_every)
            tag = "icp" if name == "ICP poses" else "gt"
            o3d.io.write_triangle_mesh(str(out / f"{stem}_{tag}_mesh.ply"), meshes[name])
        print("triangles: " + "  ".join(f"{n} {len(m.triangles):,}" for n, m in meshes.items()))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for row, (vname, view) in enumerate(views.items()):
            for col, (mname, mesh) in enumerate(meshes.items()):
                axes[row, col].imshow(render_mesh(mesh, view))
                axes[row, col].set_title(f"TSDF mesh, {mname} ({vname})", fontsize=13)
                axes[row, col].axis("off")
        fig.suptitle(f"SimCol3D {args.trajectory}: surface reconstructed from depth "
                     f"({len(range(0, len(est), args.tsdf_every))} frames fused)", fontsize=15)
        fig.tight_layout()
        fig.savefig(out / f"{stem}_mesh.png", dpi=100)
        plt.close(fig)

    print(f"saved to {out}/{stem}_*")


if __name__ == "__main__":
    main()
