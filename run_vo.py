"""
Run Panoramic Visual Odometry on the generated dataset.

Usage:
    python run_vo.py
    python run_vo.py --backend orb       # use ORB (no GPU needed)
    python run_vo.py --backend superpoint # use SuperPoint features
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.features import FeatureExtractor
from pano_track.matching import FeatureMatcher
from pano_track.vo import PanoramicVO, run_vo_on_dataset
from pano_track.visualize import plot_vo_trajectory, equirect_to_perspective_crop


def umeyama_alignment(est, gt):
    """
    Find optimal similarity transform (scale + rotation + translation)
    aligning estimated trajectory to ground truth.

    Standard evaluation protocol for monocular VO/SLAM.
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    # Center both trajectories
    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    est_centered = est - est_mean
    gt_centered = gt - gt_mean

    # Optimal rotation via orthogonal Procrustes
    # Find R such that ||gt_centered - s * est_centered @ R|| is minimized
    R, scale = orthogonal_procrustes(est_centered, gt_centered)
    scale = 1.0 / scale  # orthogonal_procrustes returns s where est * s ≈ gt

    # Compute optimal scale explicitly
    # scale = trace(gt_centered.T @ est_centered @ R) / trace(est_centered.T @ est_centered)
    num = np.trace(gt_centered.T @ (est_centered @ R))
    den = np.trace(est_centered.T @ est_centered)
    scale = num / max(den, 1e-10)

    # Apply similarity transform
    aligned = scale * (est_centered @ R) + gt_mean

    return aligned, scale, R


def main():
    parser = argparse.ArgumentParser(description="Run Panoramic Visual Odometry")
    parser.add_argument("--data-dir", type=str, default="data/proc_scene",
                        help="Path to generated dataset")
    parser.add_argument("--backend", type=str, default="orb",
                        choices=["orb", "superpoint"],
                        help="Feature extraction backend")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────
    meta_path = os.path.join(args.data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}. Run generate_data.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Dataset: {metadata['scene']}")
    print(f"VO frames: {metadata['vo_frames']}")
    print(f"Resolution: {metadata['image_resolution']}")
    print(f"Backend: {args.backend}")

    # ── Load images and GT poses ──────────────────────────
    images = []
    gt_positions = []
    for frame in metadata["vo_trajectory"]:
        img_path = os.path.join(args.data_dir, frame["filename"])
        img = np.array(Image.open(img_path))
        images.append(img)
        gt_positions.append(frame["position"])

    gt_positions = np.array(gt_positions, dtype=np.float32)

    # ── Run VO ────────────────────────────────────────────
    print(f"\nRunning panoramic VO on {len(images)} frames...")
    t0 = time.time()

    trajectory, errors = run_vo_on_dataset(
        images, gt_positions,
        feature_backend=args.backend,
        device=args.device,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(images):.2f}s/frame)")

    # ── Umeyama alignment (standard monocular VO protocol) ─
    trajectory_aligned, sim_scale, sim_R = umeyama_alignment(trajectory, gt_positions)
    pos_errors = np.linalg.norm(trajectory_aligned - gt_positions, axis=-1)
    print(f"\nUmeyama alignment: scale={sim_scale:.3f}x")
    print(f"Mean position error: {np.mean(pos_errors):.2f}m")
    print(f"Median position error: {np.median(pos_errors):.2f}m")
    print(f"RMSE: {np.sqrt(np.mean(pos_errors**2)):.2f}m")

    # ── Summary statistics ────────────────────────────────
    if errors:
        avg_matches = np.mean([e["n_matches"] for e in errors])
        avg_inliers = np.mean([e["n_inliers"] for e in errors])
        print(f"Avg matches/frame: {avg_matches:.0f}")
        print(f"Avg inliers/frame: {avg_inliers:.0f}")

    # ── Save trajectory ───────────────────────────────────
    np.savetxt(os.path.join(args.output, "vo_trajectory_est.txt"), trajectory_aligned,
               header="x y z (estimated, similarity-aligned to GT)")
    np.savetxt(os.path.join(args.output, "vo_trajectory_gt.txt"), gt_positions,
               header="x y z (ground truth)")
    print(f"\nTrajectories saved to {args.output}/")

    # ── Visualize ─────────────────────────────────────────
    plot_vo_trajectory(
        gt_positions, trajectory_aligned,
        title=f"Panoramic VO — {args.backend.upper()} Features (Median err: {np.median(pos_errors):.2f}m)",
        save_path=os.path.join(args.output, "vo_trajectory.png"),
    )

    # Save sample perspective crops
    for i in [0, 25, -1]:
        crop = equirect_to_perspective_crop(images[i], fov_deg=100,
                                             bearing_deg=(-30 if i == -1 else 0),
                                             out_size=(400, 400))
        Image.fromarray(crop).save(os.path.join(args.output, f"view_frame_{i}.png"))

    print("Done!")


if __name__ == "__main__":
    main()
