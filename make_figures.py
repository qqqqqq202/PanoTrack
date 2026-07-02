"""
Generate publication-quality visual figures for PanoTrack.

Creates:
  1. feature_matches.png  — what the algorithm "sees" between two frames
  2. homing_visual.png    — home snapshot vs query, aligned, with bearing estimate
  3. trajectory_views.png — VO trajectory + first-person perspective views
  4. summary.png          — all-in-one comprehensive figure
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.camera import erp_to_sphere
from pano_track.features import FeatureExtractor
from pano_track.matching import FeatureMatcher, draw_matches
from pano_track.homing import VisualHoming
from pano_track.pose import estimate_essential_ransac
from pano_track.visualize import equirect_to_perspective_crop, _draw_scene_layout
from pano_track.vo import run_vo_on_dataset

# ── Color palette ─────────────────────────────────────────
C = {
    "gt":       "#2ecc71",
    "est":      "#e74c3c",
    "home":     "#3498db",
    "query":    "#f39c12",
    "inlier":   "#2ecc71",
    "outlier":  "#e74c3c",
    "match":    "#9b59b6",
    "bg":       "#1a1a2e",
    "panel":    "#16213e",
    "text":     "#ecf0f1",
    "accent":   "#0f3460",
}


def load_data(data_dir="data/proc_scene_v2"):
    """Load dataset metadata and sample frames."""
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    # Load VO frames
    vo_imgs, vo_gt = [], []
    for frame in meta["vo_trajectory"]:
        img = np.array(Image.open(os.path.join(data_dir, frame["filename"])))
        vo_imgs.append(img)
        vo_gt.append(frame["position"])
    vo_gt = np.array(vo_gt, dtype=np.float32)

    # Load homing frames
    hom_imgs, hom_pos = [], []
    for view in meta["homing_views"]:
        img = np.array(Image.open(os.path.join(data_dir, view["filename"])))
        hom_imgs.append(img)
        hom_pos.append(view["position"])
    hom_pos = np.array(hom_pos, dtype=np.float32)

    return meta, vo_imgs, vo_gt, hom_imgs, hom_pos


def make_feature_matches(vo_imgs, meta, save_path="results/feature_matches.png"):
    """Figure 1: Show what the algorithm 'sees' when matching two frames."""
    W, H = meta["image_resolution"]

    # Pick two frames with visible difference (mid-corridor)
    frame_a = vo_imgs[10]
    frame_b = vo_imgs[11]
    gt_a = np.array(meta["vo_trajectory"][10]["position"])
    gt_b = np.array(meta["vo_trajectory"][11]["position"])
    gt_dist = np.linalg.norm(gt_b - gt_a)

    # Extract + match
    extractor = FeatureExtractor(backend="orb", max_keypoints=1024)
    kpts_a, descs_a = extractor.extract(frame_a)
    kpts_b, descs_b = extractor.extract(frame_b)
    matcher = FeatureMatcher(backend="flann", ratio_thresh=0.75)

    # Get raw matches first (all), then spherical verification separately
    raw_matches = matcher._match_flann(kpts_a, descs_a, kpts_b, descs_b)
    raw_a = np.array([kpts_a[m[0]] for m in raw_matches], dtype=np.float32)
    raw_b = np.array([kpts_b[m[1]] for m in raw_matches], dtype=np.float32)

    # Spherical verification
    if len(raw_a) >= 8:
        E, inlier_mask, n_inl = estimate_essential_ransac(raw_a, raw_b, W, H, n_iterations=500, threshold=0.005)
    else:
        inlier_mask = np.ones(len(raw_a), dtype=bool)

    # Build drawn matches for inliers only (cleaner visualization)
    match_a_in = raw_a[inlier_mask]
    match_b_in = raw_b[inlier_mask]

    max_show = 80
    if len(match_a_in) > max_show:
        idx = np.random.choice(len(match_a_in), max_show, replace=False)
        match_a_in = match_a_in[idx]
        match_b_in = match_b_in[idx]

    # ── Draw figure ──
    fig = plt.figure(figsize=(20, 8))

    # --- Top: equirectangular side-by-side with matches ---
    h_img, w_img = frame_a.shape[:2]

    # Resize for display (too wide at 512px)
    scale = 2
    frame_a_big = np.array(Image.fromarray(frame_a).resize((w_img * scale, h_img * scale), Image.NEAREST))
    frame_b_big = np.array(Image.fromarray(frame_b).resize((w_img * scale, h_img * scale), Image.NEAREST))

    canvas = np.zeros((h_img * scale, w_img * scale * 2 + 20, 3), dtype=np.uint8)
    canvas[:, :w_img * scale] = frame_a_big
    canvas[:, w_img * scale + 20:] = frame_b_big

    # Draw match lines
    n_draw = min(60, len(match_a_in))
    indices = np.random.choice(len(match_a_in), n_draw, replace=False) if len(match_a_in) > n_draw else np.arange(len(match_a_in))

    colors = plt.cm.viridis(np.linspace(0, 1, n_draw))
    for i, idx in enumerate(indices):
        x1, y1 = match_a_in[idx, 0] * scale, match_a_in[idx, 1] * scale
        x2, y2 = match_b_in[idx, 0] * scale + w_img * scale + 20, match_b_in[idx, 1] * scale
        color = colors[i][:3]
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=0.8, alpha=0.7)
        plt.plot(x1, y1, 'o', color=color, markersize=2, alpha=0.7)
        plt.plot(x2, y2, 'o', color=color, markersize=2, alpha=0.7)

    plt.imshow(canvas)
    plt.title(f"Feature Matches: Frame 10 → Frame 11  |  {len(raw_a)} raw matches, {inlier_mask.sum()} spherical inliers  |  Step: {gt_dist:.2f}m",
              fontsize=13, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


def make_homing_visual(hom_imgs, hom_pos, meta, home_idx=8, query_idx=5,
                        save_path="results/homing_visual.png"):
    """Figure 2: Visual homing — aligned images + dissimilarity + bearing."""
    W, H = meta["image_resolution"]

    home_img = hom_imgs[home_idx]
    home_pos_arr = hom_pos[home_idx]
    query_img = hom_imgs[query_idx]
    query_pos_arr = hom_pos[query_idx]

    # Run homing
    homing = VisualHoming(W, H, n_azimuth_bins=360)
    homing.set_home(home_img, home_pos_arr)
    bearing_deg, conf, rot_deg, dissim = homing.estimate_home_bearing(query_img)

    # True bearing
    true_vec = home_pos_arr[[0, 2]] - query_pos_arr[[0, 2]]
    true_vec = true_vec / (np.linalg.norm(true_vec) + 1e-10)
    true_bearing = np.rad2deg(np.arctan2(true_vec[1], true_vec[0]))

    home_bin = np.argmin(dissim)

    # ── Create figure ──
    fig = plt.figure(figsize=(18, 10))

    # --- Row 1: Perspective views of home and query ---
    ax_home = fig.add_subplot(2, 3, 1)
    crop_home = equirect_to_perspective_crop(home_img, fov_deg=100, out_size=(300, 300))
    ax_home.imshow(crop_home)
    ax_home.set_title(f"HOME\n({home_pos_arr[0]:.1f}, {home_pos_arr[2]:.1f})",
                      fontsize=12, fontweight="bold", color=C["home"])
    ax_home.axis("off")

    ax_query = fig.add_subplot(2, 3, 2)
    crop_query = equirect_to_perspective_crop(query_img, fov_deg=100, out_size=(300, 300))
    ax_query.imshow(crop_query)
    ax_query.set_title(f"QUERY\n({query_pos_arr[0]:.1f}, {query_pos_arr[2]:.1f})",
                       fontsize=12, fontweight="bold", color=C["query"])
    ax_query.axis("off")

    # --- Row 1, Col 3: Top-down view ---
    ax_topo = fig.add_subplot(2, 3, 3)
    _draw_scene_layout(ax_topo)
    ax_topo.scatter(*home_pos_arr[[0, 2]], c=C["home"], s=200, marker="H",
                    edgecolors="white", linewidth=2, zorder=10, label="Home")
    ax_topo.scatter(*query_pos_arr[[0, 2]], c=C["query"], s=100, marker="o",
                    edgecolors="white", linewidth=1.5, zorder=10, label="Query")

    # Draw estimated bearing arrow
    e_rad = np.deg2rad(bearing_deg)
    e_vec = np.array([np.cos(e_rad), -np.sin(e_rad)]) * 2.0
    ax_topo.arrow(query_pos_arr[0], query_pos_arr[2], e_vec[0], e_vec[1],
                  head_width=0.3, head_length=0.3, fc=C["est"], ec=C["est"],
                  alpha=0.8, width=0.08, label=f"Estimated: {bearing_deg:.0f}°")

    # Draw true bearing arrow
    t_rad = np.deg2rad(true_bearing)
    t_vec = np.array([np.cos(t_rad), -np.sin(t_rad)]) * 2.0
    ax_topo.arrow(query_pos_arr[0], query_pos_arr[2], t_vec[0], t_vec[1],
                  head_width=0.3, head_length=0.3, fc=C["gt"], ec=C["gt"],
                  alpha=0.8, width=0.08, linestyle="--", label=f"True: {true_bearing:.0f}°")

    dist = np.linalg.norm(home_pos_arr[[0, 2]] - query_pos_arr[[0, 2]])
    ax_topo.set_title(f"Home Bearing  |  Distance: {dist:.1f}m  |  Error: {abs(bearing_deg-true_bearing):.0f}°",
                      fontsize=12, fontweight="bold")
    ax_topo.legend(fontsize=9, loc="upper right")
    ax_topo.set_xlabel("X (m)")
    ax_topo.set_ylabel("Z (m)")
    ax_topo.set_aspect("equal")
    ax_topo.grid(True, alpha=0.3)

    # --- Row 2: Dissimilarity profile ---
    ax_dissim = fig.add_subplot(2, 3, (4, 6))
    bearings = np.linspace(-180, 180, len(dissim))
    ax_dissim.fill_between(bearings, 0, dissim, color=C["est"], alpha=0.15)
    ax_dissim.plot(bearings, dissim, color=C["est"], linewidth=2)
    ax_dissim.axvline(x=bearing_deg, color=C["est"], linestyle="-", linewidth=2.5,
                      label=f"Estimated: {bearing_deg:.1f}°")
    ax_dissim.axvline(x=true_bearing, color=C["gt"], linestyle="--", linewidth=2.5,
                      label=f"True: {true_bearing:.1f}°")

    # Shade the region around the estimated bearing
    ax_dissim.axvspan(bearing_deg - 10, bearing_deg + 10, color=C["est"], alpha=0.1)
    ax_dissim.set_xlabel("Bearing (degrees)", fontsize=12)
    ax_dissim.set_ylabel("Dissimilarity", fontsize=12)
    ax_dissim.set_title(
        "Panoramic Dissimilarity Profile\n"
        f"Min dissimilarity at {bearing_deg:.1f}° → estimated home direction  |  "
        f"Confidence: {conf:.0f}  |  Rotation align: {rot_deg:.1f}°",
        fontsize=12, fontweight="bold")
    ax_dissim.legend(fontsize=10, loc="upper right")
    ax_dissim.grid(True, alpha=0.3)
    ax_dissim.set_xlim(-180, 180)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


def make_trajectory_views(vo_imgs, vo_gt, meta, save_path="results/trajectory_views.png"):
    """Figure 3: VO trajectory with embedded first-person perspective snapshots."""
    W, H = meta["image_resolution"]

    # Run VO
    print("  Running VO for trajectory figure...")
    trajectory, errors = run_vo_on_dataset(vo_imgs, vo_gt, feature_backend="orb")

    # Umeyama alignment
    from run_vo import umeyama_alignment
    trajectory_aligned, scale, R = umeyama_alignment(trajectory, vo_gt)

    # ── Create figure ──
    fig = plt.figure(figsize=(20, 12))

    # --- Main: Trajectory (top-down) ---
    ax_traj = fig.add_axes([0.05, 0.35, 0.55, 0.60])
    _draw_scene_layout(ax_traj)

    gt_xz = vo_gt[:, [0, 2]]
    est_xz = trajectory_aligned[:, [0, 2]]

    ax_traj.plot(gt_xz[:, 0], gt_xz[:, 1], "o-", color=C["gt"], linewidth=3,
                 markersize=4, label="Ground Truth", alpha=0.9, zorder=3)
    ax_traj.plot(est_xz[:, 0], est_xz[:, 1], "s--", color=C["est"], linewidth=2.5,
                 markersize=4, label="Estimated (VO)", alpha=0.9, zorder=3)

    # Start and end markers
    ax_traj.scatter(*gt_xz[0], c=C["gt"], s=200, marker="o", edgecolors="white",
                    linewidth=2, zorder=10)
    ax_traj.scatter(*gt_xz[-1], c="white", s=150, marker="*", edgecolors=C["gt"],
                    linewidth=2, zorder=10)

    # Add arrow showing direction at key points
    for i in [0, 15, 30, 45]:
        ax_traj.annotate("", xy=gt_xz[min(i+2, 49)], xytext=gt_xz[i],
                         arrowprops=dict(arrowstyle="->", color=C["gt"], lw=2, alpha=0.5))

    pos_errs = np.linalg.norm(trajectory_aligned - vo_gt, axis=-1)
    ax_traj.set_title(
        f"Panoramic Visual Odometry  |  "
        f"Median error: {np.median(pos_errs):.2f}m  |  "
        f"RMSE: {np.sqrt(np.mean(pos_errs**2)):.2f}m  |  "
        f"Path: {np.sum(np.linalg.norm(np.diff(vo_gt, axis=0), axis=-1)):.0f}m",
        fontsize=13, fontweight="bold")
    ax_traj.set_xlabel("X (meters)", fontsize=11)
    ax_traj.set_ylabel("Z (meters)", fontsize=11)
    ax_traj.legend(fontsize=10, loc="upper left")
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)

    # --- Bottom: Perspective snapshots along the path ---
    key_frames = [0, 10, 20, 30, 40, 49]
    key_labels = ["Entrance", "Mid-corridor", "Corridor end", "Entering room", "Room center", "Room back"]
    key_bearings = [0, 0, 0, 30, 45, -30]

    for j, (fi, label, bear) in enumerate(zip(key_frames, key_labels, key_bearings)):
        ax_view = fig.add_axes([0.05 + j * 0.155, 0.05, 0.14, 0.25])
        crop = equirect_to_perspective_crop(vo_imgs[fi], fov_deg=90,
                                             bearing_deg=bear, out_size=(200, 200))
        ax_view.imshow(crop)
        ax_view.set_title(f"Frame {fi}: {label}", fontsize=9, fontweight="bold", pad=2)
        ax_view.axis("off")

        # Draw a border
        for spine in ax_view.spines.values():
            spine.set_edgecolor(C["est"] if j >= 3 else C["gt"])
            spine.set_linewidth(2)

    # --- Right side: Info panel ---
    ax_info = fig.add_axes([0.65, 0.35, 0.32, 0.60])
    ax_info.axis("off")

    info_text = (
        "HOW IT WORKS\n"
        "━━━━━━━━━━━━━━━━\n\n"
        "1. FEATURE EXTRACTION\n"
        "   ORB detector finds keypoints in\n"
        "   each equirectangular frame\n\n"
        "2. FEATURE MATCHING\n"
        "   Correspondences between consecutive\n"
        "   frames via Hamming distance +\n"
        "   Lowe's ratio test\n\n"
        "3. SPHERICAL GEOMETRY\n"
        "   Matches → unit sphere bearings\n"
        "   8-point algorithm on S² → E\n"
        "   RANSAC filters outliers\n\n"
        "4. POSE RECOVERY\n"
        "   SVD of E → 4 candidate (R,t)\n"
        "   Chirality check → correct solution\n"
        "   Accumulate → trajectory\n\n"
        "KEY INSIGHT\n"
        "   Panoramic cameras are rotation-\n"
        "   invariant: turning the camera\n"
        "   shifts the image horizontally.\n"
        "   Features never leave the FOV."
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.9))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


def make_summary(save_path="results/summary.png"):
    """Figure 4: The comprehensive 'money shot' — what goes on the GitHub README."""
    # Check that sub-figures exist
    figs = [
        "results/feature_matches.png",
        "results/homing_visual.png",
        "results/trajectory_views.png",
    ]

    if not all(os.path.exists(f) for f in figs):
        print("Sub-figures not all ready. Run individual make_* functions first.")
        return

    img1 = plt.imread(figs[0])
    img2 = plt.imread(figs[1])
    img3 = plt.imread(figs[2])

    fig = plt.figure(figsize=(24, 28))

    # Title banner
    ax_title = fig.add_axes([0.05, 0.96, 0.90, 0.03])
    ax_title.text(0.5, 0.5, "PanoTrack: Panoramic Visual Navigation",
                  ha="center", va="center", fontsize=22, fontweight="bold")
    ax_title.text(0.5, 0.0, "Visual Odometry + Visual Homing with Equirectangular (360°) Cameras  |  Built in one day, zero external data",
                  ha="center", va="center", fontsize=11, color="gray")
    ax_title.axis("off")

    # Feature matches (top)
    ax_fm = fig.add_axes([0.02, 0.62, 0.96, 0.33])
    ax_fm.imshow(img1)
    ax_fm.axis("off")

    # Trajectory (bottom-left)
    ax_traj = fig.add_axes([0.02, 0.02, 0.58, 0.58])
    ax_traj.imshow(img3)
    ax_traj.axis("off")

    # Homing (bottom-right)
    ax_home = fig.add_axes([0.60, 0.02, 0.38, 0.58])
    ax_home.imshow(img2)
    ax_home.axis("off")

    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Loading data...")
    meta, vo_imgs, vo_gt, hom_imgs, hom_pos = load_data("data/proc_scene_v2")

    print("\n1/3 Feature matching visualization...")
    make_feature_matches(vo_imgs, meta)

    print("\n2/3 Visual homing visualization...")
    make_homing_visual(hom_imgs, hom_pos, meta)

    print("\n3/3 Trajectory + perspective views...")
    make_trajectory_views(vo_imgs, vo_gt, meta)

    print("\n4/4 Summary figure...")
    make_summary()

    print("\nDone! All figures in results/")
