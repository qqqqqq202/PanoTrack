"""
Visualization utilities for panoramic visual navigation.

Generates publication-quality plots for:
  - VO trajectory (2D top-down + 3D)
  - Feature matching visualizations
  - Visual homing results (bearing error, dissimilarity profiles)
  - Equirectangular image crops (perspective-like views for display)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import cv2


# ── Color Palette ─────────────────────────────────────────
COLORS = {
    "gt": "#2ecc71",        # green — ground truth
    "est": "#e74c3c",       # red — estimated
    "home": "#3498db",      # blue — home
    "query": "#f39c12",     # orange — query
    "bearing": "#9b59b6",   # purple — bearing
    "furniture": "#7f8c8d", # gray
    "bg": "#ecf0f1",        # light gray background
}


def plot_vo_trajectory(gt_positions, est_positions, title="Panoramic VO Trajectory",
                        save_path=None):
    """
    Plot ground-truth vs estimated VO trajectory (2D top-down).

    Args:
        gt_positions: (N, 2) or (N, 3) ground-truth XZ positions.
        est_positions: (N, 2) or (N, 3) estimated XZ positions.
        title: plot title.
        save_path: if provided, save figure to this path.
    """
    gt = np.asarray(gt_positions)
    est = np.asarray(est_positions)

    if gt.ndim == 2 and gt.shape[1] >= 2:
        gt_xz = gt[:, [0, 2]] if gt.shape[1] == 3 else gt[:, :2]
    else:
        gt_xz = gt[:, [0, 2]]
    if est.ndim == 2 and est.shape[1] >= 2:
        est_xz = est[:, [0, 2]] if est.shape[1] == 3 else est[:, :2]
    else:
        est_xz = est[:, [0, 2]]

    # Align first position
    est_xz = est_xz - est_xz[0] + gt_xz[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw scene layout (approximate)
    _draw_scene_layout(ax)

    # Plot trajectories
    ax.plot(gt_xz[:, 0], gt_xz[:, 1], "o-", color=COLORS["gt"],
            linewidth=2, markersize=3, label="Ground Truth", alpha=0.8)
    ax.plot(est_xz[:, 0], est_xz[:, 1], "s--", color=COLORS["est"],
            linewidth=2, markersize=3, label="Estimated (up to scale)", alpha=0.8)

    # Start/end markers
    ax.scatter(*gt_xz[0], c=COLORS["gt"], s=100, marker="o",
               edgecolors="white", linewidth=1, zorder=5, label="Start")
    ax.scatter(*gt_xz[-1], c=COLORS["gt"], s=100, marker="*",
               edgecolors="white", linewidth=1, zorder=5, label="End")

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot to {save_path}")
    plt.close()


def _draw_scene_layout(ax):
    """Draw approximate scene layout (corridor + room) on the given axes."""
    # Corridor walls
    corridor_color = "#bdc3c7"
    ax.fill_between([-6, 6], -1, 1, color=corridor_color, alpha=0.2)
    ax.plot([-6, 6], [-1, -1], color=corridor_color, linewidth=1, linestyle=":")
    ax.plot([-6, 6], [1, 1], color=corridor_color, linewidth=1, linestyle=":")

    # Room
    ax.fill_between([6, 10], -3, 3, color="#d5dbdb", alpha=0.2)
    ax.plot([6, 10], [-3, -3], color=corridor_color, linewidth=1, linestyle=":")
    ax.plot([6, 10], [3, 3], color=corridor_color, linewidth=1, linestyle=":")
    ax.plot([10, 10], [-3, 3], color=corridor_color, linewidth=1, linestyle=":")


def plot_homing_results(home_pos, query_positions, estimated_bearings,
                         true_bearings, errors, title="Visual Homing Results",
                         save_path=None):
    """
    Plot visual homing results showing estimated vs true home bearings.

    Args:
        home_pos: (2,) or (3,) home position.
        query_positions: (N, 2) or (N, 3) query positions.
        estimated_bearings: (N,) estimated bearing angles in degrees.
        true_bearings: (N,) true bearing angles in degrees.
        errors: (N,) bearing errors in degrees.
        title: plot title.
        save_path: optional save path.
    """
    home = np.asarray(home_pos)[[0, 2]] if len(home_pos) == 3 else np.asarray(home_pos)[:2]
    queries = np.asarray(query_positions)
    if queries.shape[1] == 3:
        queries = queries[:, [0, 2]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: bearing arrows ──
    _draw_scene_layout(ax1)
    ax1.scatter(*home, c=COLORS["home"], s=200, marker="H",
                edgecolors="white", linewidth=2, zorder=10, label="Home")

    for i, (q, e_b, t_b, err) in enumerate(zip(queries, estimated_bearings,
                                                  true_bearings, errors)):
        ax1.scatter(*q, c=COLORS["query"], s=30, zorder=5)

        # Draw estimated bearing (short arrow)
        e_rad = np.deg2rad(e_b)
        e_vec = np.array([np.cos(e_rad), -np.sin(e_rad)]) * 1.0
        ax1.arrow(q[0], q[1], e_vec[0], e_vec[1],
                  head_width=0.15, head_length=0.15, fc=COLORS["est"], ec=COLORS["est"],
                  alpha=0.6, width=0.03, label="Estimated" if i == 0 else "")

        # Draw true bearing (short arrow)
        t_rad = np.deg2rad(t_b)
        t_vec = np.array([np.cos(t_rad), -np.sin(t_rad)]) * 1.0
        ax1.arrow(q[0], q[1], t_vec[0], t_vec[1],
                  head_width=0.15, head_length=0.15, fc=COLORS["gt"], ec=COLORS["gt"],
                  alpha=0.6, width=0.03, linestyle="--", label="True" if i == 0 else "")

    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Z (meters)")
    ax1.set_title("Home Bearing Estimates", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── Right: error vs distance ──
    distances = np.linalg.norm(queries - home, axis=-1)
    scatter = ax2.scatter(distances, errors, c=errors, cmap="RdYlGn_r",
                          s=60, edgecolors="gray", linewidth=0.5, vmin=0, vmax=90)
    ax2.axhline(y=np.mean(errors), color=COLORS["est"], linestyle="--",
                label=f"Mean error: {np.mean(errors):.1f}°")
    ax2.set_xlabel("Distance to Home (meters)")
    ax2.set_ylabel("Bearing Error (degrees)")
    ax2.set_title("Homing Error vs Distance", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label="Error (°)")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved homing plot to {save_path}")
    plt.close()


def plot_dissimilarity_profile(dissimilarity, home_bin, n_bins=360,
                                save_path=None):
    """
    Plot the per-column dissimilarity profile with home bearing marked.

    Args:
        dissimilarity: (n_bins,) array of per-bin differences.
        home_bin: index of the minimum-dissimilarity bin (home bearing).
        n_bins: number of azimuthal bins.
        save_path: optional save path.
    """
    bearings = np.linspace(-180, 180, n_bins)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bearings, dissimilarity, color=COLORS["est"], linewidth=1.5, alpha=0.8)
    ax.fill_between(bearings, 0, dissimilarity, color=COLORS["est"], alpha=0.1)

    home_bearing = (home_bin / n_bins) * 360
    if home_bearing > 180:
        home_bearing -= 360
    ax.axvline(x=home_bearing, color=COLORS["home"], linestyle="--", linewidth=2,
               label=f"Home bearing: {home_bearing:.1f}°")

    ax.set_xlabel("Bearing (degrees)")
    ax.set_ylabel("Dissimilarity")
    ax.set_title("Panoramic Dissimilarity Profile", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-180, 180)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def equirect_to_perspective_crop(erp_img, fov_deg=90, bearing_deg=0, tilt_deg=0,
                                  out_size=(400, 400)):
    """
    Extract a perspective-like crop from an equirectangular image.

    Useful for displaying "what the camera sees" in a familiar format.

    Args:
        erp_img: (H, W, 3) equirectangular image.
        fov_deg: field of view of the output crop.
        bearing_deg: horizontal bearing (0 = center/forward).
        tilt_deg: vertical tilt (0 = horizon).
        out_size: (width, height) of output crop.

    Returns:
        perspective crop as (H, W, 3) uint8 image.
    """
    h, w = erp_img.shape[:2]
    out_w, out_h = out_size

    # Map output pixels to spherical coordinates
    fov_rad = np.deg2rad(fov_deg)
    bearing_rad = np.deg2rad(bearing_deg)
    tilt_rad = np.deg2rad(tilt_deg)

    # Output pixel grid
    y, x = np.meshgrid(
        np.linspace(-1, 1, out_h),
        np.linspace(-1, 1, out_w),
        indexing="ij",
    )

    # Gnomonic projection (tangent plane)
    # For a given FOV, the tangent plane is at distance 1/tan(fov/2)
    focal = 1.0 / np.tan(fov_rad / 2)
    xs = x
    ys = y
    zs = focal

    # Normalize
    norm = np.sqrt(xs**2 + ys**2 + zs**2)
    xs, ys, zs = xs / norm, ys / norm, zs / norm

    # Rotate by bearing and tilt
    # Rotation around Y axis (bearing)
    cos_b, sin_b = np.cos(bearing_rad), np.sin(bearing_rad)
    xr = xs * cos_b + zs * sin_b
    zr = -xs * sin_b + zs * cos_b
    yr = ys  # simplified (no tilt for now, but could add)

    # Convert to spherical
    lon = np.arctan2(xr, zr)
    lat = np.arcsin(np.clip(yr, -1, 1))

    # Convert to pixel coordinates
    u = (lon + np.pi) / (2 * np.pi) * w
    v = (lat + np.pi / 2) / np.pi * h

    # Remap
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    crop = cv2.remap(erp_img, u, v, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)

    return crop


def create_summary_figure(vo_trajectory, gt_trajectory, homing_results,
                           sample_images, save_path="data/summary.png"):
    """
    Create a comprehensive summary figure combining VO + homing results.

    Args:
        vo_trajectory: (N, 3) estimated VO positions.
        gt_trajectory: (N, 3) ground-truth positions.
        homing_results: list of dicts from run_homing_experiment.
        sample_images: list of 3 sample ERP images for display.
        save_path: output path.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top-left: VO trajectory
    ax_vo = fig.add_subplot(gs[0, 0])
    gt = np.asarray(gt_trajectory)
    est = np.asarray(vo_trajectory)
    est = est - est[0] + gt[0]
    _draw_scene_layout(ax_vo)
    ax_vo.plot(gt[:, 0], gt[:, 2], "o-", color=COLORS["gt"], markersize=2, label="GT")
    ax_vo.plot(est[:, 0], est[:, 2], "s--", color=COLORS["est"], markersize=2, label="VO")
    ax_vo.set_title("Visual Odometry Trajectory", fontweight="bold")
    ax_vo.set_xlabel("X (m)")
    ax_vo.set_ylabel("Z (m)")
    ax_vo.legend(fontsize=8)
    ax_vo.set_aspect("equal")
    ax_vo.grid(True, alpha=0.3)

    # Top-middle: Homing bearings
    ax_home = fig.add_subplot(gs[0, 1])
    _draw_scene_layout(ax_home)
    if homing_results:
        home_pos = np.array([7.0, 1.5, -2.0])  # approximate
        ax_home.scatter(*home_pos[[0, 2]], c=COLORS["home"], s=120, marker="H",
                        edgecolors="white", linewidth=1.5, zorder=10)
        for hr in homing_results[:10]:  # show first 10
            q = np.array(hr["position"])
            ax_home.scatter(q[0], q[2], c=COLORS["query"], s=20, zorder=5)
    ax_home.set_title("Visual Homing", fontweight="bold")
    ax_home.set_xlabel("X (m)")
    ax_home.set_ylabel("Z (m)")
    ax_home.set_aspect("equal")
    ax_home.grid(True, alpha=0.3)

    # Top-right: Error distribution
    ax_err = fig.add_subplot(gs[0, 2])
    if homing_results:
        errs = [hr["bearing_error_deg"] for hr in homing_results]
        ax_err.hist(errs, bins=15, color=COLORS["est"], alpha=0.7, edgecolor="white")
        ax_err.axvline(np.mean(errs), color="black", linestyle="--",
                       label=f"Mean: {np.mean(errs):.1f}°")
        ax_err.set_xlabel("Bearing Error (°)")
        ax_err.set_ylabel("Count")
        ax_err.set_title("Homing Error Distribution", fontweight="bold")
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)

    # Bottom row: sample images
    for i, (img, label) in enumerate(sample_images):
        ax_img = fig.add_subplot(gs[1, i])
        crop = equirect_to_perspective_crop(img, fov_deg=100,
                                             bearing_deg=0, out_size=(200, 200))
        ax_img.imshow(crop)
        ax_img.set_title(label, fontsize=10)
        ax_img.axis("off")

    # Bottom row 2: more samples
    for i in range(3):
        ax_img2 = fig.add_subplot(gs[2, i])
        if i == 0:
            ax_img2.text(0.5, 0.5, "PanoTrack\nPanoramic Visual Navigation",
                         ha="center", va="center", fontsize=14, fontweight="bold",
                         transform=ax_img2.transAxes)
        elif i == 2:
            ax_img2.text(0.5, 0.5, "github.com/<user>/PanoTrack",
                         ha="center", va="center", fontsize=12,
                         transform=ax_img2.transAxes)
        ax_img2.axis("off")

    fig.suptitle("PanoTrack: Panoramic Visual Odometry + Visual Homing",
                 fontsize=16, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure to {save_path}")
