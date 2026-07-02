"""
Run Panoramic Visual Homing experiment.

Tests the "snapshot model" of insect navigation:
  - Store a panoramic snapshot at a "home" location
  - From query positions, estimate the direction back home
  - Compare estimated bearings with ground truth

Usage:
    python run_homing.py
    python run_homing.py --home-idx 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.homing import VisualHoming, run_homing_experiment
from pano_track.visualize import (
    plot_homing_results, plot_dissimilarity_profile,
    equirect_to_perspective_crop,
)


def main():
    parser = argparse.ArgumentParser(description="Run Visual Homing experiment")
    parser.add_argument("--data-dir", type=str, default="data/proc_scene",
                        help="Path to generated dataset")
    parser.add_argument("--home-idx", type=int, default=0,
                        help="Which viewpoint to use as home (0-14)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load metadata ─────────────────────────────────────
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    views = metadata["homing_views"]
    n_views = len(views)
    print(f"Loaded {n_views} homing viewpoints")
    print(f"Resolution: {metadata['image_resolution']}")

    # ── Load images ───────────────────────────────────────
    images = []
    positions = []
    for view in views:
        img_path = os.path.join(args.data_dir, view["filename"])
        img = np.array(Image.open(img_path))
        images.append(img)
        positions.append(view["position"])
    positions = np.array(positions, dtype=np.float32)

    # ── Set home ──────────────────────────────────────────
    home_idx = args.home_idx
    home_pos = positions[home_idx]
    home_img = images[home_idx]
    print(f"\nHome set at view {home_idx}: position=({home_pos[0]:.1f}, {home_pos[1]:.1f}, {home_pos[2]:.1f})")

    # Query = all other viewpoints
    query_indices = [i for i in range(n_views) if i != home_idx]
    query_images = [images[i] for i in query_indices]
    query_positions = [positions[i] for i in query_indices]

    print(f"Query viewpoints: {len(query_indices)}")

    # ── Run homing experiment ─────────────────────────────
    print("\nRunning visual homing...")
    t0 = time.time()

    results = run_homing_experiment(
        home_img, query_images, query_positions, home_pos
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(query_indices):.2f}s/query)")

    # ── Summary ───────────────────────────────────────────
    errors = [r["bearing_error_deg"] for r in results]
    print(f"\nHoming Results:")
    print(f"  Mean bearing error: {np.mean(errors):.1f}°")
    print(f"  Median bearing error: {np.median(errors):.1f}°")
    print(f"  Min error: {np.min(errors):.1f}°")
    print(f"  Max error: {np.max(errors):.1f}°")
    print(f"  Queries within 30°: {sum(1 for e in errors if e < 30)}/{len(errors)}")
    print(f"  Queries within 15°: {sum(1 for e in errors if e < 15)}/{len(errors)}")

    # ── Visualize ─────────────────────────────────────────
    # Main results figure
    plot_homing_results(
        home_pos, query_positions,
        estimated_bearings=[r["estimated_home_bearing_deg"] for r in results],
        true_bearings=[r["true_home_bearing_deg"] for r in results],
        errors=errors,
        title=f"Visual Homing — Home at View {home_idx}",
        save_path=os.path.join(args.output, "homing_results.png"),
    )

    # Dissimilarity profile for first query
    homing = VisualHoming(
        metadata["image_resolution"][0],
        metadata["image_resolution"][1],
        n_azimuth_bins=360,
    )
    homing.set_home(home_img, home_pos)
    bearing_deg, conf, rot_deg, dissim = homing.estimate_home_bearing(query_images[0])
    home_bin = np.argmin(dissim)
    plot_dissimilarity_profile(
        dissim, home_bin, n_bins=360,
        save_path=os.path.join(args.output, "dissimilarity_profile.png"),
    )

    # Save perspective crops
    for label, img in [("Home", home_img), ("Query (first)", query_images[0])]:
        crop = equirect_to_perspective_crop(img, fov_deg=100, out_size=(400, 400))
        Image.fromarray(crop).save(
            os.path.join(args.output, f"perspective_{label.lower().replace(' ', '_')}.png")
        )

    # Save JSON results
    with open(os.path.join(args.output, "homing_results.json"), "w") as f:
        json.dump({
            "home_position": [float(x) for x in home_pos],
            "home_view_id": home_idx,
            "results": [{
                "query_id": r["query_id"],
                "position": [float(x) for x in r["position"]],
                "estimated_home_bearing_deg": float(r["estimated_home_bearing_deg"]),
                "true_home_bearing_deg": float(r["true_home_bearing_deg"]),
                "bearing_error_deg": float(r["bearing_error_deg"]),
                "confidence": float(r["confidence"]),
                "distance_to_home": float(r["distance_to_home"]),
            } for r in results],
            "summary": {
                "mean_error_deg": float(np.mean(errors)),
                "median_error_deg": float(np.median(errors)),
            },
        }, f, indent=2)

    print(f"\nAll results saved to {args.output}/")


if __name__ == "__main__":
    main()
