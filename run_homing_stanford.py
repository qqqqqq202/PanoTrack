"""
Run Visual Homing experiment on Stanford 2D-3D-S real data.

Tests the snapshot model across multiple real indoor rooms:
  - office, hallway, lounge, conference room, WC

For each room with ≥3 viewpoints:
  - Leave-one-out: each view serves as "home" once
  - All other views are query positions
  - Compare estimated vs true home bearing

Usage:
    python run_homing_stanford.py
    python run_homing_stanford.py --room hallway_4_3
    python run_homing_stanford.py --all
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.stanford_loader import StanfordDataset
from pano_track.homing import VisualHoming, run_homing_experiment
from pano_track.visualize import (
    plot_homing_results, plot_dissimilarity_profile,
    equirect_to_perspective_crop,
)


def run_room_homing(ds, room_name, output_dir="results/stanford"):
    """Run homing experiment on a single room, leave-one-out evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    images, positions, metadata = ds.load_room_images(room_name)
    W, H = ds.target_size
    n_views = len(images)

    print(f"\n{'='*60}")
    print(f"Room: {room_name}  ({n_views} views)")
    print(f"Resolution: {W}x{H}")
    print(f"{'='*60}")

    all_errors = []
    per_home_results = {}

    for home_idx in range(n_views):
        home_img = images[home_idx]
        home_pos = positions[home_idx]

        # Query = all other viewpoints
        query_indices = [i for i in range(n_views) if i != home_idx]
        query_images = [images[i] for i in query_indices]
        query_positions = [positions[i] for i in query_indices]

        # Run homing
        results = run_homing_experiment(
            home_img, query_images, query_positions, home_pos
        )

        errors = [r["bearing_error_deg"] for r in results]
        all_errors.extend(errors)
        per_home_results[home_idx] = {
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "n_queries": len(results),
        }

    # ── Summary ───────────────────────────────────────────
    if all_errors:
        print(f"\n{'─'*40}")
        print(f"Room {room_name} — Leave-one-out summary:")
        print(f"  Total queries: {len(all_errors)}")
        print(f"  Mean error:    {np.mean(all_errors):.1f}°")
        print(f"  Median error:  {np.median(all_errors):.1f}°")
        print(f"  Std:           {np.std(all_errors):.1f}°")
        print(f"  Within 30°:    {sum(1 for e in all_errors if e < 30)}/{len(all_errors)} "
              f"({100*sum(1 for e in all_errors if e < 30)/len(all_errors):.0f}%)")
        print(f"  Within 15°:    {sum(1 for e in all_errors if e < 15)}/{len(all_errors)} "
              f"({100*sum(1 for e in all_errors if e < 15)/len(all_errors):.0f}%)")

    # ── Visualize best home ───────────────────────────────
    best_home = min(per_home_results, key=lambda k: per_home_results[k]["mean_error"])
    print(f"\nBest home: view {best_home} (mean error {per_home_results[best_home]['mean_error']:.1f}°)")

    # Generate detailed plot for best home
    home_img = images[best_home]
    home_pos = positions[best_home]
    query_indices = [i for i in range(n_views) if i != best_home]

    homing = VisualHoming(W, H, n_azimuth_bins=360)
    homing.set_home(home_img, home_pos)

    # Collect detailed results for the best home
    detailed_results = []
    for qi in query_indices:
        bearing_deg, conf, rot_deg, dissim = homing.estimate_home_bearing(images[qi])
        true_vec = home_pos[[0, 2]] - positions[qi][[0, 2]]
        true_vec = true_vec / (np.linalg.norm(true_vec) + 1e-10)
        true_bearing = np.rad2deg(np.arctan2(true_vec[1], true_vec[0]))
        err = abs(bearing_deg - true_bearing)
        if err > 180:
            err = 360 - err
        detailed_results.append({
            "query_id": qi,
            "position": positions[qi].tolist(),
            "estimated_home_bearing_deg": float(bearing_deg),
            "true_home_bearing_deg": float(true_bearing),
            "bearing_error_deg": float(err),
            "confidence": float(conf),
            "distance_to_home": float(np.linalg.norm(home_pos - positions[qi])),
        })

    # Plot homing results
    plot_homing_results(
        home_pos, [positions[i] for i in query_indices],
        estimated_bearings=[r["estimated_home_bearing_deg"] for r in detailed_results],
        true_bearings=[r["true_home_bearing_deg"] for r in detailed_results],
        errors=[r["bearing_error_deg"] for r in detailed_results],
        title=f"Visual Homing — {room_name} (Stanford 2D-3D-S)",
        save_path=os.path.join(output_dir, f"homing_{room_name}.png"),
    )

    # Dissimilarity profile for one query
    first_query = query_indices[0]
    bearing_deg, conf, rot_deg, dissim = homing.estimate_home_bearing(images[first_query])
    home_bin = np.argmin(dissim)
    plot_dissimilarity_profile(
        dissim, home_bin, 360,
        save_path=os.path.join(output_dir, f"dissimilarity_{room_name}.png"),
    )

    # Save perspective crops
    for label, img in [("Home", home_img), ("Query", images[first_query])]:
        crop = equirect_to_perspective_crop(img, fov_deg=90, out_size=(400, 400))
        Image.fromarray(crop).save(
            os.path.join(output_dir, f"{label.lower()}_{room_name}.png")
        )

    # Save JSON
    with open(os.path.join(output_dir, f"results_{room_name}.json"), "w") as f:
        json.dump({
            "room": room_name,
            "n_views": n_views,
            "best_home_idx": best_home,
            "summary": {
                "mean_error_deg": float(np.mean(all_errors)),
                "median_error_deg": float(np.median(all_errors)),
                "within_30deg_pct": float(100 * sum(1 for e in all_errors if e < 30) / len(all_errors)),
            },
            "best_home_results": detailed_results,
        }, f, indent=2)

    return all_errors, per_home_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="D:/edge download/area_3_no_xyz/area_3")
    parser.add_argument("--room", type=str, default=None,
                        help="Specific room (e.g. hallway_4_3)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all rooms with >=3 views")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", type=str, default="results/stanford")
    args = parser.parse_args()

    ds = StanfordDataset(args.data_root, target_size=(args.width, args.height))
    ds.room_stats()

    if args.room:
        rooms = [args.room]
    elif args.all:
        rooms = ds.list_rooms(3)
    else:
        # Default: pick the best rooms for homing
        rooms = ["hallway_4_3", "lounge_2_3", "office_10_3", "conferenceRoom_1_3"]
        print(f"\nDefault rooms (use --all for all): {rooms}")

    all_room_errors = {}
    for room in rooms:
        if room not in ds.rooms:
            print(f"Room '{room}' not found, skipping.")
            continue
        errors, per_home = run_room_homing(ds, room, args.output)
        all_room_errors[room] = {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "n_queries": len(errors),
        }

    # ── Cross-room summary ─────────────────────────────────
    if len(all_room_errors) > 1:
        print(f"\n{'='*60}")
        print("CROSS-ROOM SUMMARY")
        print(f"{'='*60}")
        print(f"{'Room':<25s} {'Queries':>7s} {'Mean':>8s} {'Median':>8s}")
        print("-" * 55)
        for room, stats in sorted(all_room_errors.items()):
            print(f"{room:<25s} {stats['n_queries']:7d} {stats['mean']:7.1f}deg {stats['median']:7.1f}deg")
        print("-" * 55)
        means = [s["mean"] for s in all_room_errors.values()]
        print(f"{'OVERALL':<25s} {'':>7s} {np.mean(means):7.1f}deg {np.median(means):7.1f}deg")

        with open(os.path.join(args.output, "cross_room_summary.json"), "w") as f:
            json.dump(all_room_errors, f, indent=2)


if __name__ == "__main__":
    main()
