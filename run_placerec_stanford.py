"""
Panoramic Place Recognition on Stanford 2D-3D-S.

Evaluates: given a query panoramic image, can we find the most similar
image in the database? This is the foundation of loop closure detection
in SLAM — recognizing that you've been somewhere before.

Usage:
    python run_placerec_stanford.py
    python run_placerec_stanford.py --all
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.stanford_loader import StanfordDataset
from pano_track.placerec import PlaceRecognizer
from pano_track.visualize import equirect_to_perspective_crop


def run_room_placerec(ds, room_name, output_dir="results/stanford"):
    """Run place recognition on a single room and visualize results."""
    os.makedirs(output_dir, exist_ok=True)

    images, positions, metadata = ds.load_room_images(room_name)
    n = len(images)
    W, H = ds.target_size

    print(f"\n{'='*60}")
    print(f"Room: {room_name} ({n} views)")
    print(f"{'='*60}")

    # Build database
    rec = PlaceRecognizer(device="cpu")
    rec.build_database(images, metadata=metadata)

    # Query: each image queries the database, exclude self
    correct_dist_1m = 0
    correct_dist_3m = 0
    correct_same_room = 0  # always true for single room
    top1_distances = []
    all_queries = []

    for i in range(n):
        results = rec.query(images[i], top_k=min(6, n))
        # Exclude self (rank 0 is always self with sim=1.0)
        non_self = [r for r in results if r["index"] != i]

        if not non_self:
            continue

        top1 = non_self[0]
        dist = np.linalg.norm(positions[i] - positions[top1["index"]])

        top1_distances.append(dist)
        if dist < 1.0:
            correct_dist_1m += 1
        if dist < 3.0:
            correct_dist_3m += 1

        all_queries.append({
            "query_idx": i,
            "query_pos": positions[i].tolist(),
            "top1_idx": top1["index"],
            "top1_pos": positions[top1["index"]].tolist(),
            "top1_dist": float(dist),
            "top1_sim": top1["similarity"],
            "top3_indices": [r["index"] for r in non_self[:3]],
            "top3_sims": [r["similarity"] for r in non_self[:3]],
        })

    # ── Metrics ───────────────────────────────────────────
    print(f"\nPlace Recognition Results — {room_name}:")
    print(f"  Top-1 mean distance:   {np.mean(top1_distances):.2f}m")
    print(f"  Top-1 median distance: {np.median(top1_distances):.2f}m")
    print(f"  Top-1 within 1m:       {correct_dist_1m}/{n} ({100*correct_dist_1m/n:.0f}%)")
    print(f"  Top-1 within 3m:       {correct_dist_3m}/{n} ({100*correct_dist_3m/n:.0f}%)")

    # ── Visualize: 3 best and 3 worst queries ─────────────
    sorted_queries = sorted(all_queries, key=lambda q: q["top1_dist"])
    showcase_queries = sorted_queries[:3] + sorted_queries[-3:]

    fig, axes = plt.subplots(len(showcase_queries), 4, figsize=(16, 3.5 * len(showcase_queries)))

    if len(showcase_queries) == 1:
        axes = axes.reshape(1, -1)

    for row, q in enumerate(showcase_queries):
        qi = q["query_idx"]

        # Query image
        crop_q = equirect_to_perspective_crop(images[qi], fov_deg=90, out_size=(250, 250))
        axes[row, 0].imshow(crop_q)
        axes[row, 0].set_title(f"Query #{qi}\n({positions[qi][0]:.1f}, {positions[qi][2]:.1f})",
                               fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Top-3 retrievals
        for col, (ti, sim) in enumerate(zip(q["top3_indices"], q["top3_sims"])):
            crop_t = equirect_to_perspective_crop(images[ti], fov_deg=90, out_size=(250, 250))
            axes[row, col + 1].imshow(crop_t)
            dist = np.linalg.norm(positions[qi] - positions[ti])
            color = "green" if dist < 3 else "orange" if dist < 5 else "red"
            axes[row, col + 1].set_title(
                f"Rank {col+1}: #{ti}  |  {dist:.1f}m  |  sim={sim:.3f}",
                fontsize=9, color=color, fontweight="bold")
            axes[row, col + 1].axis("off")

    status = "BEST" if q["top1_dist"] < 3 else "WORST"
    plt.suptitle(f"Place Recognition — {room_name}\nTop 3 BEST + Bottom 3 WORST queries",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"placerec_{room_name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return {
        "room": room_name,
        "n_views": n,
        "mean_top1_dist": float(np.mean(top1_distances)),
        "median_top1_dist": float(np.median(top1_distances)),
        "within_1m_pct": float(100 * correct_dist_1m / n),
        "within_3m_pct": float(100 * correct_dist_3m / n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="D:/edge download/area_3_no_xyz/area_3")
    parser.add_argument("--room", type=str, default=None)
    parser.add_argument("--all", action="store_true")
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
        rooms = ["lounge_2_3", "hallway_4_3", "office_10_3", "conferenceRoom_1_3"]

    all_results = {}
    for room in rooms:
        if room not in ds.rooms:
            continue
        result = run_room_placerec(ds, room, args.output)
        all_results[room] = result

    # ── Cross-room summary ─────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("PLACE RECOGNITION — CROSS-ROOM SUMMARY")
        print(f"{'='*60}")
        print(f"{'Room':<25s} {'Views':>5s} {'MeanDist':>9s} {'MedDist':>9s} {'<1m':>6s} {'<3m':>6s}")
        print("-" * 65)
        for room, r in sorted(all_results.items()):
            print(f"{room:<25s} {r['n_views']:5d} {r['mean_top1_dist']:8.2f}m "
                  f"{r['median_top1_dist']:8.2f}m {r['within_1m_pct']:5.0f}% {r['within_3m_pct']:5.0f}%")
        print("-" * 65)
        mean_dists = [r["mean_top1_dist"] for r in all_results.values()]
        print(f"{'OVERALL':<25s} {'':>5s} {np.mean(mean_dists):8.2f}m")

        with open(os.path.join(args.output, "placerec_summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
