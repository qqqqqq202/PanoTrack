"""
Comprehensive visualizations for panoramic place recognition.

Generates:
  1. retrieval_gallery.png  — query + top-5 matches (side by side)
  2. match_map.png          — 2D room map with query→match arrows
  3. sim_vs_dist.png        — similarity vs physical distance scatter
  4. room_confusion.png     — cross-room confusion matrix
  5. placerec_summary.png   — all-in-one summary figure
"""

import json
import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.stanford_loader import StanfordDataset
from pano_track.placerec import PlaceRecognizer
from pano_track.visualize import equirect_to_perspective_crop

C = {
    "gt":       "#2ecc71",
    "est":      "#e74c3c",
    "home":     "#3498db",
    "query":    "#f39c12",
    "match":    "#27ae60",
    "bad_match": "#e74c3c",
    "bg":       "#f8f9fa",
}


def make_retrieval_gallery(ds, rec, room_name, output_dir="results/stanford"):
    """Show query + top-5 retrieved images for best, median, worst queries."""
    images, positions, _ = ds.load_room_images(room_name)
    n = len(images)

    # Get all queries with their top-5
    all_queries = []
    for i in range(n):
        results = rec.query(images[i], top_k=min(7, n))
        non_self = [r for r in results if r["index"] != i]
        if non_self:
            top1_dist = np.linalg.norm(positions[i] - positions[non_self[0]["index"]])
            all_queries.append((i, non_self, top1_dist))

    # Sort by distance
    all_queries.sort(key=lambda x: x[2])
    best = all_queries[0]
    median = all_queries[len(all_queries)//2]
    worst = all_queries[-1]

    showcase = [("BEST", best), ("MEDIAN", median), ("WORST", worst)]

    fig, axes = plt.subplots(3, 6, figsize=(22, 12))

    for row, (label, (qi, non_self, top1_dist)) in enumerate(showcase):
        # Query
        crop_q = equirect_to_perspective_crop(images[qi], fov_deg=90, out_size=(280, 280))
        axes[row, 0].imshow(crop_q)
        axes[row, 0].set_title(f"QUERY #{qi}\npos=({positions[qi][0]:.1f},{positions[qi][2]:.1f})",
                               fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # Top-5 matches
        for col in range(5):
            if col < len(non_self):
                r = non_self[col]
                ti = r["index"]
                dist = np.linalg.norm(positions[qi] - positions[ti])
                crop_t = equirect_to_perspective_crop(images[ti], fov_deg=90, out_size=(280, 280))
                axes[row, col+1].imshow(crop_t)

                color = C["match"] if dist < 3 else C["bad_match"] if dist > 8 else C["query"]
                axes[row, col+1].set_title(
                    f"Rank{col+1} #{ti} | {dist:.1f}m | sim={r['similarity']:.3f}",
                    fontsize=9, color=color, fontweight="bold")
            axes[row, col+1].axis("off")

        # Add label on left
        axes[row, 0].set_ylabel(f"{label}\nTop-1: {top1_dist:.1f}m",
                                fontsize=13, fontweight="bold", rotation=0,
                                labelpad=60, va="center")

    plt.suptitle(f"Place Recognition Retrieval — {room_name} (Stanford 2D-3D-S)\n"
                 "ResNet-18 global descriptors | Cosine similarity retrieval",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"retrieval_gallery_{room_name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved retrieval_gallery_{room_name}.png")


def make_match_map(ds, rec, room_name, output_dir="results/stanford"):
    """2D map of the room with arrows from query to top-1 match."""
    images, positions, _ = ds.load_room_images(room_name)
    n = len(images)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: Query → Match arrows ──
    pos_xz = positions[:, [0, 2]]

    # Plot all camera positions
    ax1.scatter(pos_xz[:, 0], pos_xz[:, 1], c=C["home"], s=80, zorder=3,
                edgecolors="white", linewidth=1.5)

    # Label each position
    for i, (x, z) in enumerate(pos_xz):
        ax1.annotate(str(i), (x, z), fontsize=8, ha="center", va="center",
                     color="white", fontweight="bold")

    # For each query, draw arrow to top-1 match
    for i in range(n):
        results = rec.query(images[i], top_k=min(3, n))
        non_self = [r for r in results if r["index"] != i]
        if not non_self:
            continue
        top1 = non_self[0]
        j = top1["index"]
        dist = np.linalg.norm(positions[i] - positions[j])

        color = C["match"] if dist < 3 else C["query"] if dist < 6 else C["bad_match"]
        alpha = 0.7 if dist < 3 else 0.4

        ax1.annotate("", xy=pos_xz[j], xytext=pos_xz[i],
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=alpha))

    ax1.set_xlabel("X (meters)", fontsize=12)
    ax1.set_ylabel("Z (meters)", fontsize=12)
    ax1.set_title(f"Query → Top-1 Match Map\n{room_name} ({n} views)",
                  fontweight="bold", fontsize=13)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C["match"], alpha=0.7, label="Top-1 < 3m (correct)"),
        mpatches.Patch(color=C["query"], alpha=0.5, label="Top-1 3-6m"),
        mpatches.Patch(color=C["bad_match"], alpha=0.4, label="Top-1 > 6m (wrong)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # ── Right: Error histogram ──
    all_dists = []
    for i in range(n):
        results = rec.query(images[i], top_k=min(3, n))
        non_self = [r for r in results if r["index"] != i]
        if non_self:
            dist = np.linalg.norm(positions[i] - positions[non_self[0]["index"]])
            all_dists.append(dist)

    ax2.hist(all_dists, bins=15, color=C["home"], alpha=0.7, edgecolor="white", linewidth=1)
    ax2.axvline(np.mean(all_dists), color=C["bad_match"], linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(all_dists):.2f}m")
    ax2.axvline(np.median(all_dists), color=C["match"], linestyle="-", linewidth=2,
                label=f"Median: {np.median(all_dists):.2f}m")
    ax2.set_xlabel("Top-1 Match Distance (meters)", fontsize=12)
    ax2.set_ylabel("Number of Queries", fontsize=12)
    ax2.set_title("Top-1 Retrieval Error Distribution", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Place Recognition — {room_name}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"match_map_{room_name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved match_map_{room_name}.png")


def make_sim_vs_dist(ds, rec, room_name, output_dir="results/stanford"):
    """Scatter plot: cosine similarity vs physical distance."""
    images, positions, _ = ds.load_room_images(room_name)
    n = len(images)

    all_pairs = []
    for i in range(n):
        results = rec.query(images[i], top_k=min(n, 20))
        for r in results:
            if r["index"] != i:
                dist = np.linalg.norm(positions[i] - positions[r["index"]])
                all_pairs.append((dist, r["similarity"]))

    dists, sims = zip(*all_pairs)
    dists = np.array(dists)
    sims = np.array(sims)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(dists, sims, c=sims, cmap="RdYlGn", alpha=0.5,
                         s=30, edgecolors="none")

    # Trend line
    z = np.polyfit(dists, sims, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, max(dists), 100)
    ax.plot(x_line, p(x_line), "--", color=C["bad_match"], linewidth=2,
            label=f"Trend: sim = {z[0]:.3f}·dist + {z[1]:.3f}")

    # Correlation
    corr = np.corrcoef(dists, sims)[0, 1]
    ax.set_xlabel("Physical Distance (meters)", fontsize=13)
    ax.set_ylabel("Cosine Similarity", fontsize=13)
    ax.set_title(f"Visual Similarity vs Physical Distance — {room_name}\n"
                 f"Pearson r = {corr:.3f}  |  {len(all_pairs)} pairs",
                 fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Similarity")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sim_vs_dist_{room_name}.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved sim_vs_dist_{room_name}.png")


def make_room_confusion(ds, output_dir="results/stanford"):
    """Cross-room confusion: which rooms look like which other rooms."""
    # Load all images
    all_images, all_positions, all_rooms, room_names = [], [], [], []
    for room in ds.list_rooms(1):
        imgs, poss, _ = ds.load_room_images(room)
        all_images.extend(imgs)
        all_positions.append(poss)
        all_rooms.extend([room] * len(imgs))
        room_names.append(room)
    all_positions = np.concatenate(all_positions)
    n_rooms = len(room_names)

    # Build database
    rec = PlaceRecognizer(device="cpu")
    rec.build_database(all_images)

    # Build confusion matrix: for each query room, which room does top-1 come from?
    confusion = np.zeros((n_rooms, n_rooms), dtype=int)
    for i, room_i in enumerate(all_rooms):
        results = rec.query(all_images[i], top_k=min(10, len(all_images)))
        non_self = [r for r in results if r["index"] != i]
        if non_self:
            j = non_self[0]["index"]
            room_j = all_rooms[j]
            confusion[room_names.index(room_i), room_names.index(room_j)] += 1

    # Normalize to percentages
    confusion_pct = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-10) * 100

    fig, ax = plt.subplots(figsize=(16, 13))
    im = ax.imshow(confusion_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    # Labels
    short_names = [r.replace("_3", "") for r in room_names]
    ax.set_xticks(range(n_rooms))
    ax.set_yticks(range(n_rooms))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Retrieved Room (Top-1 Match)", fontsize=13)
    ax.set_ylabel("Query Room", fontsize=13)
    ax.set_title("Cross-Room Place Recognition Confusion Matrix\n"
                 f"{len(all_images)} images, {n_rooms} rooms | ResNet-18 | Cosine Similarity",
                 fontweight="bold", fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Top-1 Match Percentage (%)", fontsize=11)

    # Annotate diagonal
    for i in range(n_rooms):
        if confusion_pct[i, i] > 50:
            ax.text(i, i, f"{confusion_pct[i,i]:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "room_confusion_matrix.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved room_confusion_matrix.png")


if __name__ == "__main__":
    os.makedirs("results/stanford", exist_ok=True)

    ds = StanfordDataset("D:/edge download/area_3_no_xyz/area_3", target_size=(512, 256))

    # Focus on the best room
    room = "lounge_2_3"
    images, positions, _ = ds.load_room_images(room)

    print(f"Building database for {room}...")
    rec = PlaceRecognizer(device="cpu")
    rec.build_database(images)

    print("\n1/4 Retrieval gallery...")
    make_retrieval_gallery(ds, rec, room)

    print("\n2/4 Match map...")
    make_match_map(ds, rec, room)

    print("\n3/4 Similarity vs distance...")
    make_sim_vs_dist(ds, rec, room)

    print("\n4/4 Cross-room confusion matrix...")
    make_room_confusion(ds)

    print("\nDone! All figures in results/stanford/")
