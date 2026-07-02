"""
Generate the synthetic panoramic dataset.

This script:
  1. Creates the procedural indoor 3D scene
  2. Samples a camera trajectory (50 frames for VO)
  3. Samples random viewpoints (15 for visual homing)
  4. Renders equirectangular panoramas at each position
  5. Saves images + ground-truth poses to data/proc_scene/

Usage:
    python generate_data.py
    python generate_data.py --width 512 --height 256
    python generate_data.py --quick  # fast preview mode
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pano_track.scene import create_corridor_scene, sample_camera_path, sample_homing_viewpoints
from pano_track.renderer import render_equirectangular, render_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic panoramic dataset")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=256, help="Image height (default: 256)")
    parser.add_argument("--vo-frames", type=int, default=50, help="VO trajectory frames (default: 50)")
    parser.add_argument("--homing-views", type=int, default=15, help="Homing viewpoints (default: 15)")
    parser.add_argument("--quick", action="store_true", help="Quick preview: low res + few frames")
    parser.add_argument("--outdir", type=str, default="data/proc_scene", help="Output directory")
    args = parser.parse_args()

    if args.quick:
        args.width = 256
        args.height = 128
        args.vo_frames = 10
        args.homing_views = 5

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "vo"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "homing"), exist_ok=True)

    # ── Step 1: Build scene ───────────────────────────────
    print("=" * 60)
    print("Step 1/4: Building procedural indoor scene...")
    t0 = time.time()
    scene = create_corridor_scene()
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Step 2: Sample camera positions ───────────────────
    print("\nStep 2/4: Sampling camera positions...")
    vo_positions, vo_rotations = sample_camera_path(args.vo_frames)
    homing_positions = sample_homing_viewpoints(args.homing_views)
    print(f"  VO path:      {len(vo_positions)} frames")
    print(f"  Homing views: {len(homing_positions)} viewpoints")

    # ── Step 3: Render VO sequence ────────────────────────
    print(f"\nStep 3/4: Rendering VO sequence ({args.width}x{args.height})...")
    t0 = time.time()
    vo_images = render_dataset(
        scene, vo_positions, vo_rotations,
        width=args.width, height=args.height,
        verbose=True,
    )
    render_time = time.time() - t0
    print(f"  Done in {render_time:.1f}s "
          f"({render_time / len(vo_positions):.1f}s/frame)")

    # Save VO data
    vo_metadata = []
    for i, (pos, rot, img) in enumerate(zip(vo_positions, vo_rotations, vo_images)):
        fname = f"frame_{i:04d}.png"
        from PIL import Image
        Image.fromarray(img).save(os.path.join(outdir, "vo", fname))
        vo_metadata.append({
            "frame_id": i,
            "filename": f"vo/{fname}",
            "position": pos.tolist(),
            "rotation": rot.tolist(),
        })
        # Also save individual pose as text for easy parsing
        pose_row = np.concatenate([pos, rot.flatten()])
        np.savetxt(os.path.join(outdir, "vo", f"frame_{i:04d}_pose.txt"),
                   pose_row.reshape(1, -1),
                   header="px py pz r00 r01 r02 r10 r11 r12 r20 r21 r22")

    # ── Step 4: Render homing viewpoints ──────────────────
    print(f"\nStep 4/4: Rendering homing viewpoints...")
    t0 = time.time()
    homing_images = []
    for i, pos in enumerate(homing_positions):
        print(f"\r  Homing {i+1}/{len(homing_positions)} — pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
              end="", flush=True)
        img = render_equirectangular(scene, pos, width=args.width, height=args.height)
        homing_images.append(img)
    print(f"\n  Done in {time.time() - t0:.1f}s")

    # Save homing data
    homing_metadata = []
    for i, (pos, img) in enumerate(zip(homing_positions, homing_images)):
        fname = f"view_{i:04d}.png"
        from PIL import Image
        Image.fromarray(img).save(os.path.join(outdir, "homing", fname))
        homing_metadata.append({
            "view_id": i,
            "filename": f"homing/{fname}",
            "position": pos.tolist(),
        })

    # ── Save metadata ─────────────────────────────────────
    metadata = {
        "scene": "procedural_corridor_room",
        "image_resolution": [args.width, args.height],
        "vo_frames": args.vo_frames,
        "homing_viewpoints": args.homing_views,
        "camera_height": 1.5,
        "vo_trajectory": vo_metadata,
        "homing_views": homing_metadata,
    }

    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────
    total_images = len(vo_images) + len(homing_images)
    total_size_mb = sum(
        os.path.getsize(os.path.join(outdir, m["filename"]))
        for m in vo_metadata + homing_metadata
    ) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Dataset generated successfully!")
    print(f"  Output:    {os.path.abspath(outdir)}")
    print(f"  Images:    {total_images} ({args.width}×{args.height})")
    print(f"  Size:      {total_size_mb:.1f} MB")
    print(f"  VO frames: {args.vo_frames}")
    print(f"  Homing:    {args.homing_views} views")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
