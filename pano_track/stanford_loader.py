"""
Stanford 2D-3D-Semantics dataset loader.

Loads equirectangular panoramic images and camera poses, grouped by room.
Handles RGBA → RGB conversion and optional downscaling.
"""

import json
import os
import numpy as np
from PIL import Image
from collections import defaultdict


class StanfordDataset:
    """Loader for Stanford 2D-3D-S panoramic data."""

    def __init__(self, data_root, target_size=(512, 256)):
        """
        Args:
            data_root: path to area_X_no_xyz/area_X/ directory
                       (contains pano/rgb/ and pano/pose/ subdirectories).
            target_size: (width, height) to resize images to.
        """
        self.data_root = data_root
        self.target_size = target_size
        self.rgb_dir = os.path.join(data_root, "pano", "rgb")
        self.pose_dir = os.path.join(data_root, "pano", "pose")

        # Build room index
        self.rooms = defaultdict(list)
        self._build_index()

    def _build_index(self):
        """Scan pose files and group by room name."""
        for fname in sorted(os.listdir(self.pose_dir)):
            if not fname.endswith(".json"):
                continue

            pose_path = os.path.join(self.pose_dir, fname)
            with open(pose_path) as f:
                data = json.load(f)

            room = data.get("room", "unknown")
            rgb_fname = fname.replace("_pose.json", "_rgb.png")
            rgb_path = os.path.join(self.rgb_dir, rgb_fname)

            if os.path.exists(rgb_path):
                self.rooms[room].append({
                    "rgb_path": rgb_path,
                    "pose_path": pose_path,
                    "uuid": data.get("camera_uuid", ""),
                    "location": np.array(data["camera_location"], dtype=np.float32),
                    "rt_matrix": np.array(data["camera_rt_matrix"], dtype=np.float32),
                    "room": room,
                })

    def list_rooms(self, min_images=3):
        """Return room names with at least `min_images` viewpoints."""
        return sorted([r for r, v in self.rooms.items() if len(v) >= min_images])

    def room_stats(self):
        """Print summary of each room."""
        print(f"{'Room':<25s} {'Views':>5s}  {'Position Range':>30s}")
        print("-" * 65)
        for room in sorted(self.rooms.keys()):
            views = self.rooms[room]
            locs = np.array([v["location"] for v in views])
            x_range = f"x=[{locs[:,0].min():.1f}, {locs[:,0].max():.1f}]"
            z_range = f"z=[{locs[:,2].min():.1f}, {locs[:,2].max():.1f}]"
            print(f"{room:<25s} {len(views):5d}  {x_range} {z_range}")

    def load_room_images(self, room_name):
        """
        Load all images and poses for a room.

        Args:
            room_name: room identifier string.

        Returns:
            images: list of (H, W, 3) uint8 RGB arrays.
            positions: (N, 3) float32 camera positions.
            metadata: list of dicts with per-view info.
        """
        views = self.rooms.get(room_name, [])
        if not views:
            raise ValueError(f"Room '{room_name}' not found. Available: {self.list_rooms(1)}")

        images = []
        positions = []
        metadata = []

        for view in views:
            # Load and convert RGBA → RGB, resize
            img = np.array(Image.open(view["rgb_path"]))
            if img.shape[-1] == 4:
                img = img[:, :, :3]  # drop alpha
            if self.target_size is not None:
                img = np.array(Image.fromarray(img).resize(
                    self.target_size, Image.LANCZOS))
            images.append(img)
            positions.append(view["location"].copy())
            metadata.append(view)

        return images, np.array(positions, dtype=np.float32), metadata

    def get_room_view(self, room_name, view_idx=0):
        """Load a single view from a room."""
        views = self.rooms.get(room_name, [])
        if view_idx >= len(views):
            raise IndexError(f"Room {room_name} has only {len(views)} views")
        img = np.array(Image.open(views[view_idx]["rgb_path"]))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        if self.target_size is not None:
            img = np.array(Image.fromarray(img).resize(
                self.target_size, Image.LANCZOS))
        return img, views[view_idx]["location"].copy()


if __name__ == "__main__":
    ds = StanfordDataset("D:/edge download/area_3_no_xyz/area_3")
    ds.room_stats()
    print(f"\nUsable rooms (>=3 views): {ds.list_rooms(3)}")
