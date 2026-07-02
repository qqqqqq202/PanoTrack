"""
Panoramic Visual Odometry (PVO).

Estimates camera trajectory from a sequence of equirectangular images
by tracking features between consecutive frames and recovering relative
pose via the spherical essential matrix.

Pipeline:
  Frame_i → Extract Features → Match with Frame_{i+1}
         → Spherical Essential Matrix → Decompose → (R, t)
         → Accumulate pose → Trajectory
"""

import numpy as np
from pano_track.camera import erp_to_sphere
from pano_track.pose import estimate_essential_ransac, decompose_essential, relative_pose_error


class PanoramicVO:
    """Panoramic Visual Odometry engine."""

    def __init__(self, feature_extractor, feature_matcher, width, height):
        """
        Args:
            feature_extractor: FeatureExtractor instance.
            feature_matcher: FeatureMatcher instance.
            width, height: ERP image dimensions.
        """
        self.extractor = feature_extractor
        self.matcher = feature_matcher
        self.width = width
        self.height = height

        # State
        self.trajectory = []  # list of (3,) positions
        self.rotations = []   # list of (3, 3) rotation matrices
        self.prev_kpts = None
        self.prev_descs = None
        self.frame_count = 0

    def process_frame(self, image, init_pose=True):
        """
        Process one frame and update the trajectory.

        Args:
            image: (H, W, 3) equirectangular image.
            init_pose: if True, initialize trajectory at origin.

        Returns:
            position: (3,) estimated world position.
            rotation: (3, 3) world-from-camera rotation matrix.
            info: dict with debug info (matches, inliers, E, etc.).
        """
        # Extract features
        kpts, descs = self.extractor.extract(image)

        info = {
            "n_keypoints": len(kpts),
            "n_matches": 0,
            "n_inliers": 0,
            "E": None,
            "R_rel": None,
            "t_rel": None,
        }

        if self.frame_count == 0:
            # First frame: initialize
            if init_pose:
                position = np.zeros(3, dtype=np.float32)
                rotation = np.eye(3, dtype=np.float32)
            else:
                position = np.array([0, 0, 0], dtype=np.float32)
                rotation = np.eye(3, dtype=np.float32)

            self.prev_kpts = kpts
            self.prev_descs = descs
            self.frame_count = 1
            self.trajectory.append(position.copy())
            self.rotations.append(rotation.copy())
            return position, rotation, info

        # Match with previous frame
        matched_kpts1, matched_kpts2, mask = self.matcher.match(
            self.prev_kpts, self.prev_descs,
            kpts, descs,
            self.width, self.height,
        )

        info["n_matches"] = len(matched_kpts1)

        if len(matched_kpts1) < 8:
            # Not enough matches: hold position
            position = self.trajectory[-1].copy()
            rotation = self.rotations[-1].copy()
            info["n_inliers"] = 0
            self.prev_kpts = kpts
            self.prev_descs = descs
            self.frame_count += 1
            self.trajectory.append(position.copy())
            self.rotations.append(rotation.copy())
            return position, rotation, info

        # Estimate essential matrix
        E, inlier_mask, n_inliers = estimate_essential_ransac(
            matched_kpts1, matched_kpts2,
            self.width, self.height,
            n_iterations=500,
            threshold=0.005,
        )

        inliers = inlier_mask.sum() if inlier_mask is not None else 0
        info["n_inliers"] = inliers
        info["E"] = E

        if inliers < 8:
            position = self.trajectory[-1].copy()
            rotation = self.rotations[-1].copy()
            self.prev_kpts = kpts
            self.prev_descs = descs
            self.frame_count += 1
            self.trajectory.append(position.copy())
            self.rotations.append(rotation.copy())
            return position, rotation, info

        # Filter to inliers
        kpts1_in = matched_kpts1[inlier_mask]
        kpts2_in = matched_kpts2[inlier_mask]

        # Lift inlier keypoints to unit sphere bearings
        pts1_sphere = erp_to_sphere(kpts1_in, self.width, self.height)
        pts2_sphere = erp_to_sphere(kpts2_in, self.width, self.height)

        # Decompose essential matrix
        R_rel, t_rel, n_front = decompose_essential(
            E, pts1_sphere, pts2_sphere,
        )
        info["R_rel"] = R_rel
        info["t_rel"] = t_rel
        info["n_front"] = n_front

        # Quality check: chirality must be satisfied for most inliers
        front_ratio = n_front / max(inliers, 1)
        if front_ratio < 0.4 or n_front < 10:
            # Unreliable estimate — hold previous pose
            position = self.trajectory[-1].copy()
            rotation = self.rotations[-1].copy()
            self.prev_kpts = kpts
            self.prev_descs = descs
            self.frame_count += 1
            self.trajectory.append(position.copy())
            self.rotations.append(rotation.copy())
            info["skipped"] = True
            return position, rotation, info

        # Accumulate pose
        # Derivation: X_cam2 = R_rel @ X_cam1 + t_rel
        # where R_rel = R_curr^T @ R_prev, t_rel = R_curr^T @ (p_prev - p_curr)
        # Therefore:
        #   R_curr = R_prev @ R_rel^T
        #   p_curr = p_prev - R_curr @ t_rel
        prev_R = self.rotations[-1]
        prev_t = self.trajectory[-1]

        new_R = prev_R @ R_rel.T
        new_t = prev_t - new_R @ t_rel.ravel()

        # Store
        self.trajectory.append(new_t.copy())
        self.rotations.append(new_R.copy())

        # Update state
        self.prev_kpts = kpts
        self.prev_descs = descs
        self.frame_count += 1

        return new_t, new_R, info

    def get_trajectory(self):
        """Return the accumulated trajectory as a (N, 3) array."""
        return np.array(self.trajectory)


def run_vo_on_dataset(images, gt_positions, feature_backend="orb", device="cpu"):
    """
    Run panoramic VO on a sequence of images with ground truth.

    Args:
        images: list of (H, W, 3) equirectangular images.
        gt_positions: (N, 3) ground-truth camera positions.
        feature_backend: "superpoint" or "orb".
        device: device for feature extraction.

    Returns:
        estimated_trajectory: (N, 3) array.
        errors: list of per-frame error metrics.
    """
    from pano_track.features import FeatureExtractor
    from pano_track.matching import FeatureMatcher

    h, w = images[0].shape[:2]

    extractor = FeatureExtractor(backend=feature_backend, max_keypoints=2048, device=device)
    matcher = FeatureMatcher(backend="flann", ratio_thresh=0.75)
    vo = PanoramicVO(extractor, matcher, w, h)

    errors = []
    for i, img in enumerate(images):
        pos, rot, info = vo.process_frame(img, init_pose=(i == 0))

        if i > 0:
            gt_t = gt_positions[i] - gt_positions[0]
            gt_t = gt_t / (np.linalg.norm(gt_t) + 1e-10)
            est_t = pos / (np.linalg.norm(pos) + 1e-10)
            angle_err = np.rad2deg(np.arccos(np.clip(np.dot(gt_t, est_t), -1, 1)))
            errors.append({
                "frame": i,
                "angle_error_deg": angle_err,
                "n_matches": info["n_matches"],
                "n_inliers": info["n_inliers"],
            })

    trajectory = vo.get_trajectory()
    return trajectory, errors
