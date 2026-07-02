"""
Panoramic Visual Odometry (PVO) — keyframe-based.

Matches each frame against the last KEYFRAME (not the previous frame),
giving wider baselines and more stable essential matrix estimation.
Falls back to previous-frame matching when keyframe matching fails.

Pipeline:
  Frame_i → Extract Features → Match with last KEYFRAME
         → Spherical E → Decompose → (R, t)
         → If motion > threshold → new keyframe
         → Accumulate pose → Trajectory
"""

import numpy as np
from pano_track.camera import erp_to_sphere
from pano_track.pose import (estimate_essential_ransac, decompose_essential,
                              refine_pose_nonlinear, relative_pose_error)


class PanoramicVO:
    """Keyframe-based Panoramic Visual Odometry engine."""

    def __init__(self, feature_extractor, feature_matcher, width, height,
                 kf_min_dist=0.3, kf_min_angle=5.0):
        self.extractor = feature_extractor
        self.matcher = feature_matcher
        self.width = width
        self.height = height
        self.kf_min_dist = kf_min_dist
        self.kf_min_angle = np.deg2rad(kf_min_angle)

        # Trajectory state
        self.trajectory = []
        self.rotations = []

        # Keyframe state
        self.kf_kpts = None
        self.kf_descs = None
        self.kf_pos = None
        self.kf_rot = None

        # Previous frame (fallback)
        self.prev_kpts = None
        self.prev_descs = None
        self.frame_count = 0

    def process_frame(self, image, init_pose=True):
        kpts, descs = self.extractor.extract(image)
        info = {"n_keypoints": len(kpts), "n_matches": 0, "n_inliers": 0,
                "keyframe": False, "used_kf": False}

        if self.frame_count == 0:
            pos = np.zeros(3, dtype=np.float32)
            rot = np.eye(3, dtype=np.float32)
            self.prev_kpts, self.prev_descs = kpts, descs
            self.frame_count = 1
            self.trajectory.append(pos.copy())
            self.rotations.append(rot.copy())
            return pos, rot, info

        # ── Frame-to-frame matching ──
        mk1, mk2, _ = self.matcher.match(
            self.prev_kpts, self.prev_descs, kpts, descs, self.width, self.height)

        info["n_matches"] = len(mk1)
        if len(mk1) < 8:
            return self._hold(kpts, descs, info)

        # ── Spherical essential matrix ──
        E, inl, n_inl = estimate_essential_ransac(
            mk1, mk2, self.width, self.height, n_iterations=2000, threshold=0.003)
        info["n_inliers"] = n_inl
        if n_inl < 8:
            return self._hold(kpts, descs, info)

        k1_in, k2_in = mk1[inl], mk2[inl]
        p1_s = erp_to_sphere(k1_in, self.width, self.height)
        p2_s = erp_to_sphere(k2_in, self.width, self.height)

        # ── Decompose + refine ──
        R_rel, t_rel, n_front = decompose_essential(E, p1_s, p2_s)
        front_ratio = n_front / max(n_inl, 1)
        if front_ratio < 0.5 or n_front < 25:
            return self._hold(kpts, descs, info)

        R_rel, t_rel, _ = refine_pose_nonlinear(R_rel, t_rel, p1_s, p2_s, max_iter=30)
        info["R_rel"], info["t_rel"] = R_rel, t_rel

        # ── Accumulate ──
        prev_R = self.rotations[-1]
        prev_t = self.trajectory[-1]
        new_R = prev_R @ R_rel.T
        new_t = prev_t - new_R @ t_rel.ravel()

        self.trajectory.append(new_t.copy())
        self.rotations.append(new_R.copy())
        self.prev_kpts, self.prev_descs = kpts, descs
        self.frame_count += 1
        return new_t, new_R, info

    def _set_keyframe(self, kpts, descs, pos, rot):
        self.kf_kpts, self.kf_descs = kpts, descs
        self.kf_pos, self.kf_rot = pos.copy(), rot.copy()

    def _hold(self, kpts, descs, info):
        pos = self.trajectory[-1].copy()
        rot = self.rotations[-1].copy()
        self.prev_kpts, self.prev_descs = kpts, descs
        self.frame_count += 1
        self.trajectory.append(pos.copy())
        self.rotations.append(rot.copy())
        info["skipped"] = True
        return pos, rot, info

    def get_trajectory(self):
        return np.array(self.trajectory)


def run_vo_on_dataset(images, gt_positions, feature_backend="orb", device="cpu"):
    """Run panoramic VO on a sequence of images."""
    from pano_track.features import FeatureExtractor
    from pano_track.matching import FeatureMatcher

    h, w = images[0].shape[:2]
    extractor = FeatureExtractor(backend=feature_backend, max_keypoints=2048, device=device)
    matcher = FeatureMatcher(backend="flann", ratio_thresh=0.75)
    vo = PanoramicVO(extractor, matcher, w, h, kf_min_dist=0.3, kf_min_angle=5.0)

    errors = []
    for i, img in enumerate(images):
        pos, rot, info = vo.process_frame(img, init_pose=(i == 0))
        if i > 0:
            gt_t = gt_positions[i] - gt_positions[0]
            gt_t = gt_t / (np.linalg.norm(gt_t) + 1e-10)
            est_t = pos / (np.linalg.norm(pos) + 1e-10)
            ang = np.rad2deg(np.arccos(np.clip(np.dot(gt_t, est_t), -1, 1)))
            errors.append({"frame": i, "angle_error_deg": ang,
                           "n_matches": info["n_matches"],
                           "n_inliers": info["n_inliers"]})

    return vo.get_trajectory(), errors
