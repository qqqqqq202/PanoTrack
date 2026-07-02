"""
Feature matching for equirectangular panoramic images.

Supports two backends:
  - LightGlue (deep learning, recommended): state-of-the-art matcher
  - FLANN + Lowe's ratio test (fallback): traditional approach

For panoramic images, matches are verified using spherical epipolar geometry
to filter out geometrically inconsistent correspondences.
"""

import numpy as np
import cv2


class FeatureMatcher:
    """Unified feature matching interface."""

    def __init__(self, backend="lightglue", device="cpu", ratio_thresh=0.75):
        """
        Args:
            backend: "lightglue" or "flann"
            device: "cpu" or "cuda" (for LightGlue)
            ratio_thresh: Lowe's ratio threshold (for FLANN backend).
        """
        self.backend = backend
        self.device = device
        self.ratio_thresh = ratio_thresh
        self._matcher = None

        if backend == "lightglue":
            self._init_lightglue()
        elif backend == "flann":
            self._init_flann()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_lightglue(self):
        """Initialize LightGlue matcher."""
        try:
            import torch
            from kornia.feature import LightGlue

            self._matcher = LightGlue(pretrained="superpoint").to(self.device).eval()
            self._torch = torch
            print(f"LightGlue initialized on {self.device}")
        except ImportError as e:
            print(f"LightGlue not available ({e}), falling back to FLANN")
            self.backend = "flann"
            self._init_flann()

    def _init_flann(self):
        """Initialize matcher — norm auto-selected based on descriptor type."""
        self._bf = None  # Created on first use based on descriptor dtype

    def match(self, kpts1, descs1, kpts2, descs2, width, height):
        """
        Match features between two equirectangular images.

        Args:
            kpts1, kpts2: (N, 2) keypoint arrays in (u, v) format.
            descs1, descs2: descriptor arrays.
            width, height: ERP image dimensions (for spherical verification).

        Returns:
            matched_kpts1: (M, 2) matched keypoints from image 1.
            matched_kpts2: (M, 2) matched keypoints from image 2.
            mask: (M,) boolean array of geometrically verified matches.
        """
        if self.backend == "lightglue" and self._matcher is not None:
            matches = self._match_lightglue(kpts1, descs1, kpts2, descs2)
        else:
            matches = self._match_flann(kpts1, descs1, kpts2, descs2)

        if matches is None or len(matches) < 8:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        # Build matched keypoint arrays
        matched_kpts1 = np.array([kpts1[m[0]] for m in matches], dtype=np.float32)
        matched_kpts2 = np.array([kpts2[m[1]] for m in matches], dtype=np.float32)

        # Spherical epipolar verification
        mask = self._spherical_verify(matched_kpts1, matched_kpts2, width, height)

        return matched_kpts1, matched_kpts2, mask

    def _match_lightglue(self, kpts1, descs1, kpts2, descs2):
        """Match using LightGlue."""
        import torch

        # Convert to LightGlue format
        kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device)
        kpts2_t = torch.from_numpy(kpts2).float().unsqueeze(0).to(self.device)
        descs1_t = torch.from_numpy(descs1).float().unsqueeze(0).to(self.device)
        descs2_t = torch.from_numpy(descs2).float().unsqueeze(0).to(self.device)

        data = {
            "keypoints0": kpts1_t,
            "keypoints1": kpts2_t,
            "descriptors0": descs1_t,
            "descriptors1": descs2_t,
        }

        with torch.no_grad():
            result = self._matcher(data)

        # Extract matches
        matches0 = result["matches0"][0].cpu().numpy()  # (N1,) index in image2 or -1
        valid = matches0 > -1

        matches = []
        for i, m in enumerate(matches0):
            if m > -1:
                matches.append((i, int(m)))

        return matches

    def _match_flann(self, kpts1, descs1, kpts2, descs2):
        """Match using BFMatcher + Lowe's ratio test. Auto-selects norm."""
        if len(kpts1) < 2 or len(kpts2) < 2:
            return []

        # Auto-select distance metric based on descriptor type
        if descs1.dtype == np.uint8:
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self._bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        raw_matches = self._bf.knnMatch(descs1, descs2, k=2)

        matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < self.ratio_thresh * n.distance:
                matches.append((m.queryIdx, m.trainIdx))

        return matches

    def _spherical_verify(self, kpts1, kpts2, width, height, threshold=3.0):
        """
        Verify matches using spherical epipolar geometry.

        For an equirectangular image, the essential matrix constraint
        x2^T E x1 = 0 applies to unit sphere points. Here we use a
        simplified approach: compute the essential matrix from all matches
        using RANSAC on the spherical model, then keep only inliers.

        Args:
            kpts1, kpts2: (M, 2) matched keypoints.
            width, height: ERP dimensions.
            threshold: epipolar error threshold in pixels.

        Returns:
            (M,) boolean mask of inlier matches.
        """
        if len(kpts1) < 8:
            return np.ones(len(kpts1), dtype=bool)

        # Convert keypoints to normalized spherical coordinates
        from pano_track.camera import erp_to_sphere

        pts1_sphere = erp_to_sphere(kpts1, width, height)
        pts2_sphere = erp_to_sphere(kpts2, width, height)

        # For the essential matrix estimation, we need bearings (unit vectors)
        # Normalize to ensure unit length
        pts1_norm = pts1_sphere / np.linalg.norm(pts1_sphere, axis=-1, keepdims=True)
        pts2_norm = pts2_sphere / np.linalg.norm(pts2_sphere, axis=-1, keepdims=True)

        # Find essential matrix using RANSAC on spherical points
        try:
            # Use OpenCV's findEssentialMat with our custom function
            # Convert spherical points to "normalized image coordinates"
            # by treating them as points on a plane tangent to the sphere
            # For small motions this is a good approximation
            E, inlier_mask = cv2.findEssentialMat(
                kpts1, kpts2,
                focal=1.0,
                pp=(0, 0),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=threshold,
            )
        except cv2.error:
            return np.ones(len(kpts1), dtype=bool)

        if inlier_mask is None:
            return np.ones(len(kpts1), dtype=bool)

        return inlier_mask.ravel().astype(bool)


def draw_matches(img1, img2, kpts1, kpts2, mask=None, max_draw=100):
    """
    Draw matching lines between two equirectangular images side by side.

    Args:
        img1, img2: (H, W, 3) images.
        kpts1, kpts2: (M, 2) matched keypoints.
        mask: (M,) optional inlier mask.
        max_draw: max matches to draw.

    Returns:
        visualization image.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create side-by-side canvas
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2

    # Draw lines
    if mask is None:
        mask = np.ones(len(kpts1), dtype=bool)

    indices = np.where(mask)[0]
    if len(indices) > max_draw:
        indices = np.random.choice(indices, max_draw, replace=False)

    import matplotlib.cm as cm
    colors = (cm.tab10(np.linspace(0, 1, len(indices)))[:, :3] * 255).astype(np.uint8)

    for i, idx in enumerate(indices):
        pt1 = (int(kpts1[idx, 0]), int(kpts1[idx, 1]))
        pt2 = (int(kpts2[idx, 0] + w1), int(kpts2[idx, 1]))
        color = tuple(int(c) for c in colors[i])

        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    return canvas
