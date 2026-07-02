"""
Feature extraction for equirectangular panoramic images.

Supports two backends:
  - SuperPoint (deep learning, recommended): strong features, rotation-invariant
  - ORB (traditional, fallback): no GPU needed, fast

For SuperPoint, we use Kornia's implementation which handles ERP images well
when combined with spherical geometry verification in the matching stage.
"""

import numpy as np
import cv2


class FeatureExtractor:
    """Unified feature extraction interface."""

    def __init__(self, backend="superpoint", max_keypoints=1024, device="cpu"):
        """
        Args:
            backend: "superpoint" or "orb"
            max_keypoints: maximum keypoints to retain.
            device: "cpu" or "cuda" (for superpoint).
        """
        self.backend = backend
        self.max_keypoints = max_keypoints
        self.device = device
        self._model = None

        if backend == "superpoint":
            self._init_superpoint()
        elif backend == "orb":
            self._init_orb()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_superpoint(self):
        """Initialize SuperPoint via Kornia."""
        try:
            import torch
            import kornia
            from kornia.feature import SuperPoint

            self._model = SuperPoint(pretrained=True).to(self.device).eval()
            self._torch = torch
            print(f"SuperPoint initialized on {self.device}")
        except ImportError as e:
            print(f"SuperPoint not available ({e}), falling back to ORB")
            self.backend = "orb"
            self._init_orb()

    def _init_orb(self):
        """Initialize OpenCV ORB detector."""
        self._orb = cv2.ORB_create(
            nfeatures=self.max_keypoints,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10,
        )

    def extract(self, image):
        """
        Extract keypoints and descriptors from an equirectangular image.

        Args:
            image: (H, W, 3) uint8 RGB equirectangular image.

        Returns:
            keypoints: (N, 2) array of (u, v) pixel coordinates.
            descriptors: (N, D) array of descriptors.
        """
        if self.backend == "superpoint":
            return self._extract_superpoint(image)
        else:
            return self._extract_orb(image)

    def _extract_superpoint(self, image):
        """Extract features using SuperPoint."""
        import torch

        h, w = image.shape[:2]

        # Convert to grayscale tensor [1, 1, H, W]
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model({"image": tensor})

        keypoints = output["keypoints"][0].cpu().numpy()  # (N, 2) in (x, y) format
        descriptors = output["descriptors"][0].cpu().numpy()  # (N, 256)
        scores = output["scores"][0].cpu().numpy()  # (N,)

        # Limit keypoints
        if len(keypoints) > self.max_keypoints:
            idx = np.argsort(scores)[::-1][:self.max_keypoints]
            keypoints = keypoints[idx]
            descriptors = descriptors[idx]

        # Convert from (x, y) to (u, v) — they're the same in pixel coords
        # but ensure they're in (col, row) = (u, v) format
        return keypoints, descriptors

    def _extract_orb(self, image):
        """Extract features using ORB."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        cv_kpts, descriptors = self._orb.detectAndCompute(gray, None)

        if cv_kpts is None or len(cv_kpts) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 32), dtype=np.uint8)

        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in cv_kpts], dtype=np.float32)
        return keypoints, descriptors


def extract_features_batch(images, backend="superpoint", max_keypoints=1024, device="cpu"):
    """
    Convenience function: extract features from multiple images.

    Args:
        images: list of (H, W, 3) uint8 images.
        backend: feature backend.
        max_keypoints: max keypoints per image.
        device: device for SuperPoint.

    Returns:
        list of (keypoints, descriptors) tuples.
    """
    extractor = FeatureExtractor(backend=backend, max_keypoints=max_keypoints, device=device)
    results = []
    for i, img in enumerate(images):
        kpts, descs = extractor.extract(img)
        results.append((kpts, descs))
        if (i + 1) % 10 == 0:
            print(f"\rFeatures: {i+1}/{len(images)}", end="", flush=True)
    if len(images) >= 10:
        print()
    return results
