"""
Panoramic Visual Homing — the "snapshot model" of insect navigation.

Biological inspiration: Ants and bees store panoramic snapshots at their nest.
To return home, they compare the current view with the stored snapshot and
move to minimize the visual difference ("image difference").

Algorithm overview:
  1. Store an equirectangular snapshot at the home location.
  2. At a query location, rotationally align the current view to the home view
     (via circular cross-correlation along the longitude axis).
  3. Compute a column-wise dissimilarity profile.
  4. The bearing with minimum dissimilarity points toward home.
  5. Optional: iterative gradient descent on image difference for refinement.

Reference:
  - Zeil et al., "The learning and maintenance of local vectors in desert ant navigation"
  - Franz & Möller, "Biomimetic robot navigation"
"""

import numpy as np
import cv2


class VisualHoming:
    """Panoramic visual homing using the snapshot model."""

    def __init__(self, width, height, n_azimuth_bins=360):
        """
        Args:
            width, height: ERP image dimensions.
            n_azimuth_bins: number of columns to use for signature matching.
        """
        self.width = width
        self.height = height
        self.n_bins = n_azimuth_bins
        self.home_image = None
        self.home_position = None
        self.home_signature = None

    def set_home(self, image, position=None):
        """
        Store the home snapshot.

        Args:
            image: (H, W, 3) equirectangular image at home location.
            position: (3,) optional home position (for evaluation).
        """
        self.home_image = image.copy()
        self.home_position = np.asarray(position, dtype=np.float32) if position is not None else None
        self.home_signature = self._compute_signature(image)

    def _compute_signature(self, image):
        """
        Compute a multi-band circular image signature.

        Splits the image into N horizontal bands and, for each azimuthal bin,
        computes:
          - Mean RGB (3 channels)
          - Horizontal gradient magnitude (1 channel) — captures edges

        Returns a (n_bins, 4 * n_bands) descriptor that preserves both
        horizontal bearing information AND vertical scene structure.

        Multiple bands are critical for disambiguating directions in
        corridor-like scenes where left/right walls look similar but
        the floor and ceiling provide vertical cues.
        """
        h, w = image.shape[:2]
        n_bands = 4
        band_height = h // n_bands
        bin_width = w / self.n_bins

        # Feature per band: [R_mean, G_mean, B_mean, grad_mag]
        feat_dim = 4 * n_bands
        sig = np.zeros((self.n_bins, feat_dim), dtype=np.float32)

        # Pre-compute horizontal gradients
        gray = image.mean(axis=-1).astype(np.float32)
        grad_x = np.abs(np.diff(gray, axis=1, append=gray[:, :1]))

        for band in range(n_bands):
            r_start = band * band_height
            r_end = (band + 1) * band_height if band < n_bands - 1 else h
            offset = band * 4

            for i in range(self.n_bins):
                c_start = int(i * bin_width)
                c_end = int((i + 1) * bin_width)

                patch = image[r_start:r_end, c_start:c_end]
                grad_patch = grad_x[r_start:r_end, c_start:c_end]

                sig[i, offset + 0] = patch[:, :, 0].mean()  # R
                sig[i, offset + 1] = patch[:, :, 1].mean()  # G
                sig[i, offset + 2] = patch[:, :, 2].mean()  # B
                sig[i, offset + 3] = grad_patch.mean()       # edge strength

        # Standardize each feature dimension independently
        sig_mean = sig.mean(axis=0, keepdims=True)
        sig_std = sig.std(axis=0, keepdims=True) + 1e-6
        sig = (sig - sig_mean) / sig_std

        return sig

    def find_rotation(self, current_signature):
        """
        Find the relative rotation between home and current view.

        For equirectangular images, a camera rotation around the vertical
        (Y) axis corresponds to a circular shift along the longitude (width)
        axis. We find the shift that maximizes cross-correlation.

        Args:
            current_signature: signature of current view.

        Returns:
            best_shift_bins: circular shift amount in bins.
            best_correlation: peak correlation value.
        """
        best_shift = 0
        best_corr = -np.inf

        for shift in range(self.n_bins):
            shifted = np.roll(current_signature, shift, axis=0)
            # Sum of element-wise products = correlation
            corr = np.sum(self.home_signature * shifted)
            if corr > best_corr:
                best_corr = corr
                best_shift = shift

        return best_shift, best_corr

    def compute_dissimilarity_profile(self, current_signature, rotation_shift):
        """
        Compute the column-wise dissimilarity between aligned signatures.

        After rotational alignment, compute the per-column difference.
        The bearing with the minimum difference is the home direction.

        Args:
            current_signature: signature of current view.
            rotation_shift: optimal circular shift for alignment (in bins).

        Returns:
            dissimilarity: (n_bins,) array of per-column differences.
        """
        aligned = np.roll(current_signature, rotation_shift, axis=0)
        diff = np.linalg.norm(self.home_signature - aligned, axis=-1)
        return diff

    def estimate_home_bearing(self, image):
        """
        Estimate the direction toward home.

        Args:
            image: (H, W, 3) equirectangular image at current position.

        Returns:
            bearing_deg: bearing angle toward home (0 = forward/+X, 90 = right/+Y).
            confidence: correlation strength (higher = more confident).
            rotation_deg: relative rotation that aligns the views.
            dissimilarity: (n_bins,) per-column difference profile.
        """
        if self.home_signature is None:
            raise ValueError("Home not set. Call set_home() first.")

        current_sig = self._compute_signature(image)

        # Step 1: Find relative rotation
        rot_shift, confidence = self.find_rotation(current_sig)

        # Step 2: Compute dissimilarity profile
        dissimilarity = self.compute_dissimilarity_profile(current_sig, rot_shift)

        # Step 3: The bin with MINIMUM dissimilarity points TOWARD home
        # (because looking toward home, the scene changes the least)
        home_bin = np.argmin(dissimilarity)

        # Convert bin index to bearing angle
        # Bin 0 = looking along +X (lon=0), increasing CCW
        bearing_deg = (home_bin / self.n_bins) * 360.0
        if bearing_deg > 180:
            bearing_deg -= 360  # convert to [-180, 180]

        rotation_deg = (rot_shift / self.n_bins) * 360.0
        if rotation_deg > 180:
            rotation_deg -= 360

        return bearing_deg, confidence, rotation_deg, dissimilarity

    def compute_home_vector(self, image):
        """
        Compute the 2D home vector (on the ground plane).

        Returns a unit vector in the XZ plane pointing toward home.

        Args:
            image: (H, W, 3) equirectangular image.

        Returns:
            home_vec_2d: (2,) unit vector (dx, dz) pointing toward home.
            angle_deg: bearing angle.
            confidence: match confidence.
        """
        bearing_deg, confidence, rot_deg, dissim = self.estimate_home_bearing(image)

        # Convert bearing to 2D direction
        # bearing=0 means home is straight ahead (+X)
        # bearing>0 means home is to the right (+Z for our coordinates)
        angle_rad = np.deg2rad(bearing_deg)
        dx = np.cos(angle_rad)
        dz = -np.sin(angle_rad)  # negative because +Z is "forward-right"
        home_vec = np.array([dx, dz], dtype=np.float32)

        return home_vec, bearing_deg, confidence

    def home_vector_error(self, estimated_vec, true_vec):
        """
        Compute angular error between estimated and true home vectors.

        Args:
            estimated_vec: (2,) estimated home direction.
            true_vec: (2,) ground-truth home direction.

        Returns:
            error_deg: angular error in degrees.
        """
        e_norm = estimated_vec / (np.linalg.norm(estimated_vec) + 1e-10)
        t_norm = true_vec / (np.linalg.norm(true_vec) + 1e-10)
        cos_angle = np.clip(np.dot(e_norm, t_norm), -1, 1)
        return np.rad2deg(np.arccos(cos_angle))

    def image_difference_field(self, image, n_samples=20, step_size=0.5):
        """
        Compute an approximate "image difference field" around the current
        position by simulating small camera motions.

        This is the core of the gradient-descent homing approach:
          1. At current position, predict what the home image would look like
             if we moved in various directions
          2. The direction that MINIMIZES the predicted difference IS home

        Simplified version: use the dissimilarity profile to estimate
        the home direction directly (already done in estimate_home_bearing).

        Args:
            image: current equirectangular image.
            n_samples: not used in simple version.
            step_size: not used in simple version.

        Returns:
            home_vec: (2,) unit vector toward home.
        """
        # For the simple version, the column-wise dissimilarity directly
        # gives us the home direction — no need for iterative search
        return self.compute_home_vector(image)


def run_homing_experiment(home_image, query_images, query_positions, home_position):
    """
    Run visual homing experiment: for each query image, estimate home vector
    and compare with ground truth.

    Args:
        home_image: (H, W, 3) equirectangular home snapshot.
        query_images: list of (H, W, 3) query images.
        query_positions: (N, 3) ground-truth query positions.
        home_position: (3,) ground-truth home position.

    Returns:
        results: list of dicts with per-query evaluation.
    """
    h, w = home_image.shape[:2]
    homing = VisualHoming(w, h, n_azimuth_bins=360)
    homing.set_home(home_image, home_position)

    results = []
    for i, (img, pos) in enumerate(zip(query_images, query_positions)):
        # Ground-truth home vector
        true_vec_3d = home_position[:3] - pos[:3]
        true_vec_2d = true_vec_3d[[0, 2]]  # XZ plane
        true_vec_2d = true_vec_2d / (np.linalg.norm(true_vec_2d) + 1e-10)

        # Estimated home vector
        est_vec, bearing_deg, confidence = homing.compute_home_vector(img)
        error_deg = homing.home_vector_error(est_vec, true_vec_2d)

        results.append({
            "query_id": i,
            "position": pos.tolist(),
            "estimated_home_bearing_deg": bearing_deg,
            "true_home_bearing_deg": np.rad2deg(np.arctan2(true_vec_2d[1], true_vec_2d[0])),
            "bearing_error_deg": error_deg,
            "confidence": float(confidence),
            "distance_to_home": float(np.linalg.norm(true_vec_3d)),
        })

        print(f"  Query {i:2d}: dist={np.linalg.norm(true_vec_3d):.1f}m, "
              f"bearing_err={error_deg:.1f}°, conf={confidence:.2f}")

    return results
