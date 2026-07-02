"""
Pose estimation for spherical (panoramic) cameras.

For spherical cameras, points are unit bearing vectors (x, y, z) on S².
The essential matrix constraint is identical to perspective cameras:
    x2^T · E · x1 = 0
but x1, x2 are unit sphere points rather than normalized image coordinates.

This module implements:
  - Spherical 8-point algorithm (direct on S²)
  - RANSAC for robust estimation
  - Essential matrix decomposition with chirality check
"""

import numpy as np


def estimate_essential_spherical_8pt(bearings1, bearings2):
    """
    Estimate essential matrix from 8+ spherical point correspondences.

    The epipolar constraint on the unit sphere:
        x2^T · E · x1 = 0
    where x1, x2 ∈ S² are unit bearing vectors.

    Expanding: x1·x2·E11 + x1·y2·E12 + ... + z1·z2·E33 = 0

    Args:
        bearings1: (N, 3) unit bearing vectors from first view.
        bearings2: (N, 3) unit bearing vectors from second view.

    Returns:
        E: (3, 3) essential matrix.
    """
    b1 = np.asarray(bearings1, dtype=np.float64)
    b2 = np.asarray(bearings2, dtype=np.float64)

    # Normalize to unit vectors
    b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-10)
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-10)

    x1, y1, z1 = b1[:, 0], b1[:, 1], b1[:, 2]
    x2, y2, z2 = b2[:, 0], b2[:, 1], b2[:, 2]

    # Design matrix: each row = [x1*x2, x1*y2, x1*z2, y1*x2, ..., z1*z2]
    A = np.column_stack([
        x1 * x2, x1 * y2, x1 * z2,
        y1 * x2, y1 * y2, y1 * z2,
        z1 * x2, z1 * y2, z1 * z2,
    ])

    # Solve Ae = 0 subject to ||e|| = 1
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    E_vec = Vt[-1]  # last row of Vt = smallest singular vector
    E = E_vec.reshape(3, 3)

    # Enforce essential matrix constraint: rank(E) = 2
    U, S, Vt2 = np.linalg.svd(E)
    S[2] = 0.0
    S_mean = (S[0] + S[1]) / 2.0
    if S_mean > 1e-10:
        S[0] = S_mean
        S[1] = S_mean
    E_enforced = U @ np.diag(S) @ Vt2

    return E_enforced


def compute_epipolar_errors(E, bearings1, bearings2):
    """
    Compute Sampson-like epipolar errors for spherical points.

    For spherical model, the algebraic error is:
        |x2^T · E · x1|

    Args:
        E: (3, 3) essential matrix.
        bearings1: (N, 3) unit bearing vectors.
        bearings2: (N, 3) unit bearing vectors.

    Returns:
        errors: (N,) epipolar errors.
    """
    b1 = np.asarray(bearings1, dtype=np.float64)
    b2 = np.asarray(bearings2, dtype=np.float64)

    Ex1 = E @ b1.T  # (3, N)
    errors = np.abs(np.sum(b2 * Ex1.T, axis=-1))

    return errors


def estimate_essential_ransac(kpts1, kpts2, width, height,
                               n_iterations=500, threshold=0.005):
    """
    Robust essential matrix estimation using RANSAC + spherical 8-point.

    Args:
        kpts1, kpts2: (N, 2) matched keypoints in ERP pixel coordinates.
        width, height: ERP image dimensions.
        n_iterations: RANSAC iterations.
        threshold: inlier threshold on spherical epipolar error.

    Returns:
        E: (3, 3) essential matrix.
        inlier_mask: (N,) boolean array.
        n_inliers: number of inliers.
    """
    from pano_track.camera import erp_to_sphere

    N = len(kpts1)
    if N < 8:
        return np.eye(3), np.ones(N, dtype=bool), 0

    # Lift all keypoints to unit sphere
    pts1 = erp_to_sphere(kpts1, width, height)
    pts2 = erp_to_sphere(kpts2, width, height)

    # RANSAC
    best_E = np.eye(3)
    best_inliers = np.zeros(N, dtype=bool)
    best_n_inliers = 0

    rng = np.random.RandomState(42)

    for _ in range(n_iterations):
        # Sample 8 random points
        idx = rng.choice(N, 8, replace=False)

        try:
            E = estimate_essential_spherical_8pt(pts1[idx], pts2[idx])
        except np.linalg.LinAlgError:
            continue

        # Count inliers
        errors = compute_epipolar_errors(E, pts1, pts2)
        inliers = errors < threshold
        n_inliers = inliers.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_E = E
            best_inliers = inliers

    # Refit E on all inliers
    if best_n_inliers >= 8:
        try:
            best_E = estimate_essential_spherical_8pt(
                pts1[best_inliers], pts2[best_inliers]
            )
        except np.linalg.LinAlgError:
            pass

    return best_E, best_inliers, best_n_inliers


def decompose_essential(E, bearings1, bearings2):
    """
    Decompose essential matrix into (R, t) and select the correct
    solution via chirality (positive depth) check.

    E = [t]× · R

    Args:
        E: (3, 3) essential matrix.
        bearings1: (N, 3) unit bearing vectors from view 1.
        bearings2: (N, 3) unit bearing vectors from view 2.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation direction (unit vector, up to scale).
        n_front: number of points with positive depth in both views.
    """
    # SVD decomposition of E
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # Two possible rotations
    W = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Translation (up to scale) — third column of U
    t = U[:, 2]

    # Ensure proper rotation matrices
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Test 4 solutions: (R1, t), (R1, -t), (R2, t), (R2, -t)
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]

    best_score = -np.inf
    best_R, best_t = R1, t
    best_count = 0

    for R_cand, t_cand in solutions:
        count = _count_points_in_front_spherical(
            bearings1, bearings2, R_cand, t_cand
        )

        # Score: front_count + epipolar consistency
        # Higher is better. We weight front_count heavily but also
        # consider the median epipolar error (lower error = better).
        t_norm = t_cand / (np.linalg.norm(t_cand) + 1e-10)
        E_cand = np.cross(np.eye(3), t_norm) @ R_cand

        # Compute epipolar errors for all bearings
        Ex = E_cand @ bearings1.T
        epi_errs = np.abs(np.sum(bearings2 * Ex.T, axis=-1))

        # Score: combine front ratio with negative median epipolar error
        front_ratio = count / max(len(bearings1), 1)
        median_epi = np.median(epi_errs)
        score = front_ratio * 100 - median_epi * 10  # weight front more

        if score > best_score:
            best_score = score
            best_R = R_cand
            best_t = t_cand
            best_count = count

    # Normalize translation
    best_t = best_t / (np.linalg.norm(best_t) + 1e-10)

    return best_R, best_t, best_count


def _count_points_in_front_spherical(bearings1, bearings2, R, t):
    """
    Count points with positive depth in both cameras.

    Uses the cross-product method for direct depth computation:
    From X_cam2 = R @ X_cam1 + t and X_cam1 = λ1*b1, X_cam2 = λ2*b2:
        λ1 = |b2 × t| / |b2 × (R @ b1)|  (with sign from cross product)
        λ2 = |b1 × R^T @ t| / |b1 × (R^T @ b2)|

    This avoids the ill-conditioned least-squares that fails when bearings
    are nearly parallel (distant features in corridors).

    Args:
        bearings1, bearings2: (N, 3) unit bearing vectors.
        R: (3, 3) rotation from camera 1 to camera 2.
        t: (3,) translation from camera 1 to camera 2.

    Returns:
        count: number of points with positive depth in both cameras.
    """
    b1 = np.asarray(bearings1, dtype=np.float64)
    b2 = np.asarray(bearings2, dtype=np.float64)
    t_vec = np.asarray(t, dtype=np.float64).ravel()

    count = 0
    for i in range(len(b1)):
        # Cross product: b2 × (R @ b1)
        Rb1 = R @ b1[i]
        cross_b2_Rb1 = np.cross(b2[i], Rb1)
        denom1 = np.linalg.norm(cross_b2_Rb1)

        # Cross product: b2 × t
        cross_b2_t = np.cross(b2[i], t_vec)

        # Skip if bearings are nearly parallel (distant points)
        if denom1 < 1e-6:
            continue

        # λ1: depth in camera 1
        # Sign: if cross_b2_Rb1 and cross_b2_t point in SAME direction, λ1 > 0
        lambda1_sign = np.sign(np.dot(cross_b2_Rb1, cross_b2_t))

        # For camera 2:
        RTb2 = R.T @ b2[i]
        RTt = R.T @ t_vec
        cross_b1_RTb2 = np.cross(b1[i], RTb2)
        cross_b1_RTt = np.cross(b1[i], RTt)
        denom2 = np.linalg.norm(cross_b1_RTb2)

        if denom2 < 1e-6:
            continue

        lambda2_sign = np.sign(np.dot(cross_b1_RTb2, cross_b1_RTt))

        if lambda1_sign > 0 and lambda2_sign > 0:
            count += 1

    return count


def refine_pose_nonlinear(R_init, t_init, bearings1, bearings2, max_iter=50):
    """
    Nonlinear refinement of relative pose using all inlier bearings.

    Minimizes the algebraic epipolar error over (R, t) directly on the sphere:
        min_{R,t}  sum_i (b2_i^T · [t]_x · R · b1_i)^2

    Parameterized as:
      Rotation: axis-angle (3 params) → R = exp([ω]_x)
      Translation: unit sphere lat/lon (2 params) → t = (cosφ cosλ, cosφ sinλ, sinφ)

    Args:
        R_init: (3, 3) initial rotation matrix.
        t_init: (3,) initial translation (unit vector).
        bearings1, bearings2: (N, 3) inlier bearings on unit sphere.
        max_iter: max LM iterations.

    Returns:
        R_refined: (3, 3) refined rotation.
        t_refined: (3,) refined translation (unit vector).
        cost: final mean squared error.
    """
    from scipy.optimize import least_squares
    from scipy.linalg import expm

    b1 = np.asarray(bearings1, dtype=np.float64)
    b2 = np.asarray(bearings2, dtype=np.float64)

    # Initial parameters: axis-angle for R + spherical coords for t
    def R_to_aa(R):
        """Rotation matrix → axis-angle (compact, 3 values)."""
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-10:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
        return angle * axis

    def aa_to_R(aa):
        """Axis-angle → rotation matrix."""
        angle = np.linalg.norm(aa)
        if angle < 1e-10:
            return np.eye(3)
        axis = aa / angle
        return expm(np.cross(np.eye(3), axis * angle))

    def t_to_sph(t):
        """Unit vector → spherical coordinates (lat, lon)."""
        t = t / (np.linalg.norm(t) + 1e-10)
        lat = np.arcsin(np.clip(t[2], -1, 1))       # latitude
        lon = np.arctan2(t[1], t[0])                 # longitude
        return np.array([lat, lon])

    def sph_to_t(sph):
        """Spherical coordinates → unit vector."""
        lat, lon = sph[0], sph[1]
        return np.array([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         np.sin(lat)])

    # Initial parameter vector: [aa_x, aa_y, aa_z, lat, lon]
    aa0 = R_to_aa(R_init)
    sph0 = t_to_sph(t_init)
    params0 = np.concatenate([aa0, sph0])

    def cost_fn(params):
        R = aa_to_R(params[:3])
        t = sph_to_t(params[3:5])
        # Epipolar error for all points: b2^T [t]_x R b1
        t_cross = np.array([[0, -t[2], t[1]],
                            [t[2], 0, -t[0]],
                            [-t[1], t[0], 0]])
        E = t_cross @ R
        errors = np.sum(b2 * (E @ b1.T).T, axis=-1)  # (N,)
        return errors

    result = least_squares(cost_fn, params0, method='lm',
                           max_nfev=max_iter, verbose=0)

    R_refined = aa_to_R(result.x[:3])
    t_refined = sph_to_t(result.x[3:5])

    return R_refined, t_refined, float(np.mean(np.abs(result.fun)))


def relative_pose_error(R_est, t_est, R_gt, t_gt):
    """
    Compute angular error between estimated and ground-truth relative pose.

    Args:
        R_est, R_gt: (3, 3) rotation matrices.
        t_est, t_gt: (3,) translation vectors.

    Returns:
        rot_error_deg: rotation error in degrees.
        trans_error_deg: translation direction error in degrees.
    """
    # Rotation error
    R_diff = R_est.T @ R_gt
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    rot_error = np.rad2deg(angle)

    # Translation direction error
    t_est_u = t_est / (np.linalg.norm(t_est) + 1e-10)
    t_gt_u = t_gt / (np.linalg.norm(t_gt) + 1e-10)
    cos_angle = np.clip(np.dot(t_est_u, t_gt_u), -1, 1)
    trans_error = np.rad2deg(np.arccos(cos_angle))

    return rot_error, trans_error
