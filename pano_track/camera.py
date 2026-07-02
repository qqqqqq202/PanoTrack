"""
Spherical camera model for equirectangular (ERP) panoramic images.

Equirectangular projection maps:
  - u (column)  → longitude λ ∈ [-π, π]
  - v (row)     → latitude  φ ∈ [-π/2, π/2]

Unit sphere point for pixel (u, v):
  X = cos(φ) cos(λ)
  Y = cos(φ) sin(λ)
  Z = sin(φ)

This module handles conversions between ERP images and the unit sphere,
which is the foundation for all spherical geometry operations.
"""

import numpy as np


def erp_to_sphere(uv, width, height):
    """
    Convert equirectangular pixel coordinates to unit sphere 3D points.

    Args:
        uv: (N, 2) or (H, W, 2) array of (u, v) pixel coordinates.
        width, height: image dimensions.

    Returns:
        (N, 3) or (H, W, 3) array of unit sphere points (X, Y, Z).
    """
    uv = np.asarray(uv, dtype=np.float64)
    # Longitude: u → [-π, π]
    lon = (uv[..., 0] / width) * 2.0 * np.pi - np.pi
    # Latitude: v → [-π/2, π/2]
    lat = (uv[..., 1] / height) * np.pi - np.pi / 2.0

    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def sphere_to_erp(points, width, height):
    """
    Convert unit sphere 3D points back to equirectangular pixel coordinates.

    Args:
        points: (N, 3) or (H, W, 3) array of unit sphere points.
        width, height: image dimensions.

    Returns:
        (N, 2) or (H, W, 2) array of (u, v) pixel coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]

    # Normalize to unit sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x / norm, y / norm, z / norm

    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)

    u = (lon + np.pi) / (2.0 * np.pi) * width
    v = (lat + np.pi / 2.0) / np.pi * height

    return np.stack([u, v], axis=-1)


def lift_keypoints_to_sphere(kpts, width, height):
    """
    Lift 2D keypoints from ERP image to unit sphere.

    Args:
        kpts: (N, 2) array of (u, v) keypoint coordinates.
        width, height: ERP image dimensions.

    Returns:
        (N, 3) array of unit sphere points.
    """
    return erp_to_sphere(kpts, width, height)


def angular_distance(p1, p2):
    """
    Compute angular (great-circle) distance between two points on unit sphere.

    Args:
        p1, p2: (N, 3) arrays of unit sphere points.

    Returns:
        (N,) array of angular distances in radians.
    """
    # cos(angle) = dot product
    cos_angle = np.clip(np.sum(p1 * p2, axis=-1), -1.0, 1.0)
    return np.arccos(cos_angle)


def essential_matrix_spherical_error(E, pts1_sphere, pts2_sphere):
    """
    Compute epipolar error on the unit sphere.

    For spherical camera model, the essential matrix constraint is:
    x2^T · E · x1 = 0
    where x1, x2 are unit sphere points.

    Args:
        E: (3, 3) essential matrix.
        pts1_sphere: (N, 3) points from first frame on unit sphere.
        pts2_sphere: (N, 3) points from second frame on unit sphere.

    Returns:
        (N,) array of epipolar errors.
    """
    Ex1 = E @ pts1_sphere.T  # (3, N)
    errors = np.abs(np.sum(pts2_sphere * Ex1.T, axis=-1))
    return errors
