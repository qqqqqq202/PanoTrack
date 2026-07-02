"""
Equirectangular (360°) panoramic rendering via ray casting.

Renders a trimesh scene from any camera position as an equirectangular image.
Each pixel's ray direction is computed from spherical coordinates, then cast
into the scene to find the first intersection. The color at the hit point is
interpolated from the triangle's vertex colors.

Output: equirectangular RGB image (H × W × 3), standard 360° format.
"""

import numpy as np
import trimesh


def _build_ray_grid(width, height):
    """
    Pre-compute unit-sphere ray directions for every ERP pixel.

    Args:
        width, height: ERP image dimensions.

    Returns:
        ray_dirs: (height, width, 3) unit direction vectors.
    """
    u = np.arange(width)
    v = np.arange(height)
    u_grid, v_grid = np.meshgrid(u, v)  # (H, W)

    lon = (u_grid / width) * 2.0 * np.pi - np.pi          # [-pi, pi]
    lat = (v_grid / height) * np.pi - np.pi / 2.0          # [-pi/2, pi/2]

    cos_lat = np.cos(lat)
    dx = cos_lat * np.cos(lon)
    dy = cos_lat * np.sin(lon)
    dz = np.sin(lat)

    ray_dirs = np.stack([dx, dy, dz], axis=-1).astype(np.float32)
    return ray_dirs


def _barycentric_interpolate_colors(locations, index_tri, mesh):
    """
    Interpolate vertex colors at hit points using barycentric coordinates.

    Args:
        locations: (N, 3) world-space hit points.
        index_tri: (N,) triangle indices that were hit.
        mesh: trimesh.Trimesh with vertex_colors.

    Returns:
        colors: (N, 3) uint8 RGB colors.
    """
    if len(locations) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    faces = mesh.faces[index_tri]  # (N, 3) vertex indices
    v0 = mesh.vertices[faces[:, 0]]
    v1 = mesh.vertices[faces[:, 1]]
    v2 = mesh.vertices[faces[:, 2]]

    # Compute barycentric coordinates
    # For point P on triangle ABC: P = u*A + v*B + w*C, with u+v+w=1
    v0p = locations - v0
    v1p = locations - v1
    v2p = locations - v2

    # Area of triangle PBC (proportional to weight for A)
    d0 = np.linalg.norm(np.cross(v1p, v2p), axis=-1)
    # Area of triangle PCA (proportional to weight for B)
    d1 = np.linalg.norm(np.cross(v2p, v0p), axis=-1)
    # Area of triangle PAB (proportional to weight for C)
    d2 = np.linalg.norm(np.cross(v0p, v1p), axis=-1)

    d_sum = d0 + d1 + d2
    # Avoid division by zero
    mask = d_sum > 1e-12
    d_sum = np.where(mask, d_sum, 1.0)

    w0 = d0 / d_sum  # weight for v0
    w1 = d1 / d_sum  # weight for v1
    w2 = d2 / d_sum  # weight for v2

    # Get vertex colors
    vc = mesh.visual.vertex_colors
    c0 = vc[faces[:, 0]].astype(np.float32)  # (N, 4)
    c1 = vc[faces[:, 1]].astype(np.float32)
    c2 = vc[faces[:, 2]].astype(np.float32)

    colors = (w0[:, None] * c0 + w1[:, None] * c1 + w2[:, None] * c2)
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    # Set color to black for degenerate hits
    colors[~mask] = [0, 0, 0, 255]

    return colors[:, :3]  # RGB only


def render_equirectangular(scene, camera_position, rotation=None,
                           width=512, height=256, sky_color=(135, 206, 235)):
    """
    Render an equirectangular panorama from a camera position.

    Args:
        scene: trimesh.Trimesh scene to render.
        camera_position: (3,) array (x, y, z) of camera position.
        rotation: (3, 3) rotation matrix (world-from-camera), or None for default +X.
        width, height: output image dimensions.
        sky_color: (R, G, B) for rays that miss the scene.

    Returns:
        image: (height, width, 3) uint8 RGB equirectangular image.
    """
    ray_dirs = _build_ray_grid(width, height)  # (H, W, 3)
    ray_dirs_flat = ray_dirs.reshape(-1, 3)     # (N, 3)

    # Apply camera rotation: d_world = R @ d_camera
    if rotation is not None:
        R = np.asarray(rotation, dtype=np.float32)
        ray_dirs_flat = (R @ ray_dirs_flat.T).T

    # Camera position for all rays
    ray_origins = np.tile(np.asarray(camera_position, dtype=np.float32),
                          (len(ray_dirs_flat), 1))

    # Cast all rays at once
    locations, index_ray, index_tri = scene.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs_flat,
        multiple_hits=False,
    )

    # Initialize image with sky color
    image_flat = np.full((len(ray_dirs_flat), 3), sky_color, dtype=np.uint8)

    if len(locations) > 0:
        colors = _barycentric_interpolate_colors(locations, index_tri, scene)
        image_flat[index_ray] = colors

    image = image_flat.reshape(height, width, 3)
    return image


def render_dataset(scene, positions, rotations, width=512, height=256,
                   verbose=True):
    """
    Render a sequence of equirectangular panoramas along a trajectory.

    Note: rotations are currently simplified — the rendering assumes the
    camera looks along +X by default. For proper view rotation, we'd rotate
    the ray directions by the camera's rotation matrix.

    Args:
        scene: trimesh.Trimesh scene.
        positions: (N, 3) camera positions.
        rotations: (N, 3, 3) camera rotation matrices (world-from-camera).
        width, height: output resolution.
        verbose: print progress.

    Returns:
        images: list of (height, width, 3) uint8 numpy arrays.
    """
    images = []
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        if verbose:
            print(f"\rRendering {i+1}/{len(positions)} — pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                  end="", flush=True)
        img = render_equirectangular(scene, pos, rot, width, height)
        images.append(img)
    if verbose:
        print()
    return images


if __name__ == "__main__":
    from pano_track.scene import create_corridor_scene, sample_camera_path
    import time

    print("Creating scene...")
    scene = create_corridor_scene()

    print("Generating camera path...")
    pos, rot = sample_camera_path(5)  # just 5 frames for testing

    print("Rendering test frame...")
    t0 = time.time()
    img = render_equirectangular(scene, pos[0], width=256, height=128)
    elapsed = time.time() - t0
    print(f"Rendered 256×128 in {elapsed:.1f}s — shape={img.shape}")
    print(f"Estimated time for 50 frames at 512×256: ~{elapsed * 4 * 50 / 60:.0f} min")
