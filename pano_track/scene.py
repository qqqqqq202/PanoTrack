"""
Procedural indoor 3D scene generation.

Creates a corridor + room layout with:
  - Checkerboard floor (strong corner features for SuperPoint)
  - Multi-colored walls with different textures
  - Furniture objects as visual landmarks
  - All geometry stored as a single trimesh object with vertex colors.

No external data download required — everything is procedural.
"""

import numpy as np
import trimesh


def _create_checkerboard_mesh(vertices_2d, height, tile_grid, colors_light, colors_dark):
    """
    Create a subdivided horizontal mesh with checkerboard vertex colors.

    Args:
        vertices_2d: (N, 2) array of (x, z) corner positions defining the polygon.
        height: Y coordinate for all vertices.
        tile_grid: (nx, nz) number of subdivisions along x and z.
        colors_light, colors_dark: RGBA uint8 arrays for checkerboard tiles.

    Returns:
        trimesh.Trimesh with vertex colors.
    """
    x_min, x_max = vertices_2d[:, 0].min(), vertices_2d[:, 0].max()
    z_min, z_max = vertices_2d[:, 1].min(), vertices_2d[:, 1].max()
    nx, nz = tile_grid

    dx = (x_max - x_min) / nx
    dz = (z_max - z_min) / nz

    verts = []
    face_list = []
    vert_colors = []

    # Generate grid vertices
    for i in range(nx + 1):
        for j in range(nz + 1):
            x = x_min + i * dx
            z = z_min + j * dz
            verts.append([x, height, z])

            # Determine if this vertex is in a light or dark tile
            ti = int(i / max(1, nx / 8))
            tj = int(j / max(1, nz / 8))
            c = colors_light if (ti + tj) % 2 == 0 else colors_dark
            vert_colors.append(c)

    # Generate quad faces (2 triangles per quad)
    stride = nz + 1
    for i in range(nx):
        for j in range(nz):
            v00 = i * stride + j
            v10 = (i + 1) * stride + j
            v11 = (i + 1) * stride + (j + 1)
            v01 = i * stride + (j + 1)
            face_list.append([v00, v10, v11])
            face_list.append([v00, v11, v01])

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(face_list, dtype=np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = np.array(vert_colors, dtype=np.uint8)
    return mesh


def _create_wall_mesh(x_start, z_start, x_end, z_end, height_bottom, height_top,
                      color, subdivisions=4):
    """
    Create a vertical wall with vertex colors.

    The wall extends along the line from (x_start, z_start) to (x_end, z_end),
    rising from height_bottom to height_top.
    """
    dx_total = x_end - x_start
    dz_total = z_end - z_start
    length = np.sqrt(dx_total**2 + dz_total**2)

    verts = []
    face_list = []
    vert_colors = []

    stride = 2 * (subdivisions + 1)  # 2 rows of vertices per subdivision

    for s in range(subdivisions + 1):
        t = s / subdivisions
        x = x_start + t * dx_total
        z = z_start + t * dz_total

        # Bottom vertex
        verts.append([x, height_bottom, z])
        vert_colors.append(color * 0.6)  # slightly darker at bottom

        # Top vertex
        verts.append([x, height_top, z])
        vert_colors.append(color)

    for s in range(subdivisions):
        b0 = s * 2       # bottom of segment s
        t0 = s * 2 + 1   # top of segment s
        b1 = (s + 1) * 2
        t1 = (s + 1) * 2 + 1

        face_list.append([b0, b1, t0])
        face_list.append([t0, b1, t1])

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(face_list, dtype=np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = np.array(vert_colors, dtype=np.uint8)
    return mesh


def _create_ceiling_mesh(x_min, x_max, z_min, z_max, height, color, tile_grid=(16, 16)):
    """Create a flat ceiling mesh with subtle checkerboard."""
    verts_2d = np.array([
        [x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]
    ])
    dark = (np.array(color) * 0.85).astype(np.uint8)
    return _create_checkerboard_mesh(verts_2d, height, tile_grid, color, dark)


def create_corridor_scene():
    """
    Create the complete indoor scene: corridor + room + furniture.

    Layout (top-down, meters):
        z=4  +------------------+
             |                  |
        z=2  |      ROOM        |
             |     (4×4)        |
        z=0  +--+---------+-----+
                | CORRIDOR |
        z=-2 +--+--(2×10)--+----+
             |                  |
        z=-4 +------------------+
             x=-6              x=6

    Returns:
        trimesh.Trimesh: Combined scene mesh with vertex colors.
    """
    meshes = []
    FLOOR_Y = 0.0
    CEILING_Y = 3.0

    # ── Floor ──────────────────────────────────────────────
    # Corridor floor: x=[-6, 6], z=[-1, 1]
    verts2d_corridor = np.array([[-6, -1], [6, -1], [6, 1], [-6, 1]], dtype=np.float32)
    light_gray = np.array([220, 215, 210, 255], dtype=np.uint8)
    dark_gray = np.array([50, 48, 45, 255], dtype=np.uint8)
    floor_corridor = _create_checkerboard_mesh(
        verts2d_corridor, FLOOR_Y, (24, 8), light_gray, dark_gray
    )
    meshes.append(floor_corridor)

    # Room floor: x=[6, 10], z=[-3, 3]
    verts2d_room = np.array([[6, -3], [10, -3], [10, 3], [6, 3]], dtype=np.float32)
    floor_room = _create_checkerboard_mesh(
        verts2d_room, FLOOR_Y, (16, 24), light_gray, dark_gray
    )
    meshes.append(floor_room)

    # ── Walls ──────────────────────────────────────────────
    wall_colors = {
        'left': np.array([180, 140, 100, 255], dtype=np.uint8),    # warm beige
        'right': np.array([140, 160, 180, 255], dtype=np.uint8),   # cool blue-gray
        'front': np.array([200, 190, 170, 255], dtype=np.uint8),   # cream
        'room_left': np.array([170, 150, 120, 255], dtype=np.uint8),
        'room_right': np.array([130, 150, 170, 255], dtype=np.uint8),
        'room_back': np.array([190, 180, 160, 255], dtype=np.uint8),
    }

    # Corridor left wall: x=-6 to x=6, z=-1
    meshes.append(_create_wall_mesh(-6, -1, 6, -1, FLOOR_Y, CEILING_Y,
                                     wall_colors['left']))
    # Corridor right wall: x=-6 to x=6, z=1
    meshes.append(_create_wall_mesh(-6, 1, 6, 1, FLOOR_Y, CEILING_Y,
                                     wall_colors['right']))
    # Room left wall: x=6 to x=10, z=-3
    meshes.append(_create_wall_mesh(6, -3, 10, -3, FLOOR_Y, CEILING_Y,
                                     wall_colors['room_left']))
    # Room right wall: x=6 to x=10, z=3
    meshes.append(_create_wall_mesh(6, 3, 10, 3, FLOOR_Y, CEILING_Y,
                                     wall_colors['room_right']))
    # Room back wall: x=10, z=-3 to z=3
    meshes.append(_create_wall_mesh(10, -3, 10, 3, FLOOR_Y, CEILING_Y,
                                     wall_colors['room_back']))
    # Room front walls (connecting corridor to room):
    #  upper: x=6, z=1 to z=3
    meshes.append(_create_wall_mesh(6, 1, 6, 3, FLOOR_Y, CEILING_Y,
                                     wall_colors['front']))
    #  lower: x=6, z=-3 to z=-1
    meshes.append(_create_wall_mesh(6, -3, 6, -1, FLOOR_Y, CEILING_Y,
                                     wall_colors['front']))

    # ── Ceiling ────────────────────────────────────────────
    ceiling_color = np.array([240, 238, 230, 255], dtype=np.uint8)
    # Corridor ceiling
    meshes.append(_create_ceiling_mesh(-6, 6, -1, 1, CEILING_Y, ceiling_color))
    # Room ceiling
    meshes.append(_create_ceiling_mesh(6, 10, -3, 3, CEILING_Y, ceiling_color))

    # ── Furniture / Landmarks ──────────────────────────────
    furniture = [
        # (x_center, z_center, w, h, d, color_name)
        # Corridor landmarks — placed at distinctive positions
        (-4, -0.4, 0.5, 1.4, 0.5, 'red'),           # tall red locker near entrance
        (-4, 0.4, 0.3, 0.6, 0.3, 'cyan'),            # small cyan box
        (-1, -0.3, 0.6, 0.9, 0.6, 'green'),          # green crate mid-corridor
        (-1, 0.3, 0.4, 1.1, 0.4, 'magenta'),         # magenta pillar
        (2, -0.3, 0.7, 1.0, 0.5, 'yellow'),          # yellow box
        (2, 0.3, 0.3, 0.5, 0.3, 'brown'),            # brown stool
        (5, -0.4, 0.4, 1.6, 0.4, 'blue'),            # tall blue pillar near doorway
        # Room landmarks
        (7, -2, 1.0, 1.5, 0.8, 'orange'),            # large orange crate
        (8, 2, 0.5, 0.7, 0.5, 'purple'),             # purple box
        (8.5, 1.5, 0.3, 1.8, 0.3, 'white'),          # tall white pillar (near room back)
        (9, -2.5, 0.6, 1.2, 0.6, 'lime'),            # lime green crate
    ]

    color_map = {
        'red': np.array([200, 50, 40, 255], dtype=np.uint8),
        'blue': np.array([40, 60, 190, 255], dtype=np.uint8),
        'green': np.array([50, 180, 60, 255], dtype=np.uint8),
        'yellow': np.array([210, 190, 40, 255], dtype=np.uint8),
        'orange': np.array([220, 130, 30, 255], dtype=np.uint8),
        'purple': np.array([140, 50, 180, 255], dtype=np.uint8),
        'white': np.array([230, 225, 220, 255], dtype=np.uint8),
        'cyan': np.array([30, 200, 200, 255], dtype=np.uint8),
        'magenta': np.array([200, 50, 150, 255], dtype=np.uint8),
        'brown': np.array([140, 100, 60, 255], dtype=np.uint8),
        'lime': np.array([150, 220, 50, 255], dtype=np.uint8),
    }

    for fx, fz, fw, fh, fd, cname in furniture:
        box = trimesh.creation.box(extents=[fw, fh, fd])
        box.apply_translation([fx, FLOOR_Y + fh / 2, fz])
        box.visual.vertex_colors = color_map[cname]
        meshes.append(box)

    # ── Directional wall markers (break left/right symmetry) ──
    # Place colored panels on walls at specific longitudes
    wall_panels = [
        # Left wall (z=-1): x_pos, w, h, color
        (-3, 1.0, 1.5, np.array([60, 180, 220, 255], dtype=np.uint8)),   # bright blue panel
        (1, 0.8, 1.2, np.array([220, 180, 40, 255], dtype=np.uint8)),    # gold panel
        (5, 1.2, 1.8, np.array([60, 220, 100, 255], dtype=np.uint8)),    # bright green panel
        # Right wall (z=1): x_pos, w, h, color
        (-1, 1.0, 1.5, np.array([220, 80, 60, 255], dtype=np.uint8)),    # bright red panel
        (3, 0.8, 1.2, np.array([180, 60, 200, 255], dtype=np.uint8)),    # purple panel
        # Room walls
        (7, 1.5, 1.5, np.array([240, 140, 40, 255], dtype=np.uint8)),    # orange panel (right wall of room)
    ]

    for wx, ww, wh, wcolor in wall_panels:
        # Determine which wall and position
        if wx <= 6:  # corridor walls
            for wz, wdir in [(-1.0, 'left'), (1.0, 'right')]:
                # Create thin colored panel on wall
                panel = trimesh.creation.box(extents=[ww, wh, 0.02])
                # Offset slightly from wall to avoid z-fighting
                z_offset = -0.99 if wdir == 'left' else 0.99
                panel.apply_translation([wx, FLOOR_Y + wh / 2 + 0.5, z_offset])
                panel.visual.vertex_colors = wcolor
                meshes.append(panel)
        else:  # room walls
            for wz, wdir in [(-3.0, 'left'), (3.0, 'right')]:
                panel = trimesh.creation.box(extents=[ww, wh, 0.02])
                z_offset = -2.99 if wdir == 'left' else 2.99
                panel.apply_translation([wx, FLOOR_Y + wh / 2 + 0.5, z_offset])
                panel.visual.vertex_colors = wcolor
                meshes.append(panel)

    # ── Combine ────────────────────────────────────────────
    scene = trimesh.util.concatenate(meshes)
    print(f"Scene created: {len(scene.vertices)} vertices, {len(scene.faces)} faces")
    return scene


def sample_camera_path(n_frames=50, seed=42):
    """
    Generate a 3D camera trajectory through the corridor and room.

    The path starts at the corridor entrance and walks through to the room,
    with slight lateral variation for realism.

    Args:
        n_frames: Number of frames along the path.
        seed: Random seed for reproducibility.

    Returns:
        positions: (n_frames, 3) array of (x, y, z) camera positions.
        rotations: (n_frames, 3, 3) rotation matrices (world-from-camera).
    """
    rng = np.random.RandomState(seed)

    # Define waypoints for the path
    waypoints = np.array([
        # (x, z, facing_angle_deg) — y is always 1.5m (eye level)
        [-5.0,  0.0,  0.0],    # entrance, facing +x
        [-2.0,  0.0,  0.0],    # mid corridor
        [ 1.0,  0.0,  0.0],    # further down
        [ 4.0,  0.0,  0.0],    # corridor end
        [ 6.0,  0.0, 30.0],    # entering room, slight right
        [ 7.0,  1.5, 45.0],    # turning into room
        [ 8.5,  2.0, 60.0],    # room back-right corner
        [ 8.5, -1.5, 120.0],   # sweep left
        [ 7.0, -2.5, 160.0],   # room front-left
    ])

    # Interpolate path with cubic spline
    n_waypoints = len(waypoints)
    t_waypoints = np.linspace(0, 1, n_waypoints)
    t_frames = np.linspace(0, 1, n_frames)

    from scipy.interpolate import CubicSpline

    positions = np.zeros((n_frames, 3), dtype=np.float32)
    positions[:, 1] = 1.5  # constant eye level

    cs_x = CubicSpline(t_waypoints, waypoints[:, 0])
    cs_z = CubicSpline(t_waypoints, waypoints[:, 1])
    cs_angle = CubicSpline(t_waypoints, waypoints[:, 2])

    positions[:, 0] = cs_x(t_frames)
    positions[:, 2] = cs_z(t_frames)
    facing_angles = np.deg2rad(cs_angle(t_frames))

    # Add slight sinusoidal lateral variation for realism
    positions[:, 2] += 0.15 * np.sin(t_frames * np.pi * 4 + rng.randn() * 0.1)

    # Build rotation matrices (camera looks along +x by default)
    rotations = np.zeros((n_frames, 3, 3), dtype=np.float32)
    for i in range(n_frames):
        angle = facing_angles[i]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        # Rotation around Y axis (world-from-camera)
        # Camera looks along +x when angle=0; angle rotates look direction rightward
        rotations[i] = np.array([
            [cos_a, 0, sin_a],
            [0,     1, 0     ],
            [-sin_a, 0, cos_a],
        ], dtype=np.float32)

    return positions, rotations


def sample_homing_viewpoints(n_viewpoints=15, seed=123):
    """
    Sample random camera positions for visual homing experiments.

    These are scattered throughout the scene and used as query/target pairs.

    Args:
        n_viewpoints: Number of viewpoints to sample.
        seed: Random seed.

    Returns:
        positions: (n_viewpoints, 3) array of (x, y, z) positions.
    """
    rng = np.random.RandomState(seed)

    positions = np.zeros((n_viewpoints, 3), dtype=np.float32)
    positions[:, 1] = 1.5  # eye level

    # Sample in corridor region
    n_corridor = n_viewpoints // 2
    positions[:n_corridor, 0] = rng.uniform(-5.5, 5.5, n_corridor)
    positions[:n_corridor, 2] = rng.uniform(-0.7, 0.7, n_corridor)

    # Sample in room region
    n_room = n_viewpoints - n_corridor
    positions[n_corridor:, 0] = rng.uniform(6.5, 9.5, n_room)
    positions[n_corridor:, 2] = rng.uniform(-2.5, 2.5, n_room)

    return positions


if __name__ == "__main__":
    scene = create_corridor_scene()
    pos, rot = sample_camera_path(50)
    print(f"Camera path: {len(pos)} frames")
    print(f"Path range: x=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}], "
          f"z=[{pos[:,2].min():.1f}, {pos[:,2].max():.1f}]")
