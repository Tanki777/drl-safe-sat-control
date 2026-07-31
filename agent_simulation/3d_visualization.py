import os
import sys
import time
import viser
import numpy as np

drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if drl_repo_dir not in sys.path:
    sys.path.insert(0, drl_repo_dir)

from agent_training.trainer import YELLOW_START, COLOR_END
from agent_simulation.evaluation import load_evaluation_data
from agent_training.environment import normalize_vector


def add_satellite(server: viser.ViserServer, init_attitude: np.ndarray) -> viser.FrameHandle:
    """
    Adds a simple satellite body and body axes.

    Args:
        server: Viser server to add objects to scene.
        init_attitude: Initial attitude quaternion.
    Returns:
        satellite_frame: The satellite frame.
    """

    initial_attitude = init_attitude

    # Add parent frame
    satellite_frame = server.scene.add_frame(
        name="/satellite",
        show_axes=False,
        wxyz=initial_attitude,
        position=(0.0, 0.0, 0.0)
    )

    # Add body
    server.scene.add_box(
        name="/satellite/body",
        dimensions=(0.3, 0.1, 0.1),
        color=(150, 155, 165),
        material="standard",
        flat_shading=True
    )

    axis_length = 0.32

    # Body x axis (boresight vector)
    server.scene.add_line_segments(
        name="/satellite/axes/boresight_x",
        points=np.asarray([[[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0]]]),
        colors=(255, 170, 0),
        line_width=8.0,
    )

    # Body y axis
    server.scene.add_line_segments(
        name="/satellite/axes/body_y",
        points=np.asarray([[[0.0, 0.0, 0.0], [0.0, axis_length, 0.0]]]),
        colors=(0, 190, 80),
        line_width=4.0,
    )

    # Body z axis
    server.scene.add_line_segments(
        name="/satellite/axes/body_z",
        points=np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]]]),
        colors=(60, 120, 255),
        line_width=4.0,
    )

    # Boresight label
    # server.scene.add_label(
    #     name="/satellite/axes/boresight_label",
    #     text="Boresight",
    #     position=(axis_length, 0.0, 0.0),
    # )

    return satellite_frame


def add_unit_sphere(server: viser.ViserServer):
    """
    Adds the unit sphere.

    Args:
        server: Viser server to add objects to scene.
    """
    return server.scene.add_icosphere(
        name="/environment/unit_sphere",
        radius=0.999,
        subdivisions=20,
        color=(180, 190, 205),
        opacity=0.05,
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
    )


def create_orthonormal_basis(normal_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs two unit tangent vectors u and v such that
    (u, v, normal_vector) forms an orthonormal basis.

    Args:
        normal_vector: The normal vector from which to create the basis.
    Returns:
        tuple: (u, v).
    """

    # Select a reference vector that is not almost parallel to normal.
    if abs(normal_vector[2]) < 0.9:
        ref_vector = np.array([0.0, 0.0, 1.0])
    else:
        ref_vector = np.array([0.0, 1.0, 0.0])

    u = normalize_vector(np.cross(ref_vector, normal_vector))
    v = np.cross(normal_vector, u)

    return u, v


def create_koz_mesh(normal_vector: np.ndarray, half_angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a part of a spherical triangle mesh and its border points for a keep-out zone.

    Args:
        normal_vector: Keep-out zone normal vector in world frame.
        half_angle: Keep-out zone half angle in rad.
    Returns:
        tuple: (vertices_array, faces_array, border).
    """
    radius_sphere = 1.0
    segments_radial = 16 # Number of subdivisions from the KOZ center to its border.
    segments_angular = 96 # Number of subdivisions around the KOZ.

    basis_u, basis_v = create_orthonormal_basis(normal_vector)

    vertices: list[np.ndarray] = []

    # Vertex 0 is the center of the spherical part.
    vertices.append(radius_sphere * normal_vector)

    # Create concentric rings on the sphere.
    for ring_index in range(1, segments_radial + 1):
        theta = half_angle * ring_index / segments_radial

        for segment_index in range(segments_angular):
            phi = 2.0 * np.pi * segment_index / segments_angular

            tangent_direction = (
                np.cos(phi) * basis_u
                + np.sin(phi) * basis_v
            )

            point = radius_sphere * (
                np.cos(theta) * normal_vector
                + np.sin(theta) * tangent_direction
            )

            vertices.append(point)

    faces: list[tuple[int, int, int]] = []

    # Triangles from center vertex to first ring.
    first_ring_start = 1

    for segment_index in range(segments_angular):
        next_segment = (segment_index + 1) % segments_angular

        faces.append(
            (
                0,
                first_ring_start + segment_index,
                first_ring_start + next_segment,
            )
        )

    # Triangles connecting the remaining adjacent rings.
    for ring_index in range(1, segments_radial):
        inner_ring_start = 1 + (ring_index - 1) * segments_angular
        outer_ring_start = 1 + ring_index * segments_angular

        for segment_index in range(segments_angular):
            next_segment = (segment_index + 1) % segments_angular

            inner_current = inner_ring_start + segment_index
            inner_next = inner_ring_start + next_segment
            outer_current = outer_ring_start + segment_index
            outer_next = outer_ring_start + next_segment

            faces.append(
                (
                    inner_current,
                    outer_current,
                    outer_next,
                )
            )
            faces.append(
                (
                    inner_current,
                    outer_next,
                    inner_next,
                )
            )

    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.uint32)

    # Repeat first point at the end to close the border polyline.
    border_start = 1 + (segments_radial - 1) * segments_angular
    border = vertices_array[
        border_start:border_start + segments_angular
    ]
    border = np.vstack((border, border[0]))

    return vertices_array, faces_array, border


def add_koz(server: viser.ViserServer, name: str, normal_vector: np.ndarray, half_angle: float, color: tuple):
    """
    Adds a keep-out zone.

    Args:
        server: Viser server to add objects to scene.
        name: Keep-out zone name.
        normal_vector: Keep-out zone normal vector in world frame.
        half_angle: Keep-out zone half angle in rad.
        color: RGB color.
    Returns:
        tuple: (koz_handle, border_handle).
    """

    # Create mesh
    vertices, faces, border = create_koz_mesh(normal_vector, half_angle)

    # Add mesh
    koz_handle = server.scene.add_mesh_simple(
        name=f"/koz/{name}/mesh",
        vertices=vertices,
        faces=faces,
        color=color,
        opacity=0.15,
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
    )

    # add_line_segments expects shape (N, 2, 3).
    border_segments = np.stack((border[:-1], border[1:]), axis=1)

    # Add border
    border_handle = server.scene.add_line_segments(
        name=f"/koz/{name}/border",
        points=border_segments,
        colors=color,
        line_width=1.0,
    )

    return koz_handle, border_handle


def start_server():
    print(f"|---{YELLOW_START}Starting Viser server...{COLOR_END}")

    server = viser.ViserServer()

    # Load episode data
    episodes = load_evaluation_data("rewMod22_phFull_3_ph2_schedStage22_3800000_[150.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[1000]_2026-07-21-22-10-16.npz")
    episode_data = episodes[0] # First episode

    sat_frame = add_satellite(server, episode_data["quaternion"][0])
    
    add_koz(server, "KOZ 1", episode_data["normal_vector_koz"], episode_data["half_angle_koz"], (255, 0, 0))
    add_unit_sphere(server)

    print("|-----Access Viser at: http://localhost:8080")
    print("|-----Press Ctrl+C to stop the server")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    start_server()
