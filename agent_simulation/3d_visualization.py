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


def add_satellite(server: viser.ViserServer, init_attitude: np.ndarray):
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
        radius=0.995,
        subdivisions=20,
        color=(180, 190, 205),
        opacity=0.18,
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
    )


def start_server():
    print(f"|---{YELLOW_START}Starting Viser server...{COLOR_END}")

    server = viser.ViserServer()

    # Load episode data
    episodes = load_evaluation_data("rewMod22_phFull_3_ph2_schedStage22_3800000_[150.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[1000]_2026-07-21-22-10-16.npz")
    episode_data = episodes[0] # First episode

    sat_frame = add_satellite(server, episode_data["quaternion"][0])
    add_unit_sphere(server)

    print("|-----Access Viser at: http://localhost:8080")
    print("|-----Press Ctrl+C to stop the server")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    start_server()
