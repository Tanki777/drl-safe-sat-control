import os
import sys
import time
import viser
import viser.transforms as tf
import numpy as np
from dataclasses import dataclass
from pathlib import Path

drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if drl_repo_dir not in sys.path:
    sys.path.insert(0, drl_repo_dir)

from agent_training.trainer import YELLOW_START, COLOR_END
from agent_simulation.evaluation import load_evaluation_data
from agent_training.environment import normalize_vector, rotate_vector_by_quaternion
from agent_training.constants import Constants

repo_parent_dir = os.path.dirname(drl_repo_dir)
eval_data_dir = os.path.join(repo_parent_dir, "evaluation_data")

# Define custom colors
class Colors():
    """
    Custom colors.
    """

    GOLD = (194, 168, 0)
    GRAY = (180, 190, 205)
    RED = (255, 0, 0)
    ORANGE = (255, 130, 0)
    MAGENTA = (255, 0, 200)
    MINT = (0, 255, 200)
    DARK_BLUE = (0, 0, 100)

    KOZ_COLORS_ITERATOR = [
        RED,
        MINT,
        MAGENTA,
        ORANGE
    ]

class EpisodePlaybackController:
    """
    Controls the replay of one loaded episode.
    """

    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        

        self._last_frame = -1

        # Init camera
        self.CAMERA_POSITION_TARGET = (1.6, 0.0, 0.0)
        self.CAMERA_LOOK_AT_ORIGIN = (0.0, 0.0, 0.0)
        self.CAMERA_UP_DIRECTION_Z = (0.0, 0.0, 1.0)
        self.CAMERA_POSITION_BORESIGHT_SCALE = 1.6
        self.server.initial_camera.position = self.CAMERA_POSITION_TARGET
        self.server.initial_camera.look_at = self.CAMERA_LOOK_AT_ORIGIN
        self.camera_rate = 60.0

        self.client: viser.ClientHandle = None
        
        self._create_gui()
        self._register_callbacks()

    def _get_episode_selection_options(self, evaluation_file_name: str) -> list:
        file_path = os.path.join(eval_data_dir, evaluation_file_name)
        initial_episode_count = len(load_evaluation_data(file_path))
        options = np.arange(initial_episode_count).tolist() # E.g. 20 --> [0,1,2,...,19]

        # Convert list elements to string
        for idx in iter(options):
            options[idx] = str(options[idx])

        return options

    def _create_scene_objects(self) -> None:
        koz_cnt = len(self.episode_data["normal_vector_koz_array"])
    
        for koz_idx in range(koz_cnt):
            add_koz(self.server, f"KOZ {koz_idx+1}", self.episode_data["normal_vector_koz_array"][koz_idx], self.episode_data["half_angle_koz_array"][koz_idx], Colors.KOZ_COLORS_ITERATOR[koz_idx])
    
        add_target(self.server)
        self.sat_frame_handle = add_satellite(self.server, self.episode_data["quaternion"][0])
        
        self.times = self.episode_data["times"]
        self.quaternions = self.episode_data["quaternion"]
        self.num_frames = len(self.times)

        # Precompute the boresight direction in world frame for all frames for trajectory.
        body_boresight = np.array([1.0, 0.0, 0.0])
        self.trajectory_points = np.asarray(
            [rotate_vector_by_quaternion(body_boresight,quaternion) for quaternion in self.quaternions],
            dtype=np.float64
        )

        initial_boresight = self.trajectory_points[0]

        # Viser expects line segments with shape (N, 2, 3).
        initial_segment = np.asarray([[initial_boresight, initial_boresight]])

        self.trajectory_handle = self.server.scene.add_line_segments(
            name="/episode/trajectory",
            points=initial_segment,
            colors=Colors.GOLD,
            line_width=3.0
        )

    def _load_episode(self, file_name: str, episode_nr: str) -> None:
        episodes = load_evaluation_data(file_name)
        self.episode_data = episodes[int(episode_nr)] # Need to cast as GUI dropdown value is string.

        self._create_scene_objects()

        # Set GUI timestep slider max value to correct value
        self.gui_timestep.max = self.num_frames - 1

        self._enable_playback_gui()

        self._reset_episode()

    def _enable_playback_gui(self) -> None:
        # Playback
        self.gui_playing.disabled = False
        self.gui_timestep.disabled = False
        self.gui_previous.disabled = False
        self.gui_next.disabled = False
        self.gui_reset.disabled = False
        self.gui_speed.disabled = False
        self.gui_loop.disabled = False

        # Display
        self.gui_show_trajectory.disabled = False
        self.gui_show_full_trajectory.disabled = False

        # Camera
        self.gui_camera_perspective.disabled = False

        # Command
        self.command_play_pause.disabled = False

    def _reset_episode(self) -> None:
  
        self.gui_playing.value = False
        self.gui_timestep.value = 0
        self.set_frame(0)

    def _create_gui(self) -> None:

        # Set label name of the GUI panel.
        self.server.gui.set_panel_label("Episode viewer")

        # Add episode selection GUI folder
        with self.server.gui.add_folder("Episode Selection"):
            # If evaluation files exist, load them.
            if os.listdir(eval_data_dir):
                file_options = os.listdir(eval_data_dir)
                episode_options = self._get_episode_selection_options(os.listdir(eval_data_dir)[0])
                disabled = False

            # Otherwise, show warning.
            else:
                file_options = ("Could not find evaluation_data folder",)
                episode_options = ("0")
                disabled = True

            # Add a dropdown menu to list all evaluation files. For some reason it is bugged if not using the extra comma here...
            self.gui_evaluation_file = self.server.gui.add_dropdown("Evaluation file", options=file_options, disabled=disabled)

            # Add a dropdown menu to select the episode among the episodes of the selected file.
            self.gui_selected_episode = self.server.gui.add_dropdown("Episode", options=episode_options, disabled=disabled)

            # Add a button to load episode
            self.gui_load_episode = self.server.gui.add_button("Load episode", disabled=disabled)

            
        # Add a playback GUI folder.
        with self.server.gui.add_folder("Playback"):

            # Add a checkbox to toggle playing.
            self.gui_playing = self.server.gui.add_checkbox("Playing", initial_value=False, disabled=True)

            # Add a slider for timestep
            self.gui_timestep = self.server.gui.add_slider("Timestep", min=0, max=2999, step=1, initial_value=0, disabled=True) # Dummy max value on init

            # Add a button to go to previous frame.
            self.gui_previous = self.server.gui.add_button("Previous frame", disabled=True)

            # Add a button to go to next frame.
            self.gui_next = self.server.gui.add_button("Next frame", disabled=True)

            # Add a button to go to initial frame.
            self.gui_reset = self.server.gui.add_button("Reset", disabled=True)

            # Add a dropdown menu for playback speed
            self.gui_speed = self.server.gui.add_dropdown("Speed", options=("0.25x","0.5x","1x","2x","4x"), initial_value="1x", disabled=True)

            # Add a checkbox to toggle playback looping.
            self.gui_loop = self.server.gui.add_checkbox("Loop", initial_value=True, disabled=True)

        # Add a GUI folder for episode state information.
        with self.server.gui.add_folder("Episode state"):

            # Add current frame number
            self.gui_frame_number = self.server.gui.add_number("Frame", initial_value=0, disabled=True)

            # Show current simulation time
            self.gui_time = self.server.gui.add_number("Simulation time [s]",initial_value=0.0, disabled=True)

        # Add a GUI folder for toggling object visibility.
        with self.server.gui.add_folder("Display"):

            # Add checkbox to toggle trajectory so far.
            self.gui_show_trajectory = self.server.gui.add_checkbox("Show trajectory", initial_value=True, disabled=True)

            # Add checkbox to toggle entire episode trajectory.
            self.gui_show_full_trajectory = self.server.gui.add_checkbox("Show complete trajectory", initial_value=True, disabled=True)

        # Add a GUI folder for camera settings.
        with self.server.gui.add_folder("Camera"):

            # Add dropdown menu to select camera perspective.
            self.gui_camera_perspective = self.server.gui.add_dropdown("Perspective", options=("target","boresight"), initial_value="target", disabled=True)

        # Add command to play / pause with spacebar.
        self.command_play_pause = self.server.gui.add_command(label="Toggle play/pause", hotkey="space", disabled=True)

    # TODO: add callbacks for episode selection
    def _register_callbacks(self) -> None:

        # On updating evaluation file selection, update episode GUI selection.
        @self.gui_evaluation_file.on_update
        def _(_) -> None:

            new_value = self.gui_evaluation_file.value
            self.gui_selected_episode.options = self._get_episode_selection_options(new_value)

        # On clicking load episode button, load episode
        @self.gui_load_episode.on_click
        def _(_) -> None:
            # TODO: dont load if same episode as before
            self._load_episode(self.gui_evaluation_file.value, self.gui_selected_episode.value)

        # On updating the timestep, set new frame.
        @self.gui_timestep.on_update
        def _(_) -> None:
            self.set_frame(int(self.gui_timestep.value))

        # On clicking previous frame button, update frame.
        @self.gui_previous.on_click
        def _(_) -> None:
            self.gui_playing.value = False
            previous_frame = max(0, int(self.gui_timestep.value) - 1)
            self.gui_timestep.value = previous_frame

        # On clicking next frame button, update frame.
        @self.gui_next.on_click
        def _(_) -> None:
            self.gui_playing.value = False
            next_frame = min(self.num_frames - 1, int(self.gui_timestep.value) + 1)
            self.gui_timestep.value = next_frame

        # On clicking frame reset button, update frame.
        @self.gui_reset.on_click
        def _(_) -> None:
            self._reset_episode()

        # On updating show trajectory checkbox, update its visibility.
        @self.gui_show_trajectory.on_update
        def _(_) -> None:
            self.trajectory_handle.visible = self.gui_show_trajectory.value

        # On updating show entire episode trajectory, update trajectory points shown.
        @self.gui_show_full_trajectory.on_update
        def _(_) -> None:
            self._update_trajectory(int(self.gui_timestep.value))

        # On updating camera perspective, handle it.
        @self.gui_camera_perspective.on_update
        def _(_) -> None:
            self._handle_camera_perspective_change()

        # On triggering play / pause hotkey, toggle play / pause.
        @self.command_play_pause.on_trigger
        def _(_) -> None:
            self.gui_playing.value = not self.gui_playing.value

        # On client connect, save client handle (we only consider single client in this project).
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.client = client

    def _handle_camera_perspective_change(self) -> None:
        """
        Switch to the selected camera mode.
        """

        if self.client is None:
            return

        frame_index = int(self.gui_timestep.value)

        # Change camera parameters atomically to prevent jitter.
        with self.client.atomic():
            if self.gui_camera_perspective.value == "target":
                self.client.camera.position = self.CAMERA_POSITION_TARGET
                self.client.camera.look_at = self.CAMERA_LOOK_AT_ORIGIN
                self.client.camera.up_direction = self.CAMERA_UP_DIRECTION_Z
            else:
                self.client.camera.position = self.trajectory_points[frame_index] * self.CAMERA_POSITION_BORESIGHT_SCALE
                self.client.camera.look_at = self.CAMERA_LOOK_AT_ORIGIN

    def _update_trajectory(self, frame_index: int) -> None:
        """
        Updates trajectory path per frame.
        """
        
        # If selected to show entire episode trajectory, show all trajectory points.
        if self.gui_show_full_trajectory.value:
            points = self.trajectory_points
        # Otherwise, only show trajectory points until current frame.
        else:
            points = self.trajectory_points[: frame_index + 1]

        if len(points) < 2:
            # add_line_segments and its update expect at least one segment.
            points = np.vstack((points[0], points[0]))

        segments = np.stack((points[:-1], points[1:]), axis=1)

        self.trajectory_handle.points = segments

    def _get_speed_multiplier(self) -> float:
        speed_lookup = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1x": 1.0,
            "2x": 2.0,
            "4x": 4.0
        }

        return speed_lookup[self.gui_speed.value]

    def set_frame(self, frame_index: int) -> None:
        """
        Sets a specific frame number.
        """

        frame_index = int(np.clip(frame_index, 0, self.num_frames - 1))

        quaternion = self.quaternions[frame_index]

        with self.server.atomic():
            # Rotate the parent satellite frame. The body and body axis children inherit this attitude.
            self.sat_frame_handle.wxyz = quaternion

            self._update_trajectory(frame_index)

            self.gui_frame_number.value = frame_index
            self.gui_time.value = float(self.times[frame_index])

        # Camera is client-specific.
        if self.client is not None and self.gui_camera_perspective.value == "boresight":
            with self.client.atomic():
                self.client.camera.position = self.trajectory_points[frame_index] * self.CAMERA_POSITION_BORESIGHT_SCALE
                self.client.camera.look_at = self.CAMERA_LOOK_AT_ORIGIN

        self._last_frame = frame_index

    def run(self) -> None:
        """
        Starts the playback loop.
        """

        last_update_time = time.perf_counter()
        accumulated_frames = 0.0

        while True:
            current_time = time.perf_counter()
            elapsed = current_time - last_update_time
            last_update_time = current_time

            if self.gui_playing.value:
                effective_fps = 10 * self._get_speed_multiplier()

                accumulated_frames += elapsed * effective_fps
                frames_to_advance = int(accumulated_frames)

                if frames_to_advance > 0:
                    accumulated_frames -= frames_to_advance

                    current_frame = int(
                        self.gui_timestep.value
                    )
                    next_frame = current_frame + frames_to_advance

                    if next_frame >= self.num_frames:
                        if self.gui_loop.value:
                            next_frame %= self.num_frames
                        else:
                            next_frame = self.num_frames - 1
                            self.gui_playing.value = False

                    # Updating the GUI value triggers the timestep callback which then calls set_frame().
                    self.gui_timestep.value = next_frame
            else:
                # Prevent a large jump when playback resumes.
                accumulated_frames = 0.0

            # Add sleep to prevent loop from blocking other tasks.
            time.sleep(1.0 / self.camera_rate)


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
    server.scene.add_glb(
        name="/satellite/body",
        glb_data=(Path(__file__).resolve().parent.parent / "assets" / "cubesat.glb").read_bytes(),
        scale=1.0
    )

    origin = [0.0, 0.0, 0.0]

    # Define body axes as arrows
    axes_points = np.array([
        [origin, [1.0, 0.0, 0.0]], # X axis (boresight)
        [origin, [0.0, 1.0, 0.0]], # Y axis
        [origin, [0.0, 0.0, 1.0]] # Z axis
    ])

    axes_colors = np.array([
        Colors.GOLD,
        Colors.GRAY,
        Colors.GRAY
    ])

    server.scene.add_arrows(
        name="/satellite/axes/x",
        points=axes_points,
        colors=axes_colors,
        shaft_radius=0.005,
        head_radius=0.01,
        head_length=0.05
    )

    return satellite_frame


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
                first_ring_start + next_segment
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
                    outer_next
                )
            )
            faces.append(
                (
                    inner_current,
                    outer_next,
                    inner_next
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
        receive_shadow=False
    )

    # add_line_segments expects shape (N, 2, 3).
    border_segments = np.stack((border[:-1], border[1:]), axis=1)

    # Add border
    border_handle = server.scene.add_line_segments(
        name=f"/koz/{name}/border",
        points=border_segments,
        colors=color,
        line_width=1.0
    )

    return koz_handle, border_handle


def add_target(server: viser.ViserServer):
    """
    Adds the target.

    Args:
        server: Viser server to add objects to scene.
    """

    segments_angular = 96
    radius = 0.02
    normal_vector = np.array((1.0, 0.0, 0.0))

    basis_u, basis_v = create_orthonormal_basis(normal_vector)

    phi = np.linspace(
        0.0,
        2.0 * np.pi,
        segments_angular,
        endpoint=False
    )

    tangent_directions = (
        np.cos(phi)[:, None] * basis_u[None, :]
        + np.sin(phi)[:, None] * basis_v[None, :]
    )

    points = (
        np.cos(radius) * normal_vector[None, :]
        + np.sin(radius) * tangent_directions
    )

    # Close the circle.
    points = np.vstack((points, points[0]))

    points_stacked = np.stack((points[:-1], points[1:]), axis=1)

    server.scene.add_line_segments(
        name="/target/border",
        points=points_stacked,
        colors=Colors.GOLD,
        line_width=2.0
    )


def start_server():
    print(f"|---{YELLOW_START}Starting Viser server...{COLOR_END}")

    server = viser.ViserServer()

    print("|-----Access Viser at: http://localhost:8080")
    print("|-----Press Ctrl+C to stop the server")

    # Init theme
    server.gui.configure_theme(show_logo=False, dark_mode=True)

    playback = EpisodePlaybackController(server)

    playback.run()


if __name__ == "__main__":
    start_server()
