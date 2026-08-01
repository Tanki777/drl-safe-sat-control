import os
import sys
import time
import viser
import viser.transforms as tf
import numpy as np
from dataclasses import dataclass

drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if drl_repo_dir not in sys.path:
    sys.path.insert(0, drl_repo_dir)

from agent_training.trainer import YELLOW_START, COLOR_END
from agent_simulation.evaluation import load_evaluation_data
from agent_training.environment import normalize_vector, rotate_vector_by_quaternion
from agent_training.constants import Constants


# Define custom colors
class Colors():
    """
    Custom colors.
    """

    GOLD = (194, 168, 0)
    GRAY = (180, 190, 205)
    RED = (255, 0, 0)

@dataclass
class CameraTransition:
    """
    Active smooth camera transition for one connected client.
    """

    start_pose: tf.SE3
    target_pose: tf.SE3
    start_time: float
    duration: float
    final_look_at: np.ndarray

class EpisodePlaybackController:
    """
    Controls the replay of one loaded episode.
    """

    def __init__(self, server: viser.ViserServer, episode_data: dict, sat_frame_handle: viser.FrameHandle) -> None:
        self.server = server
        self.sat_frame_handle = sat_frame_handle

        self.times = episode_data["times"]
        self.quaternions = episode_data["quaternion"]
        self.num_frames = len(self.times)

        # Precompute the boresight direction in world frame for all frames for trajectory.
        body_boresight = np.array([1.0, 0.0, 0.0])

        self.trajectory_points = np.asarray(
            [rotate_vector_by_quaternion(body_boresight,quaternion) for quaternion in self.quaternions],
            dtype=np.float64
        )

        self._last_frame = -1

        # Init camera
        self.server.initial_camera.position = (1.8, 0.0, 0.0)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        self.camera_transition: CameraTransition = None
        self.camera_rate = 60.0
        self.camera_perspective_change_duration = 0.4

        self.client: viser.ClientHandle = None
        

        self._create_dynamic_objects()
        self._create_gui()
        self._register_callbacks()

        self.set_frame(0)

    def _create_dynamic_objects(self) -> None:
        initial_boresight = self.trajectory_points[0]

        # Viser expects line segments with shape (N, 2, 3).
        initial_segment = np.asarray([[initial_boresight, initial_boresight]])

        self.trajectory_handle = self.server.scene.add_line_segments(
            name="/episode/trajectory",
            points=initial_segment,
            colors=Colors.GOLD,
            line_width=3.0
        )

    def _create_gui(self) -> None:

        # Set label name of the GUI panel.
        self.server.gui.set_panel_label("Episode viewer")

        # Add a playback GUI folder.
        with self.server.gui.add_folder("Playback"):

            # Add a checkbox to toggle playing.
            self.gui_playing = self.server.gui.add_checkbox("Playing", initial_value=False)

            # Add a slider for timestep
            self.gui_timestep = self.server.gui.add_slider("Timestep", min=0, max=self.num_frames-1, step=1, initial_value=0)

            # Add a button to go to previous frame.
            self.gui_previous = self.server.gui.add_button("Previous frame")

            # Add a button to go to next frame.
            self.gui_next = self.server.gui.add_button("Next frame")

            # Add a button to go to initial frame.
            self.gui_reset = self.server.gui.add_button("Reset")

            # Add a dropdown menu for playback speed
            self.gui_speed = self.server.gui.add_dropdown("Speed", options=("0.25x","0.5x","1x","2x","4x"), initial_value="1x")

            # Add a checkbox to toggle playback looping.
            self.gui_loop = self.server.gui.add_checkbox("Loop", initial_value=True)

        # Add a GUI folder for episode state information.
        with self.server.gui.add_folder("Episode state"):

            # Add current frame number
            self.gui_frame_number = self.server.gui.add_number("Frame", initial_value=0, disabled=True)

            # Show current simulation time
            self.gui_time = self.server.gui.add_number("Simulation time [s]",initial_value=0.0, disabled=True)

        # Add a GUI folder for toggling object visibility.
        with self.server.gui.add_folder("Display"):

            # Add checkbox to toggle trajectory so far.
            self.gui_show_trajectory = self.server.gui.add_checkbox("Show trajectory", initial_value=True)

            # Add checkbox to toggle entire episode trajectory.
            self.gui_show_full_trajectory = self.server.gui.add_checkbox("Show complete trajectory", initial_value=False)

        # Add a GUI folder for camera settings.
        with self.server.gui.add_folder("Camera"):

            # Add dropdown menu to select camera perspective.
            self.gui_camera_perspective = self.server.gui.add_dropdown("Perspective", options=("target","boresight"), initial_value="target")

        # Add command to play / pause with spacebar.
        self.command_play_pause = self.server.gui.add_command(label="Toggle play/pause", hotkey="space")

    def _register_callbacks(self) -> None:

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
            self.gui_playing.value = False
            self.gui_timestep.value = 0

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
        Smoothly switch to the selected camera mode.
        """

        frame_index = int(
            self.gui_timestep.value
        )
        
        if (
            self.gui_camera_perspective.value
            == "target"
        ):
            self._start_target_camera_transition(
                client=self.client,
                duration=(
                    self.camera_perspective_change_duration
                ),
            )
        else:
            self._start_boresight_camera_transition(
                client=self.client,
                frame_index=frame_index,
                duration=(
                    self.camera_perspective_change_duration
                ),
            )

    def _update_camera_transition(self) -> None:
        """
        Advance active camera transition by one visual frame.
        """

        if not self.camera_transition:
            return

        now = time.perf_counter()

        alpha = (now - self.camera_transition.start_time) / self.camera_transition.duration

        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Smoothstep easing.
        eased_alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        relative_transform = self.camera_transition.start_pose.inverse() @ self.camera_transition.target_pose

        interpolated_pose = self.camera_transition.start_pose @ tf.SE3.exp(relative_transform.log() * eased_alpha)

        with self.client.atomic():
            self.client.camera.wxyz = interpolated_pose.rotation().wxyz

            self.client.camera.position = interpolated_pose.translation()

        if alpha >= 1.0:
            # Set the final orbit center after the pose transition.
            self.client.camera.look_at = (self.camera_transition.final_look_at)

        self.camera_transition = None # TODO

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

    def _get_frame_interval(self, frame_index: int) -> float:
        """
        Returns the wall clock interval until the next episode frame.
        """

        frame_interval = Constants.TIME_DELTA / self._get_speed_multiplier()

        return frame_interval

    @staticmethod
    def _rotation_matrix_to_wxyz(
        rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a 3x3 rotation matrix to a normalized quaternion.
        """
        matrix = np.asarray(rotation_matrix)

        trace = float(np.trace(matrix))

        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0

            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale

        elif (matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]):
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0

            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale

        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0

            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale

        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0

            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

        quaternion = np.array([w, x, y, z])

        quaternion /= np.linalg.norm(quaternion)

        return quaternion

    def _make_camera_pose_looking_at(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        preferred_up: np.ndarray,
    ) -> tf.SE3:
        """
        Construct a camera SE(3) pose that looks at a world point.

        The rotation columns contain the camera-local axes expressed in
        the world frame.

        Viser camera pose conventions use P_world = [R | t] P_camera.
        """
        position = np.asarray(
            position,
            dtype=np.float64,
        )
        look_at = np.asarray(
            look_at,
            dtype=np.float64,
        )
        preferred_up = np.asarray(
            preferred_up,
            dtype=np.float64,
        )

        # Direction from the camera towards the viewed point.
        forward = look_at - position
        forward_norm = np.linalg.norm(forward)

        if forward_norm < 1e-12:
            raise ValueError(
                "Camera position and look-at point "
                "must differ."
            )

        forward /= forward_norm

        preferred_up /= np.linalg.norm(
            preferred_up
        )

        # Avoid a bad cross product near the preferred up axis.
        if abs(
            np.dot(forward, preferred_up)
        ) > 0.95:
            preferred_up = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            )

            if abs(
                np.dot(forward, preferred_up)
            ) > 0.95:
                preferred_up = np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float64,
                )

        right = np.cross(
            preferred_up,
            forward,
        )
        right /= np.linalg.norm(right)

        up = np.cross(
            forward,
            right,
        )
        up /= np.linalg.norm(up)

        rotation_matrix = np.column_stack(
            (
                right,
                up,
                forward,
            )
        )

        wxyz = self._rotation_matrix_to_wxyz(
            rotation_matrix
        )

        return tf.SE3.from_rotation_and_translation(
            tf.SO3(wxyz),
            position,
        )

    def _get_target_camera_pose(
        self,
    ) -> tuple[tf.SE3, np.ndarray]:
        """
        Returns the static target perspective camera pose.
        """
        position = np.array(
            [1.8, 0.0, 0.0],
            dtype=np.float64,
        )

        look_at = np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        pose = self._make_camera_pose_looking_at(
            position=position,
            look_at=look_at,
            preferred_up=np.array(
                [0.0, 0.0, 1.0]
            ),
        )

        return pose, look_at

    def _get_boresight_camera_pose(
        self,
        frame_index: int,
    ) -> tuple[tf.SE3, np.ndarray]:
        """
        Gets a camera pose above the current boresight tip.

        The camera is radially outside the sphere and looks down at the
        boresight point on the sphere.
        """
        boresight_tip = np.asarray(
            self.trajectory_points[frame_index],
            dtype=np.float64,
        )
        boresight_tip /= np.linalg.norm(
            boresight_tip
        )

        camera_position = (
            1.0
            + 0.8
        ) * boresight_tip

        look_at = boresight_tip

        pose = self._make_camera_pose_looking_at(
            position=camera_position,
            look_at=look_at,
            preferred_up=np.array(
                [0.0, 0.0, 1.0]
            ),
        )

        return pose, look_at

    @staticmethod
    def _get_current_camera_pose(
        client: viser.ClientHandle,
    ) -> tf.SE3:
        return tf.SE3.from_rotation_and_translation(
            tf.SO3(
                np.asarray(
                    client.camera.wxyz,
                    dtype=np.float64,
                )
            ),
            np.asarray(
                client.camera.position,
                dtype=np.float64,
            ),
        )

    def _set_camera_immediately(
        self,
        client: viser.ClientHandle,
        target_pose: tf.SE3,
        target_look_at: np.ndarray,
    ) -> None:
        """
        Immediately apply one camera pose.
        """
        with client.atomic():
            client.camera.wxyz = (
                target_pose.rotation().wxyz
            )
            client.camera.position = (
                target_pose.translation()
            )

        # Position changes move look_at with them, so assign look_at
        # afterwards when a fixed orbit center is required.
        client.camera.look_at = target_look_at
        client.flush()

    def _start_camera_transition(
        self,
        client: viser.ClientHandle,
        target_pose: tf.SE3,
        target_look_at: np.ndarray,
        duration: float,
    ) -> None:
        """
        Begin or replace a non-blocking camera transition.

        Starting from the client's current interpolated pose prevents
        transition commands from accumulating.
        """
        if duration <= 0.0:
            self.camera_transition = None

            self._set_camera_immediately(
                client=client,
                target_pose=target_pose,
                target_look_at=target_look_at,
            )
            return

        current_pose = (
            self._get_current_camera_pose(
                client
            )
        )

        self.camera_transition = CameraTransition(
            start_pose=current_pose,
            target_pose=target_pose,
            start_time=time.perf_counter(),
            duration=max(
                float(duration),
                1.0 / self.camera_rate,
            ),
            final_look_at=np.asarray(
                target_look_at,
                dtype=np.float64,
            ),
        )

    def _start_target_camera_transition(
        self,
        client: viser.ClientHandle,
        duration: float,
    ) -> None:
        target_pose, target_look_at = (
            self._get_target_camera_pose()
        )

        self._start_camera_transition(
            client=client,
            target_pose=target_pose,
            target_look_at=target_look_at,
            duration=duration,
        )

    def _start_boresight_camera_transition(
        self,
        client: viser.ClientHandle,
        frame_index: int,
        duration: float,
    ) -> None:
        target_pose, target_look_at = (
            self._get_boresight_camera_pose(
                frame_index
            )
        )

        self._start_camera_transition(
            client=client,
            target_pose=target_pose,
            target_look_at=target_look_at,
            duration=duration,
        )

    def set_frame(self, frame_index: int) -> None:
        """
        Sets a specific frame number.
        """

        frame_index = int(np.clip(frame_index, 0, self.num_frames - 1))

        if frame_index == self._last_frame:
            return

        quaternion = self.quaternions[frame_index]

        with self.server.atomic():
            # Rotate the parent satellite frame. The body and body axis children inherit this attitude.
            self.sat_frame_handle.wxyz = quaternion

            self._update_trajectory(frame_index)

            self.gui_frame_number.value = frame_index
            self.gui_time.value = float(self.times[frame_index])

        # Camera is client-specific and smoothly follows the newest
        # boresight target.
        if self.gui_camera_perspective.value == "boresight":
            transition_duration = self._get_frame_interval(frame_index)
            self._start_boresight_camera_transition(self.client, frame_index, transition_duration)

        self._last_frame = frame_index

    def run(self) -> None:
        """
        Starts the playback loop.
        """

        last_update_time = time.perf_counter()
        last_camera_update_time = last_update_time
        accumulated_frames = 0.0

        while True:
            current_time = time.perf_counter()
            elapsed = current_time - last_update_time
            last_update_time = current_time
            
            # Camera transitions update independently from episode frames.
            if (current_time - last_camera_update_time >= 1.0 / self.camera_rate):
                self._update_camera_transition()
                last_camera_update_time = current_time

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
        dimensions=(0.2, 0.6, 0.2),
        color=Colors.GRAY,
        material="standard",
        flat_shading=True
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
        shaft_radius=0.01,
        head_radius=0.02,
        head_length=0.05
    )

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
        color=Colors.GRAY,
        opacity=0.05,
        material="standard",
        flat_shading=False,
        side="double",
        cast_shadow=False,
        receive_shadow=False
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

    boundary_handle = server.scene.add_line_segments(
        name="/target/boundary",
        points=points_stacked,
        colors=Colors.GOLD,
        line_width=1.0
    )


def start_server():
    print(f"|---{YELLOW_START}Starting Viser server...{COLOR_END}")

    server = viser.ViserServer()

    print("|-----Access Viser at: http://localhost:8080")
    print("|-----Press Ctrl+C to stop the server")

    # Load episode data
    episodes = load_evaluation_data("rewMod22_phFull_3_ph2_schedStage22_3800000_[150.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[1000]_2026-07-21-22-10-16.npz")
    episode_data = episodes[0] # First episode
    
    add_koz(server, "KOZ 1", episode_data["normal_vector_koz"], episode_data["half_angle_koz"], Colors.RED)
    add_target(server)
    #add_unit_sphere(server)
    sat_frame_handle = add_satellite(server, episode_data["quaternion"][0])

    playback = EpisodePlaybackController(server, episode_data, sat_frame_handle)

    playback.run()


if __name__ == "__main__":
    start_server()
