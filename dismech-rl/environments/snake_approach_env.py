"""
Snake Approach Environment

The snake must learn to approach a cylindrical target using undulation locomotion.
Uses RFT (Resistive Force Theory) for ground interaction.
"""
import numpy as np

import alf

from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment,
    ImplicitEulerTimeStepper
)

from environments.base_env import BaseEnv
from environments.utils.common import NUM_NODES, LENGTH, DENSITY, YOUNG_MOD, POISSON
from environments.utils.snake_common import (
    Cylinder, ApproachObservation,
    compute_head_direction, compute_forward_velocity
)
from environments.utils.potential_functions import (
    approach_simple_distance, approach_distance_alignment,
    approach_time_to_goal, approach_exp_distance
)
from utils.action_converter import ActionConverter, generate_ctrl_and_state_indices

# Snake-specific parameters
SNAKE_RADIUS = 1e-3  # 1mm radius (thin snake)
SNAKE_LENGTH = 0.5   # 50cm length


def create_horizontal_snake_geometry(num_nodes: int, length: float, radius: float) -> Geometry:
    """
    Create a horizontal snake geometry lying on the ground.

    Snake extends along X-axis with z = radius (resting on ground).
    Node 0 is at origin (tail), node N-1 is at (length, 0, radius) (head).

    Args:
        num_nodes: Number of nodes in the snake
        length: Total length of snake
        radius: Snake body radius (for z-offset)

    Returns:
        Geometry object
    """
    nodes = np.zeros((num_nodes, 3), dtype=np.float64)
    nodes[:, 0] = np.linspace(0, length, num_nodes)  # X: 0 to length
    nodes[:, 1] = 0.0  # Y: centered
    nodes[:, 2] = radius  # Z: resting on ground

    edges = np.array([[i, i + 1] for i in range(num_nodes - 1)], dtype=np.int64)
    face_nodes = np.empty((0, 3), dtype=np.int64)

    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


@alf.configurable
class SnakeApproachEnv(BaseEnv):
    """
    Environment for snake to approach a cylindrical target.

    The snake uses undulation locomotion (via RFT ground model) to
    navigate toward a randomly placed cylinder.
    """

    def __init__(self,
                 sim_timestep: float = 5e-2,
                 control_interval: int = 2,
                 timeout_steps: int = 500,
                 cylinder_distance_range: tuple = (0.5, 1.5),
                 cylinder_angle_range: tuple = (-30.0, 30.0),
                 cylinder_radius: float = 0.08,
                 cylinder_height: float = 0.5,
                 success_threshold: float = 0.15,
                 rft_ct: float = 0.01,
                 rft_cn: float = 0.1,
                 potential_type: str = "none",
                 potential_gamma: float = 0.99,
                 potential_params: dict = None,
                 render: bool = False):
        """
        Args:
            sim_timestep: Simulation timestep
            control_interval: Number of sim steps per control step
            timeout_steps: Steps before episode timeout
            cylinder_distance_range: (min, max) distance from snake start
            cylinder_angle_range: (min, max) heading offset in degrees
            cylinder_radius: Radius of target cylinder
            cylinder_height: Height of target cylinder
            success_threshold: Distance to cylinder for success
            rft_ct: RFT tangential friction coefficient
            rft_cn: RFT normal friction coefficient
            potential_type: PBRS potential type (none|simple_distance|distance_alignment|time_to_goal|exp_distance)
            potential_gamma: Discount factor for PBRS shaping
            potential_params: Additional parameters for potential function
            render: Whether to render
        """
        self._ws_dim = 3  # Always 3D

        # Cylinder randomization parameters
        self._cylinder_distance_range = cylinder_distance_range
        self._cylinder_angle_range = np.radians(cylinder_angle_range)
        self._cylinder_radius = cylinder_radius
        self._cylinder_height = cylinder_height
        self._success_threshold = success_threshold

        # RFT parameters
        self._rft_ct = rft_ct
        self._rft_cn = rft_cn

        # PBRS parameters
        self._potential_type = potential_type
        self._potential_gamma = potential_gamma
        self._potential_params = potential_params or {}

        # Current cylinder target (set in _custom_reset)
        self._cylinder = None

        # These will be set by base class
        self._robot = None
        self._stepper = None

        super().__init__(
            sim_timestep=sim_timestep,
            control_interval=control_interval,
            timeout_steps=timeout_steps,
            render=render
        )

        # Override control/state indices for snake
        # Control every 4th node, observe every 2nd
        self._ctrl_indices, self._state_indices = generate_ctrl_and_state_indices(
            num_nodes=NUM_NODES,
            ctrl_spacing=4,
            state_spacing=2,
            offset=2
        )
        self._n_state_points = len(self._state_indices)
        self._n_ctrl_points = len(self._ctrl_indices)

        # Reinitialize action converter with new indices
        self._action_converter = ActionConverter(
            ctrl_indices=self._ctrl_indices,
            num_nodes=NUM_NODES
        )

        # Observation spec
        self._observation_spec = ApproachObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3,)),
            state_vel=alf.TensorSpec((self._n_state_points * 3,)),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2,)),
            cylinder_rel_pos=alf.TensorSpec((3,)),
            head_direction=alf.TensorSpec((3,)),
        )

        self._action_spec = self._action_converter.action_spec()

        if self._render:
            from environments.utils.snake_common import SnakeVisualizer
            self._renderer = SnakeVisualizer()

        # Initialize with a reset
        self._create_new_sim()

    def _create_new_sim(self):
        """Create a new snake simulation with RFT ground model."""
        # Create horizontal snake geometry
        geo = create_horizontal_snake_geometry(
            num_nodes=NUM_NODES,
            length=SNAKE_LENGTH,
            radius=SNAKE_RADIUS
        )

        # Geometric parameters
        geom_params = GeomParams(rod_r0=SNAKE_RADIUS, shell_h=0.0)

        # Material properties (from snake.ipynb)
        material = Material(
            density=1200,
            youngs_rod=2e6,
            youngs_shell=0.0,
            poisson_rod=0.5,
            poisson_shell=0.0
        )

        # Simulation parameters
        sim_params = SimParams(
            static_sim=False,
            two_d_sim=False,  # Full 3D for proper undulation
            use_mid_edge=False,
            use_line_search=False,
            show_floor=False,
            log_data=False,
            log_step=1,
            dt=self._sim_timestep,
            max_iter=25,
            total_time=1e6,
            plot_step=1,
            tol=1e-4,
            ftol=1e-4,
            dtol=1e-2
        )

        # Environment with RFT (no gravity needed)
        env = Environment()
        env.add_force('rft', ct=self._rft_ct, cn=self._rft_cn)

        # Create soft robot
        self._robot = SoftRobot(geom_params, material, geo, sim_params, env)

        # NO fixed nodes - free locomotion
        # self._robot = self._robot.fix_nodes(np.array([0]))  # REMOVED

        # Custom sim params (if any)
        self._custom_sim_params()

        # Create time stepper
        self._stepper = ImplicitEulerTimeStepper(self._robot)

        # For compatibility with base class
        self._arm = self

    def getVertices(self) -> np.ndarray:
        """Get vertex positions from the robot state."""
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)
        positions = self._robot.state.q[dof_indices].reshape(-1, 3)
        return positions

    def getVelocities(self) -> np.ndarray:
        """Get vertex velocities from the robot state."""
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)
        velocities = self._robot.state.u[dof_indices].reshape(-1, 3)
        return velocities

    def step_simulation(self, delta_action: dict):
        """
        Step the simulation with the given action.

        Uses inc_strain for actuation (following snake.ipynb example).
        """
        if 'delta_curvature' in delta_action:
            delta_curvature = delta_action['delta_curvature']
            # Use inc_strain (incremental strain) as in snake.ipynb
            self._robot.bend_springs.inc_strain[:, 0] += delta_curvature[:, 2]
            self._robot.bend_springs.inc_strain[:, 1] += delta_curvature[:, 3]

        # Step the simulation
        self._robot, _ = self._stepper.step(self._robot, debug=False)

    def _randomize_cylinder(self):
        """Randomize cylinder position for new episode."""
        # Random distance from snake head
        distance = np.random.uniform(*self._cylinder_distance_range)

        # Random angle offset from forward direction
        angle = np.random.uniform(*self._cylinder_angle_range)

        # Snake starts along X-axis, head at (SNAKE_LENGTH, 0, SNAKE_RADIUS)
        # Cylinder placed relative to head position
        head_pos = np.array([SNAKE_LENGTH, 0.0, 0.0])

        # Compute cylinder center in XY plane
        cx = head_pos[0] + distance * np.cos(angle)
        cy = head_pos[1] + distance * np.sin(angle)
        cz = 0.0  # Base of cylinder on ground

        self._cylinder = Cylinder(
            center=np.array([cx, cy, cz], dtype=np.float32),
            radius=self._cylinder_radius,
            height=self._cylinder_height
        )

    def _generate_observation(self) -> ApproachObservation:
        """Generate observation for the approach task."""
        all_positions = self._arm.getVertices()
        all_velocities = self._arm.getVelocities()

        positions = all_positions[self._state_indices].ravel().astype(np.float32)
        velocities = all_velocities[self._state_indices].ravel().astype(np.float32)

        # Head position (last node)
        head_pos = all_positions[-1]

        # Cylinder relative to head
        cylinder_rel_pos = (self._cylinder.center - head_pos).astype(np.float32)
        # Adjust z to be relative to cylinder center height
        cylinder_rel_pos[2] = self._cylinder.center[2] + self._cylinder_height/2 - head_pos[2]

        # Head direction
        head_direction = compute_head_direction(all_positions)

        return ApproachObservation(
            state_pos=positions,
            state_vel=velocities,
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            cylinder_rel_pos=cylinder_rel_pos,
            head_direction=head_direction,
        )

    def _compute_potential(self, obs: ApproachObservation) -> float:
        """
        Compute potential function value for PBRS.

        Args:
            obs: Current observation

        Returns:
            Potential value Φ(s)
        """
        if self._potential_type == "none":
            return 0.0

        # Default parameters for each potential type
        defaults = {
            "simple_distance": {"scale": 1.0, "d_max": 2.0},
            "distance_alignment": {"alpha": 1.0, "beta": 0.5, "d_max": 2.0},
            "time_to_goal": {"epsilon": 0.01, "scale": 1.0},
            "exp_distance": {"sigma": 0.3, "d_success": self._success_threshold},
        }

        # Merge defaults with user-provided params
        params = {**defaults.get(self._potential_type, {}), **self._potential_params}

        if self._potential_type == "simple_distance":
            return approach_simple_distance(
                obs.cylinder_rel_pos,
                scale=params["scale"],
                d_max=params["d_max"]
            )
        elif self._potential_type == "distance_alignment":
            return approach_distance_alignment(
                obs.cylinder_rel_pos,
                obs.head_direction,
                alpha=params["alpha"],
                beta=params["beta"],
                d_max=params["d_max"]
            )
        elif self._potential_type == "time_to_goal":
            # Compute COM velocity
            all_velocities = self._arm.getVelocities()
            com_vel = np.mean(all_velocities, axis=0)
            return approach_time_to_goal(
                obs.cylinder_rel_pos,
                com_vel,
                epsilon=params["epsilon"],
                scale=params["scale"]
            )
        elif self._potential_type == "exp_distance":
            return approach_exp_distance(
                obs.cylinder_rel_pos,
                sigma=params["sigma"],
                d_success=params["d_success"]
            )
        else:
            raise ValueError(f"Unknown potential type: {self._potential_type}")

    def _compute_reward(self, obs: ApproachObservation) -> float:
        """
        Compute approach reward with optional PBRS.

        Rewards:
        - Negative squared distance to cylinder (approach incentive)
        - Forward velocity bonus
        - Proximity bonus when close to cylinder
        - PBRS shaping term if enabled
        """
        # Distance to cylinder (XY plane)
        dist = np.linalg.norm(obs.cylinder_rel_pos[:2])

        # Base reward: negative squared distance
        base_reward = -dist ** 2

        # Forward velocity bonus
        all_positions = self._arm.getVertices()
        all_velocities = self._arm.getVelocities()
        fwd_vel = compute_forward_velocity(all_positions, all_velocities)
        base_reward += 0.1 * max(0, fwd_vel)  # Only reward forward movement

        # Proximity bonus
        if dist < 0.1:
            base_reward += 2.0
        elif dist < 0.2:
            base_reward += 0.5

        # Apply PBRS if enabled
        if self._potential_type != "none":
            # Φ(terminal) = 0
            phi_prime = 0.0 if self._is_terminal else self._compute_potential(obs)
            phi = 0.0 if self._prev_obs is None else self._compute_potential(self._prev_obs)
            shaping = self._potential_gamma * phi_prime - phi
            reward = base_reward + shaping
        else:
            reward = base_reward

        return float(reward)

    def _check_success(self, obs: ApproachObservation) -> bool:
        """Check if snake has reached the cylinder."""
        dist = np.linalg.norm(obs.cylinder_rel_pos[:2])
        return dist < self._success_threshold

    def _custom_sim_params(self):
        """Custom simulation parameters (called after robot creation)."""
        pass

    def _custom_step(self, action: np.ndarray):
        """Execute one environment step."""
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval
        )

        for _ in range(self._control_interval):
            self.step_simulation(delta_action)

    def _custom_reset(self):
        """Reset environment state."""
        self._randomize_cylinder()
        if self._render and hasattr(self, '_renderer'):
            self._renderer.clear_cylinder()

    def render(self, mode='rgb_array'):
        """Render the environment."""
        if self._render:
            self._renderer.render(
                self._arm.getVertices(),
                self._cylinder
            )
