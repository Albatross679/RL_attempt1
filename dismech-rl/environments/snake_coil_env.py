"""
Snake Coil Environment

The snake must learn to coil around a cylindrical target.
This environment is designed for the second phase of HRL training,
after the snake has learned to approach the cylinder.
"""
import numpy as np

import alf

from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment,
    ImplicitEulerTimeStepper
)

from environments.base_env import BaseEnv
from environments.snake_approach_env import (
    create_horizontal_snake_geometry, NUM_NODES, SNAKE_LENGTH, SNAKE_RADIUS
)
from environments.utils.snake_common import (
    Cylinder, CoilObservation,
    compute_head_direction, compute_node_cylinder_distances,
    compute_wrap_angle, compute_forward_velocity
)
from environments.utils.potential_functions import (
    coil_wrap_progress, coil_wrap_contact, coil_geometric
)
from utils.action_converter import ActionConverter, generate_ctrl_and_state_indices


@alf.configurable
class SnakeCoilEnv(BaseEnv):
    """
    Environment for snake to coil around a cylindrical target.

    The snake must wrap around the cylinder to maximize coverage.
    Designed for HRL where SnakeApproachEnv provides initial states.
    """

    def __init__(self,
                 sim_timestep: float = 5e-2,
                 control_interval: int = 2,
                 timeout_steps: int = 500,
                 cylinder_radius: float = 0.08,
                 cylinder_height: float = 0.5,
                 success_wrap_angle: float = 2 * np.pi,
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
            cylinder_radius: Radius of target cylinder
            cylinder_height: Height of target cylinder
            success_wrap_angle: Wrap angle (radians) for success (default 2*pi = full wrap)
            rft_ct: RFT tangential friction coefficient
            rft_cn: RFT normal friction coefficient
            potential_type: PBRS potential type (none|wrap_progress|wrap_contact|geometric)
            potential_gamma: Discount factor for PBRS shaping
            potential_params: Additional parameters for potential function
            render: Whether to render
        """
        self._ws_dim = 3

        # Cylinder parameters
        self._cylinder_radius = cylinder_radius
        self._cylinder_height = cylinder_height
        self._success_wrap_angle = success_wrap_angle

        # RFT parameters
        self._rft_ct = rft_ct
        self._rft_cn = rft_cn

        # PBRS parameters
        self._potential_type = potential_type
        self._potential_gamma = potential_gamma
        self._potential_params = potential_params or {}

        # Current cylinder and wrap tracking
        self._cylinder = None
        self._prev_wrap_angle = 0.0
        self._cumulative_wrap_angle = 0.0

        # Simulation objects
        self._robot = None
        self._stepper = None

        super().__init__(
            sim_timestep=sim_timestep,
            control_interval=control_interval,
            timeout_steps=timeout_steps,
            render=render
        )

        # Control/state indices
        self._ctrl_indices, self._state_indices = generate_ctrl_and_state_indices(
            num_nodes=NUM_NODES,
            ctrl_spacing=4,
            state_spacing=2,
            offset=2
        )
        self._n_state_points = len(self._state_indices)
        self._n_ctrl_points = len(self._ctrl_indices)

        # Action converter
        self._action_converter = ActionConverter(
            ctrl_indices=self._ctrl_indices,
            num_nodes=NUM_NODES
        )

        # Observation spec (extends ApproachObservation with coil-specific fields)
        self._observation_spec = CoilObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3,)),
            state_vel=alf.TensorSpec((self._n_state_points * 3,)),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2,)),
            cylinder_rel_pos=alf.TensorSpec((3,)),
            head_direction=alf.TensorSpec((3,)),
            node_distances=alf.TensorSpec((self._n_state_points,)),
            wrap_angle=alf.TensorSpec((1,)),
        )

        self._action_spec = self._action_converter.action_spec()

        if self._render:
            from environments.utils.snake_common import SnakeVisualizer
            self._renderer = SnakeVisualizer()

        self._create_new_sim()

    def _create_new_sim(self):
        """Create a new snake simulation."""
        geo = create_horizontal_snake_geometry(
            num_nodes=NUM_NODES,
            length=SNAKE_LENGTH,
            radius=SNAKE_RADIUS
        )

        geom_params = GeomParams(rod_r0=SNAKE_RADIUS, shell_h=0.0)

        material = Material(
            density=1200,
            youngs_rod=2e6,
            youngs_shell=0.0,
            poisson_rod=0.5,
            poisson_shell=0.0
        )

        sim_params = SimParams(
            static_sim=False,
            two_d_sim=False,
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

        env = Environment()
        env.add_force('rft', ct=self._rft_ct, cn=self._rft_cn)

        self._robot = SoftRobot(geom_params, material, geo, sim_params, env)
        self._custom_sim_params()
        self._stepper = ImplicitEulerTimeStepper(self._robot)
        self._arm = self

    def set_initial_state(self, positions: np.ndarray, velocities: np.ndarray, cylinder: Cylinder):
        """
        Initialize from SnakeApproachEnv terminal state.

        Called by HRL controller for skill chaining.

        Args:
            positions: (NUM_NODES, 3) node positions from approach env
            velocities: (NUM_NODES, 3) node velocities from approach env
            cylinder: Cylinder target from approach env
        """
        # Set cylinder
        self._cylinder = cylinder

        # Set robot state
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)

        self._robot.state.q[dof_indices] = positions.ravel()
        self._robot.state.u[dof_indices] = velocities.ravel()

        # Reset wrap tracking
        self._prev_wrap_angle = compute_wrap_angle(positions, cylinder)
        self._cumulative_wrap_angle = 0.0

    def getVertices(self) -> np.ndarray:
        """Get vertex positions."""
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)
        return self._robot.state.q[dof_indices].reshape(-1, 3)

    def getVelocities(self) -> np.ndarray:
        """Get vertex velocities."""
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)
        return self._robot.state.u[dof_indices].reshape(-1, 3)

    def step_simulation(self, delta_action: dict):
        """Step simulation with action."""
        if 'delta_curvature' in delta_action:
            delta_curvature = delta_action['delta_curvature']
            self._robot.bend_springs.inc_strain[:, 0] += delta_curvature[:, 2]
            self._robot.bend_springs.inc_strain[:, 1] += delta_curvature[:, 3]

        self._robot, _ = self._stepper.step(self._robot, debug=False)

    def _generate_observation(self) -> CoilObservation:
        """Generate observation for the coil task."""
        all_positions = self._arm.getVertices()
        all_velocities = self._arm.getVelocities()

        positions = all_positions[self._state_indices].ravel().astype(np.float32)
        velocities = all_velocities[self._state_indices].ravel().astype(np.float32)

        # Head position and direction
        head_pos = all_positions[-1]
        cylinder_rel_pos = (self._cylinder.center - head_pos).astype(np.float32)
        cylinder_rel_pos[2] = self._cylinder.center[2] + self._cylinder_height/2 - head_pos[2]
        head_direction = compute_head_direction(all_positions)

        # Node distances to cylinder surface
        state_positions = all_positions[self._state_indices]
        node_distances = compute_node_cylinder_distances(state_positions, self._cylinder)

        # Wrap angle (cumulative)
        wrap_angle = np.array([self._cumulative_wrap_angle], dtype=np.float32)

        return CoilObservation(
            state_pos=positions,
            state_vel=velocities,
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            cylinder_rel_pos=cylinder_rel_pos,
            head_direction=head_direction,
            node_distances=node_distances,
            wrap_angle=wrap_angle,
        )

    def _compute_potential(self, obs: CoilObservation) -> float:
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
            "wrap_progress": {"scale": 1.0},
            "wrap_contact": {"alpha": 1.0, "beta": 0.5, "lambda_": 0.1},
            "geometric": {"d_max": 0.5},
        }

        # Merge defaults with user-provided params
        params = {**defaults.get(self._potential_type, {}), **self._potential_params}

        wrap_angle = float(obs.wrap_angle[0])

        if self._potential_type == "wrap_progress":
            return coil_wrap_progress(
                wrap_angle,
                scale=params["scale"]
            )
        elif self._potential_type == "wrap_contact":
            return coil_wrap_contact(
                wrap_angle,
                obs.node_distances,
                self._cylinder_radius,
                alpha=params["alpha"],
                beta=params["beta"],
                lambda_=params["lambda_"]
            )
        elif self._potential_type == "geometric":
            return coil_geometric(
                wrap_angle,
                obs.node_distances,
                d_max=params["d_max"]
            )
        else:
            raise ValueError(f"Unknown potential type: {self._potential_type}")

    def _compute_reward(self, obs: CoilObservation) -> float:
        """
        Compute coil reward with optional PBRS.

        Rewards:
        - Wrapping progress (change in wrap angle)
        - Number of nodes close to cylinder
        - Full wrap bonus
        - PBRS shaping term if enabled
        """
        all_positions = self._arm.getVertices()

        # Current wrap angle
        current_wrap = compute_wrap_angle(all_positions, self._cylinder)
        wrap_delta = current_wrap - self._prev_wrap_angle

        # Handle wrap discontinuity
        if wrap_delta > np.pi:
            wrap_delta -= 2 * np.pi
        elif wrap_delta < -np.pi:
            wrap_delta += 2 * np.pi

        # Update cumulative and previous
        self._cumulative_wrap_angle += wrap_delta
        self._prev_wrap_angle = current_wrap

        base_reward = 0.0

        # Wrapping progress
        base_reward += 0.5 * wrap_delta

        # Contact bonus: nodes close to cylinder
        close_threshold = self._cylinder_radius * 0.5
        num_close = np.sum(np.abs(obs.node_distances) < close_threshold)
        base_reward += 0.2 * num_close

        # Negative distance penalty for nodes far from cylinder
        avg_distance = np.mean(np.abs(obs.node_distances))
        base_reward -= 0.1 * avg_distance

        # Full wrap bonus
        if self._cumulative_wrap_angle > self._success_wrap_angle:
            base_reward += 10.0

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

    def _check_success(self, obs: CoilObservation) -> bool:
        """Check if snake has fully wrapped the cylinder."""
        return self._cumulative_wrap_angle >= self._success_wrap_angle

    def _custom_sim_params(self):
        """Custom simulation parameters."""
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
        # Place cylinder at snake's mid-body for immediate coiling training
        # (In HRL pipeline, set_initial_state() would be used instead)
        positions = self.getVertices()
        mid_idx = len(positions) // 2
        mid_pos = positions[mid_idx]
        
        # Place cylinder centered on snake's mid-body
        cx = mid_pos[0]
        cy = mid_pos[1]
        cz = 0.0

        self._cylinder = Cylinder(
            center=np.array([cx, cy, cz], dtype=np.float32),
            radius=self._cylinder_radius,
            height=self._cylinder_height
        )

        # Reset wrap tracking
        self._prev_wrap_angle = compute_wrap_angle(self.getVertices(), self._cylinder)
        self._cumulative_wrap_angle = 0.0

        if self._render and hasattr(self, '_renderer'):
            self._renderer.clear_cylinder()

    def render(self, mode='rgb_array'):
        """Render the environment."""
        if self._render:
            self._renderer.render(
                self._arm.getVertices(),
                self._cylinder
            )
