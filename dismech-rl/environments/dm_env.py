from abc import ABC
import numpy as np

from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment,
    ImplicitEulerTimeStepper
)

from environments.base_env import BaseEnv
from environments.utils.common import NUM_NODES, LENGTH, DENSITY, RADIUS, YOUNG_MOD, POISSON, MU


def create_rod_geometry(start: np.ndarray, end: np.ndarray, num_nodes: int) -> Geometry:
    """
    Create a rod geometry programmatically.

    Args:
        start: Starting position (3D array)
        end: Ending position (3D array)
        num_nodes: Number of nodes in the rod

    Returns:
        Geometry object for the rod
    """
    # Create node positions by linearly interpolating between start and end
    nodes = np.zeros((num_nodes, 3), dtype=np.float64)
    for i in range(3):
        nodes[:, i] = np.linspace(start[i], end[i], num_nodes)

    # Create edges connecting adjacent nodes
    edges = np.array([[i, i + 1] for i in range(num_nodes - 1)], dtype=np.int64)

    # No shell faces for a rod
    face_nodes = np.empty((0, 3), dtype=np.int64)

    return Geometry(nodes, edges, face_nodes, plot_from_txt=False)


class DisMechEnv(BaseEnv, ABC):
    """
    An abstract base class for constructing DisMech environments for RL.
    Uses dismech-python (pure Python implementation).
    """

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):
        """
        Args:
            sim_timestep: Simulation timestep.
            control_interval: The number of sim steps for every control.
            timeout_steps: Number of steps before timing out an episode.
        """
        super().__init__(sim_timestep, control_interval, timeout_steps, render)

        # These will be set by _create_new_sim
        self._robot = None
        self._stepper = None
        self._ws_dim = 3  # Default, may be overridden by child classes

    def _create_new_sim(self):
        """
        This function can be called to create a new simulation.
        This is useful for episode resets.
        """
        # Create rod geometry
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([0.0, 0.0, LENGTH])
        geo = create_rod_geometry(start_pos, end_pos, NUM_NODES)

        # Geometric parameters (rod radius, no shell)
        geom_params = GeomParams(rod_r0=RADIUS, shell_h=0.0)

        # Material properties
        material = Material(
            density=DENSITY,
            youngs_rod=YOUNG_MOD,
            youngs_shell=0.0,  # No shell
            poisson_rod=POISSON,
            poisson_shell=0.0  # No shell
        )

        # Child classes can set _ws_dim before calling this
        two_d_sim = getattr(self, '_ws_dim', 3) == 2

        # Simulation parameters
        sim_params = SimParams(
            static_sim=False,
            two_d_sim=two_d_sim,
            use_mid_edge=False,
            use_line_search=False,
            show_floor=False,
            log_data=False,
            log_step=1,
            dt=self._sim_timestep,
            max_iter=100,  # Increased from 2 for better convergence
            total_time=1e6,
            plot_step=1,
            tol=1e-3,
            ftol=1e-3,
            dtol=1e-2
        )

        # Environment with gravity
        env = Environment()
        env.add_force('gravity', g=np.array([0.0, 0.0, -9.8]))

        # Create soft robot
        self._robot = SoftRobot(geom_params, material, geo, sim_params, env)

        # Fix the bottom node (node 0)
        self._robot = self._robot.fix_nodes(np.array([0]))

        # Child classes can set up custom sim params
        self._custom_sim_params()

        # Create time stepper
        self._stepper = ImplicitEulerTimeStepper(self._robot)

        # For compatibility with base class that uses self._arm
        self._arm = self  # Provide getVertices/getVelocities via this class

    def getVertices(self) -> np.ndarray:
        """
        Get vertex positions from the robot state.
        Returns positions shaped as (num_nodes, 3).
        """
        # Map all node indices to DOF indices
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)

        # Extract positions from state.q
        positions = self._robot.state.q[dof_indices].reshape(-1, 3)
        return positions

    def getVelocities(self) -> np.ndarray:
        """
        Get vertex velocities from the robot state.
        Returns velocities shaped as (num_nodes, 3).
        """
        # Map all node indices to DOF indices
        node_indices = np.arange(NUM_NODES)
        dof_indices = self._robot.map_node_to_dof(node_indices)

        # Extract velocities from state.u
        velocities = self._robot.state.u[dof_indices].reshape(-1, 3)
        return velocities

    def step_simulation(self, delta_action: dict):
        """
        Step the simulation with the given action.

        Args:
            delta_action: Dictionary containing curvature/twist deltas.
                Expected keys: 'delta_curvature' and optionally 'delta_theta'
        """
        # Apply curvature control via bend springs' natural strain
        if 'delta_curvature' in delta_action:
            delta_curvature = delta_action['delta_curvature']
            # delta_curvature shape: (num_internal_nodes, 4) where cols are [limb_idx, edge_idx, kappa1, kappa2]
            # nat_strain shape: (num_bend_springs, 2) for [kappa1, kappa2]
            # num_bend_springs = num_internal_nodes = num_nodes - 2, so shapes match directly
            self._robot.bend_springs.nat_strain[:, 0] += delta_curvature[:, 2]
            if self._ws_dim == 3:
                self._robot.bend_springs.nat_strain[:, 1] += delta_curvature[:, 3]

        # Apply theta (twist) control if present
        if 'delta_theta' in delta_action:
            delta_theta = delta_action['delta_theta']
            edge_indices = delta_theta[:, 1].astype(int)
            theta_deltas = delta_theta[:, 2]
            self._robot = self._robot.twist_edges(edge_indices, theta_deltas)

        # Step the simulation
        self._robot, _ = self._stepper.step(self._robot, debug=False)
