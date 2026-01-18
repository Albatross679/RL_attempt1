from functools import partial
import numpy as np

from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment,
    ImplicitEulerTimeStepper, ContactPair, ShellContactEnergy
)

from environments.dm_env import DisMechEnv, create_rod_geometry
from environments.utils.common import DENSITY, YOUNG_MOD, POISSON, MU, NUM_NODES, LENGTH, RADIUS
from environments.utils.obs_3d_common import (OBSTACLE_RADIUS, TARGET_POS,
                                              obstacle_3d_task_reward,
                                              Obstacle3DObservation,
                                              Obstacle3DTaskVisualizer)
from utils.obstacle_creator import create_obstacles

import alf
from alf.data_structures import TimeStep


@alf.configurable
class DisMechObstacle3DEnv(DisMechEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        self._ws_dim = 3  # 3D environment

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._observation_spec = Obstacle3DObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3, )),
            state_vel=alf.TensorSpec((self._n_state_points * 3, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2, )),
            target_pos=alf.TensorSpec((3, )),
            obstacle_pos=alf.TensorSpec((8 * 2 * 3, )),
        )
        self._obstacles = None
        self._action_spec = self._action_converter.action_spec()

        if self._render:
            self._arm_pos_history = []
            self._renderer = Obstacle3DTaskVisualizer()

        self._create_new_sim()

    def _create_new_sim(self):
        """
        Create simulation with main robot and obstacle rods.
        Overrides parent to add obstacles.
        """
        # Create main rod geometry
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([0.0, 0.0, LENGTH])
        geo = create_rod_geometry(start_pos, end_pos, NUM_NODES)

        # Geometric parameters (rod radius, no shell)
        geom_params = GeomParams(rod_r0=RADIUS, shell_h=0.0)

        # Material properties
        material = Material(
            density=DENSITY,
            youngs_rod=YOUNG_MOD,
            youngs_shell=0.0,
            poisson_rod=POISSON,
            poisson_shell=0.0
        )

        # Simulation parameters - 3D mode
        sim_params = SimParams(
            static_sim=False,
            two_d_sim=False,  # 3D simulation
            use_mid_edge=False,
            use_line_search=False,
            show_floor=False,
            log_data=False,
            log_step=1,
            dt=self._sim_timestep,
            max_iter=5,  # More iterations for contact
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

        # Create time stepper
        self._stepper = ImplicitEulerTimeStepper(self._robot)

        # For compatibility with base class
        self._arm = self

        # Create obstacles (for observation, contact not yet implemented)
        self._obstacles = create_obstacles()

        if self._render:
            self._renderer.set_static_obstacles(self._obstacles)

        # Note: Contact with obstacles is not yet implemented in dismech-python
        # The obstacles are created for observation purposes but contact handling
        # would require additional implementation

    def _generate_observation(self) -> Obstacle3DObservation:
        positions = self._arm.getVertices()[self._state_indices].ravel()
        velocities = self._arm.getVelocities()[self._state_indices].ravel()

        return Obstacle3DObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
            obstacle_pos=self._obstacles.ravel())

    def _compute_reward(self, obs: Obstacle3DObservation) -> float:
        return obstacle_3d_task_reward(obs)

    def _custom_step(self, action: np.ndarray) -> TimeStep:
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self.step_simulation(delta_action)
            if self._render:
                self._arm_pos_history.append(self._arm.getVertices())

    def render(self, mode='rgb_array'):
        if self._render:
            for arm_pos in self._arm_pos_history:
                self._renderer.render(arm_pos=arm_pos)

            self._arm_pos_history.clear()
