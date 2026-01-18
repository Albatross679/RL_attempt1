from functools import partial
import numpy as np
import alf

from dismech import (
    SoftRobot, Geometry, GeomParams, Material, SimParams, Environment,
    ImplicitEulerTimeStepper, ContactPair, ShellContactEnergy
)

from environments.dm_env import DisMechEnv, create_rod_geometry
from environments.utils.common import DENSITY, YOUNG_MOD, POISSON, MU, NUM_NODES, LENGTH, RADIUS
from environments.utils.obs_2d_common import (OBSTACLE_RADIUS, OBSTACLES_2D,
                                              TARGET_POS,
                                              Obstacle2DObservation,
                                              obstacles_2d_task_reward,
                                              Obstacle2DTaskVisualizer)


@alf.configurable
class DisMechObstacle2DEnv(DisMechEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        self._ws_dim = 2  # 2D environment

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._observation_spec = Obstacle2DObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 2, )),
            state_vel=alf.TensorSpec((self._n_state_points * 2, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points, )),
            target_pos=alf.TensorSpec((2, )),
        )

        self._action_spec = self._action_converter.action_spec()

        if self._render:
            self._arm_pos_history = []
            self._renderer = Obstacle2DTaskVisualizer(
                ctrl_indices=self._ctrl_indices)

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

        # Simulation parameters - 2D mode
        sim_params = SimParams(
            static_sim=False,
            two_d_sim=True,  # 2D simulation
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

        # Note: Contact with obstacles is not yet implemented in dismech-python
        # The obstacles are defined in OBSTACLES_2D but contact handling
        # would require additional implementation

    def _generate_observation(self) -> Obstacle2DObservation:
        positions = self._arm.getVertices()[self._state_indices, ::2].ravel()
        velocities = self._arm.getVelocities()[
            self._state_indices, ::2].ravel()

        return Obstacle2DObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
        )

    def _compute_reward(self, obs: Obstacle2DObservation) -> float:
        return obstacles_2d_task_reward(obs)

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self.step_simulation(delta_action)
            if self._render:
                self._arm_pos_history.append(self._arm.getVertices()[:, ::2])

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(arm_pos_history=self._arm_pos_history)
            self._arm_pos_history.clear()
