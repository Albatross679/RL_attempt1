"""
Snake Hierarchical RL Environment

Manager agent that orchestrates approach and coil workers.
The manager selects which skill to execute, and workers execute low-level actions.
"""
import numpy as np
from typing import NamedTuple, Union, Optional, Tuple

import alf
from alf.data_structures import TimeStep, StepType
from alf.environments.alf_environment import AlfEnvironment
from alf.utils import common

from environments.snake_approach_env import SnakeApproachEnv, NUM_NODES
from environments.snake_coil_env import SnakeCoilEnv
from environments.utils.snake_common import Cylinder, compute_wrap_angle

# Type alias for observation spec compatibility
array = Union[np.ndarray, alf.TensorSpec]


class ManagerObservation(NamedTuple):
    """
    Abstract observation for the HRL manager.

    Attributes:
        distance_to_cylinder: (1,) head-to-cylinder distance
        wrap_angle: (1,) cumulative wrap angle around cylinder
        current_skill: (2,) one-hot encoding [approach, coil]
        skill_progress: (2,) [approach_success, coil_success] flags
    """
    distance_to_cylinder: array
    wrap_angle: array
    current_skill: array
    skill_progress: array


# Skill IDs
SKILL_APPROACH = 0
SKILL_COIL = 1


@alf.configurable
class SnakeHRLEnv(AlfEnvironment):
    """
    Hierarchical RL environment for snake predation.

    The manager agent selects between:
    - SKILL_APPROACH (0): Move toward the cylinder
    - SKILL_COIL (1): Wrap around the cylinder

    State transfer occurs when switching from approach to coil,
    using set_initial_state() to pass the terminal state.
    """

    def __init__(self,
                 worker_steps_per_manager_step: int = 50,
                 approach_success_threshold: float = 0.15,
                 coil_success_wrap_angle: float = 2 * np.pi,
                 timeout_manager_steps: int = 100,
                 switch_bonus: float = 1.0,
                 completion_bonus: float = 10.0,
                 # Worker environment configs (passed through)
                 sim_timestep: float = 5e-2,
                 control_interval: int = 2,
                 worker_timeout_steps: int = 500,
                 cylinder_distance_range: tuple = (0.5, 1.5),
                 cylinder_angle_range: tuple = (-30.0, 30.0),
                 cylinder_radius: float = 0.08,
                 cylinder_height: float = 0.5,
                 rft_ct: float = 0.01,
                 rft_cn: float = 0.1,
                 render: bool = False):
        """
        Args:
            worker_steps_per_manager_step: How many worker steps per manager action
            approach_success_threshold: Distance threshold for approach success
            coil_success_wrap_angle: Wrap angle for coil success
            timeout_manager_steps: Manager steps before episode timeout
            switch_bonus: Bonus for successful skill switch
            completion_bonus: Bonus for completing full task
            sim_timestep: Simulation timestep for workers
            control_interval: Control interval for workers
            worker_timeout_steps: Timeout steps for individual workers
            cylinder_distance_range: Cylinder placement range
            cylinder_angle_range: Cylinder angle range (degrees)
            cylinder_radius: Target cylinder radius
            cylinder_height: Target cylinder height
            rft_ct: RFT tangential friction
            rft_cn: RFT normal friction
            render: Whether to render
        """
        super().__init__()

        self._worker_steps_per_manager_step = worker_steps_per_manager_step
        self._approach_success_threshold = approach_success_threshold
        self._coil_success_wrap_angle = coil_success_wrap_angle
        self._timeout_manager_steps = timeout_manager_steps
        self._switch_bonus = switch_bonus
        self._completion_bonus = completion_bonus
        self._render = render

        # Store worker configs
        self._worker_config = dict(
            sim_timestep=sim_timestep,
            control_interval=control_interval,
            rft_ct=rft_ct,
            rft_cn=rft_cn,
            render=render,
        )
        self._approach_config = dict(
            timeout_steps=worker_timeout_steps,
            cylinder_distance_range=cylinder_distance_range,
            cylinder_angle_range=cylinder_angle_range,
            cylinder_radius=cylinder_radius,
            cylinder_height=cylinder_height,
            success_threshold=approach_success_threshold,
        )
        self._coil_config = dict(
            timeout_steps=worker_timeout_steps,
            cylinder_radius=cylinder_radius,
            cylinder_height=cylinder_height,
            success_wrap_angle=coil_success_wrap_angle,
        )

        # Create worker environments
        self._approach_env = SnakeApproachEnv(
            **self._worker_config, **self._approach_config
        )
        self._coil_env = SnakeCoilEnv(
            **self._worker_config, **self._coil_config
        )

        # Current state
        self._current_skill = SKILL_APPROACH
        self._manager_step_count = 0
        self._done = True
        self._approach_succeeded = False
        self._coil_succeeded = False

        # Current cylinder (shared between workers)
        self._cylinder: Optional[Cylinder] = None

        # Observation and action specs
        self._observation_spec = ManagerObservation(
            distance_to_cylinder=alf.TensorSpec((1,)),
            wrap_angle=alf.TensorSpec((1,)),
            current_skill=alf.TensorSpec((2,)),
            skill_progress=alf.TensorSpec((2,)),
        )

        # Discrete action: 0 = approach, 1 = coil
        self._action_spec = alf.BoundedTensorSpec(
            shape=(),
            dtype='int64',
            minimum=0,
            maximum=1,
        )

    def observation_spec(self) -> alf.NestedTensorSpec:
        return self._observation_spec

    def action_spec(self) -> alf.NestedBoundedTensorSpec:
        return self._action_spec

    def env_info_spec(self) -> alf.NestedTensorSpec:
        """Return specs for env_info metrics (for TensorBoard logging)."""
        return {
            'skill_selected': alf.TensorSpec((), dtype='int64'),
            'skill_0_selected': alf.TensorSpec(()),  # approach
            'skill_1_selected': alf.TensorSpec(()),  # coil
            'worker_reward': alf.TensorSpec(()),
            'switch_reward': alf.TensorSpec(()),
            'distance_to_cylinder': alf.TensorSpec(()),
            'wrap_angle': alf.TensorSpec(()),
            'approach_succeeded': alf.TensorSpec(()),
            'coil_succeeded': alf.TensorSpec(()),
        }

    def reward_spec(self) -> alf.TensorSpec:
        return alf.TensorSpec(())

    def seed(self, seed: Optional[int] = None):
        common.set_random_seed(seed)

    def _get_current_worker(self):
        """Get the currently active worker environment."""
        if self._current_skill == SKILL_APPROACH:
            return self._approach_env
        else:
            return self._coil_env

    def _get_worker_positions_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract positions and velocities from current worker."""
        worker = self._get_current_worker()
        positions = worker.getVertices()
        velocities = worker.getVelocities()
        return positions, velocities

    def _compute_abstract_state(self) -> ManagerObservation:
        """Compute abstract manager observation from worker state."""
        positions, _ = self._get_worker_positions_velocities()
        head_pos = positions[-1]

        # Distance to cylinder
        if self._cylinder is not None:
            dist = np.linalg.norm(head_pos[:2] - self._cylinder.center[:2])
        else:
            dist = 1.0  # Default if no cylinder

        # Wrap angle
        if self._cylinder is not None:
            wrap = compute_wrap_angle(positions, self._cylinder)
        else:
            wrap = 0.0

        # Current skill one-hot
        skill_onehot = np.zeros(2, dtype=np.float32)
        skill_onehot[self._current_skill] = 1.0

        # Skill progress
        progress = np.array([
            float(self._approach_succeeded),
            float(self._coil_succeeded)
        ], dtype=np.float32)

        return ManagerObservation(
            distance_to_cylinder=np.array([dist], dtype=np.float32),
            wrap_angle=np.array([wrap], dtype=np.float32),
            current_skill=skill_onehot,
            skill_progress=progress,
        )

    def _switch_to_skill(self, new_skill: int):
        """Switch to a different skill, transferring state if needed."""
        if new_skill == self._current_skill:
            return

        if new_skill == SKILL_COIL and self._current_skill == SKILL_APPROACH:
            # Transfer state from approach to coil
            positions, velocities = self._get_worker_positions_velocities()
            self._coil_env.set_initial_state(positions, velocities, self._cylinder)
            self._current_skill = SKILL_COIL

        elif new_skill == SKILL_APPROACH and self._current_skill == SKILL_COIL:
            # Switching back to approach (unusual but allowed)
            # Reset approach env with same cylinder position
            self._approach_env._reset()
            self._approach_env._cylinder = self._cylinder
            self._current_skill = SKILL_APPROACH

    def _execute_worker_steps(self, num_steps: int) -> Tuple[float, bool, bool]:
        """
        Execute worker policy for multiple steps.

        Returns:
            (total_reward, worker_succeeded, worker_terminated)
        """
        worker = self._get_current_worker()
        total_reward = 0.0
        worker_succeeded = False
        worker_terminated = False

        for _ in range(num_steps):
            if worker._done:
                break

            # Get worker action from its policy (trained separately)
            # For now, use random actions - in practice, load trained policy
            worker_action = worker.action_spec().sample()
            # Convert to numpy if needed (ALF tensors have .numpy() method)
            if hasattr(worker_action, 'numpy'):
                worker_action = worker_action.numpy()

            # Step the worker
            time_step = worker._step(worker_action)
            total_reward += float(time_step.reward)

            if time_step.step_type == StepType.LAST:
                worker_terminated = True
                # Check if it was a success
                if self._current_skill == SKILL_APPROACH:
                    worker_succeeded = self._approach_env._check_success(
                        self._approach_env._curr_obs
                    )
                else:
                    worker_succeeded = self._coil_env._check_success(
                        self._coil_env._curr_obs
                    )
                break

        return total_reward, worker_succeeded, worker_terminated

    def _step(self, action: np.ndarray) -> TimeStep:
        """Execute one manager step."""
        if self._done:
            return self._reset()

        # Parse discrete action
        skill_choice = int(action)

        # Handle skill switching
        prev_skill = self._current_skill
        self._switch_to_skill(skill_choice)

        # Compute switch bonus
        switch_reward = 0.0
        if (prev_skill == SKILL_APPROACH and
            skill_choice == SKILL_COIL and
            self._approach_succeeded):
            switch_reward = self._switch_bonus

        # Execute worker for N steps
        worker_reward, worker_succeeded, worker_terminated = self._execute_worker_steps(
            self._worker_steps_per_manager_step
        )

        # Update success flags
        if self._current_skill == SKILL_APPROACH and worker_succeeded:
            self._approach_succeeded = True
        elif self._current_skill == SKILL_COIL and worker_succeeded:
            self._coil_succeeded = True

        # Compute total reward
        reward = worker_reward + switch_reward
        if self._coil_succeeded:
            reward += self._completion_bonus

        # Increment manager step
        self._manager_step_count += 1

        # Check termination
        obs = self._compute_abstract_state()

        if self._coil_succeeded:
            # Full task complete
            step_type = StepType.LAST
            discount = 1.0
            self._done = True
        elif self._manager_step_count >= self._timeout_manager_steps:
            # Timeout
            step_type = StepType.LAST
            discount = 1.0
            self._done = True
        elif worker_terminated and self._current_skill == SKILL_COIL:
            # Coil worker failed - episode ends
            step_type = StepType.LAST
            discount = 0.0
            self._done = True
        else:
            step_type = StepType.MID
            discount = 1.0

        # Build env_info with tracking metrics
        env_info = {
            'skill_selected': np.int64(skill_choice),
            'skill_0_selected': np.float32(1.0 if skill_choice == SKILL_APPROACH else 0.0),
            'skill_1_selected': np.float32(1.0 if skill_choice == SKILL_COIL else 0.0),
            'worker_reward': np.float32(worker_reward),
            'switch_reward': np.float32(switch_reward),
            'distance_to_cylinder': obs.distance_to_cylinder[0],
            'wrap_angle': obs.wrap_angle[0],
            'approach_succeeded': np.float32(self._approach_succeeded),
            'coil_succeeded': np.float32(self._coil_succeeded),
        }

        return TimeStep(
            step_type=step_type,
            observation=obs,
            reward=np.float32(reward),
            discount=np.float32(discount),
            prev_action=action,
            env_info=env_info,
            env_id=np.int32(0),
        )

    def _reset(self) -> TimeStep:
        """Reset the HRL environment."""
        # Reset approach env (which randomizes cylinder)
        self._approach_env._reset()
        self._cylinder = self._approach_env._cylinder

        # Reset state tracking
        self._current_skill = SKILL_APPROACH
        self._manager_step_count = 0
        self._done = False
        self._approach_succeeded = False
        self._coil_succeeded = False

        obs = self._compute_abstract_state()

        # Initial env_info (zeros for reset)
        env_info = {
            'skill_selected': np.int64(0),
            'skill_0_selected': np.float32(0.0),
            'skill_1_selected': np.float32(0.0),
            'worker_reward': np.float32(0.0),
            'switch_reward': np.float32(0.0),
            'distance_to_cylinder': obs.distance_to_cylinder[0],
            'wrap_angle': obs.wrap_angle[0],
            'approach_succeeded': np.float32(0.0),
            'coil_succeeded': np.float32(0.0),
        }

        return TimeStep(
            step_type=StepType.FIRST,
            observation=obs,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            prev_action=np.int64(0),
            env_info=env_info,
            env_id=np.int32(0),
        )

    def render(self, mode='rgb_array'):
        """Render the current worker environment."""
        if self._render:
            worker = self._get_current_worker()
            worker.render(mode)

    def get_approach_env(self) -> SnakeApproachEnv:
        """Access approach worker (for loading checkpoints)."""
        return self._approach_env

    def get_coil_env(self) -> SnakeCoilEnv:
        """Access coil worker (for loading checkpoints)."""
        return self._coil_env

    def get_current_cylinder(self) -> Optional[Cylinder]:
        """Get current episode's cylinder target."""
        return self._cylinder
