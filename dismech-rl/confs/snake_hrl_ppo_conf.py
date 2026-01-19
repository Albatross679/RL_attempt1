"""
ALF configuration for Snake HRL Manager using PPO algorithm.

The manager learns to select between approach and coil skills (discrete actions).
"""
import alf
from functools import partial

from environments.snake_hrl_env import SnakeHRLEnv

render = alf.define_config("render", False)

# Manager-specific UTD ratio (can be different from workers)
utd_ratio = alf.define_config("utd_ratio", 4)

# ============================================================
# Hyperparameter Tuning Configuration
# ============================================================
# These can be overridden via command line or gin config
# Example: --conf "worker_steps=25" or NUM_ITERATIONS=1000

# Worker steps per manager step (experiment: 10, 25, 50, 100)
worker_steps = alf.define_config("worker_steps", 50)

# Entropy regularization (experiment: 0.01, 0.05, 0.1)
entropy_reg = alf.define_config("entropy_reg", 0.1)

# Switch bonus for successful skill transition (experiment: 0.5, 1.0, 2.0)
switch_bonus = alf.define_config("switch_bonus", 1.0)

# Learning rate (experiment: 1e-4, 3e-4, 1e-3)
learning_rate = alf.define_config("learning_rate", 3e-4)

# ============================================================
# Environment Configuration
# ============================================================

alf.config("SnakeHRLEnv",
           worker_steps_per_manager_step=worker_steps,  # Tunable
           approach_success_threshold=0.15,
           coil_success_wrap_angle=6.28,  # 2*pi
           timeout_manager_steps=100,
           switch_bonus=switch_bonus,                   # Tunable
           completion_bonus=10.0,
           sim_timestep=5e-2,
           control_interval=2,
           worker_timeout_steps=500,
           cylinder_distance_range=(0.5, 1.5),
           cylinder_angle_range=(-30.0, 30.0),
           cylinder_radius=0.08,
           cylinder_height=0.5,
           rft_ct=0.01,
           rft_cn=0.1,
           render=render)

# Environment loader
load_func = lambda *args, **kwargs: SnakeHRLEnv()

alf.config("create_environment",
           env_name="snake_hrl",
           env_load_fn=load_func,
           batch_size_per_env=1,
           num_parallel_environments=64)  # Fewer than workers since HRL is more complex

# ============================================================
# Network Configuration for Discrete Actions
# ============================================================

# For discrete actions, use CategoricalProjectionNetwork
from alf.nest.utils import NestConcat

hidden_layers = (128, 128)  # Smaller network for simple manager state

# Actor network for discrete actions
actor_network_cls = partial(
    alf.networks.ActorDistributionNetwork,
    preprocessing_combiner=NestConcat(),
    fc_layer_params=hidden_layers,
    use_fc_ln=True,
    discrete_projection_net_ctor=alf.networks.CategoricalProjectionNetwork
)

# Value network
value_network_cls = partial(
    alf.networks.ValueNetwork,
    preprocessing_combiner=NestConcat(),
    fc_layer_params=hidden_layers,
    use_fc_ln=True
)

# ============================================================
# PPO Algorithm Configuration
# ============================================================

from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.agent import Agent

alf.config("PPOLoss",
           entropy_regularization=entropy_reg,  # Tunable
           td_loss_weight=0.5)

optimizer = alf.optimizers.AdamTF(lr=learning_rate)  # Tunable

rl_alg_ctor = partial(
    PPOAlgorithm,
    actor_network_ctor=actor_network_cls,
    value_network_ctor=value_network_cls,
    loss_class=PPOLoss
)

alg_ctor = partial(
    Agent,
    rl_algorithm_cls=rl_alg_ctor,
    optimizer=optimizer
)

# ============================================================
# Training Configuration
# ============================================================

alf.config("TrainerConfig",
           algorithm_ctor=alg_ctor,
           temporally_independent_train_step=True,
           mini_batch_length=1,
           mini_batch_size=256,  # Smaller batches for faster iteration
           unroll_length=8,      # Longer unroll for manager
           num_iterations=100_000,
           num_updates_per_train_iter=utd_ratio,
           evaluate=False,
           summarize_first_interval=True,
           summarize_grads_and_vars=False,
           summarize_action_distributions=True,  # Track manager action distribution
           debug_summaries=True,                 # Enable debug summaries
           summary_interval=100,                 # Log every 100 iterations
           num_checkpoints=50,
           whole_replay_buffer_training=True,
           clear_replay_buffer=True)
