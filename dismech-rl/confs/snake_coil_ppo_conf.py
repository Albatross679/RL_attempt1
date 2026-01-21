"""
ALF configuration for Snake Coil task using PPO algorithm.

Trains a snake to coil around a cylindrical target.
This is Phase 2 of the HRL pipeline after approach is learned.
"""
import alf

from environments.snake_coil_env import SnakeCoilEnv

render = alf.define_config("render", False)

# Import common PPO training configs
alf.import_config("common_ppo_training_conf.py")

# Custom action converter config for snake
alf.config("ActionConverter",
           delta_kappa_scale=0.15,
           kappa_bar_range=(-4.0, 4.0),
           smooth_action=True)

# Snake coil environment configuration
alf.config("SnakeCoilEnv",
           sim_timestep=5e-2,
           control_interval=2,
           timeout_steps=500,
           cylinder_radius=0.08,
           cylinder_height=0.5,
           success_wrap_angle=6.28,  # 2*pi radians
           rft_ct=0.01,
           rft_cn=0.1,
           potential_type="wrap_progress",  # Options: none|wrap_progress|wrap_contact|geometric
           potential_gamma=0.99,
           potential_params={},
           render=render)

# Action converter for 3D bend control
alf.config("ActionConverter", ws_dim=3)

# Environment loader
load_func = lambda *args, **kwargs: SnakeCoilEnv()

alf.config("create_environment",
           env_name="snake_coil",
           env_load_fn=load_func)
