"""
ALF configuration for Snake Approach task.

Trains a snake to approach a cylindrical target using undulation locomotion.
"""
import alf

from environments.snake_approach_env import SnakeApproachEnv

render = alf.define_config("render", False)

# Import common training configs
alf.import_config("common_sac_training_conf.py")

# Custom action converter config for snake (larger scale for faster locomotion)
alf.config("ActionConverter",
           delta_kappa_scale=0.15,  # Larger than default 0.05 for snake locomotion
           kappa_bar_range=(-4.0, 4.0),  # Wider range for snake undulation
           smooth_action=True)

# Snake approach environment configuration
alf.config("SnakeApproachEnv",
           sim_timestep=5e-2,
           control_interval=2,
           timeout_steps=500,
           cylinder_distance_range=(0.5, 1.5),
           cylinder_angle_range=(-30.0, 30.0),
           cylinder_radius=0.08,
           cylinder_height=0.5,
           success_threshold=0.15,
           rft_ct=0.01,
           rft_cn=0.1,
           render=render)

# Action converter for 3D bend control
alf.config("ActionConverter", ws_dim=3)

# Environment loader
load_func = lambda *args, **kwargs: SnakeApproachEnv()

alf.config("create_environment",
           env_name="snake_approach",
           env_load_fn=load_func)
