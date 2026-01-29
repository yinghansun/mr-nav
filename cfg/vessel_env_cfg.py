from typing import List

from .alg_cfg import ActorCriticCfg
from .env_cfg import EnvCfg


class VesselEnvCfg(EnvCfg):

    num_envs: int = 8192
    img_idx: int = 0  # only valid for num_envs = 1
    
    max_episode_length: int = 1500

    visual_info_type: str = 'scan_pts' # ['scan_pts', 'img_latent']
    visual_model_type: str = 'symmetric' # ['symmetric', 'asymmetric']

    scan_size_list: list[int] = [6]

    num_pos: int = 2
    num_vel: int = 2

    num_actions: int = 1
    action_distribution_type: str = 'normal'  # ['normal', 'beta']
    action_degree_type: str = 'radian'  # ['radian', 'degree']

    num_lidar_directions: int = 36
    lidar_max_depth: int = 20
    
    # lidar + target + pos + last action
    num_obs: int = num_lidar_directions * 2 + num_pos * 2 + num_actions

    class sim:
        sim_vessel2d_dataset_path: str = './dataset/unique_simulated_vessel_2d'
        sim_width: int = 128
        sim_height: int = 128

        robot_dim: int = 1

    class reward:
        class scales:
            # task
            reach_target: float = 10
            collision: float = -10
            
            # shape
            distance_change: float = 0.5     # optimal value: 0.5

            # regularization
            action_rate: float = -1e-2
            log_obstacle: float = 0.001

    class render:
        scale: int = 5
        robot_dim_vis: int = 4   # d
        robot_color: tuple = (255, 150, 0)   # (B, G, R)

        target_dim_vis: int = 6  # d
        target_color: tuple = (0, 90, 250)
        target_thickness: int = 10

        intersection_pts_dim_vis: int = 1


class VesselEnvActorCriticCfg(ActorCriticCfg):

    actor_hidden_dims: List[int] = [256, 128, 16]
    critic_hidden_dims: List[int] = [256, 128, 16]
    distribution_type: str = 'normal'  # ['normal', 'beta']