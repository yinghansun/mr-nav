from .env_cfg import EnvCfg


class VesselEnvCfg(EnvCfg):

    num_envs: int = 40
    img_idx: int = 0  # only valid for num_envs = 1
    
    max_episode_length: int = 1500

    visual_info_type: str = 'scan_pts' # ['scan_pts', 'img_latent']
    visual_model_type: str = 'symmetric' # ['symmetric', 'asymmetric']

    scan_size_list: list[int] = [6]

    num_pos: int = 2
    num_vel: int = 2

    num_actions: int = 1
    action_distribution_type: str = 'beta'  # ['normal', 'beta']
    action_degree_type: str = 'degree'  # ['radian', 'degree']

    num_lidar_directions: int = 36
    lidar_max_depth: int = 20
    
    # lidar + target + pos + last action
    num_obs: int = num_lidar_directions * 2 + num_pos * 2 + num_actions 

    class sim:
        sim_vessel2d_dataset_path: str = './dataset/unique_simulated_vessel_2d'
        sim_width: int = 128
        sim_height: int = 128

        # sim_vessel2d_dataset_path: str = './dataset/unique_simulated_vessel_2d_downsampled'
        # sim_width: int = 32
        # sim_height: int = 32

        # sim_vessel2d_dataset_path: str = './dataset/selected_32x32'
        # sim_width: int = 32
        # sim_height: int = 32

        robot_dim: int = 1

    class reward:
        class scales:
            # task
            reach_target: float = 10
            collision: float = -10
            
            # shape
            # distance_change: float = 0.5     # optimal value: 0.5
            distance: float = -1e-3            # optimal value: 1e-5 when it is the main objective

            # regularization
            # action_rate: float = -1e-2
            # log_obstacle: float = 0.001
            # pbrs_log_obstacle: float = -0.1
            # inverse_obstacle: float = 0.01

    class render:
        scale: int = 5
        robot_dim_vis: int = 16   # d
        # robot_color: tuple = (0, 150, 255)
        robot_color: tuple = (255, 150, 0)   # (B, G, R)

        target_dim_vis: int = 6  # d
        target_color: tuple = (0, 90, 250)
        target_thickness: int = 10

        intersection_pts_dim_vis: int = 1