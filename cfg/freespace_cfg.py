from .env_cfg import EnvCfg


class FreespaceCfg(EnvCfg):

    num_envs: int = 1024

    num_channels: int = 3
    
    dt: float = 0.1

    # dynamics
    mu: float = 2. / 9
    frequency: float = 9
    noise_level: float = 0.1

    max_episode_length: int = 1000

    reach_target_threshold: int = 3

    num_pos: int = 2
    num_vel: int = 2
    history_length: int = 1

    num_actions: int = 1
    action_distribution_type: str = 'beta'  # ['normal', 'beta']
    action_degree_type: str = 'degree'  # ['radian', 'degree']

    use_img: bool = False
    num_img_latent: int = 64
    if use_img: 
        num_obs = num_img_latent + num_actions
    else:
        num_obs = num_pos + num_actions

    class sim:
        sim_width: int = 128
        sim_height: int = 128
        # sim_width: int = 32
        # sim_height: int = 32
        robot_dim: int = 1

    # class render:
    #     scale: int = 5
    #     robot_dim_vis: int = 4   # d
    #     # robot_color: tuple = (0, 150, 255)
    #     robot_color: tuple = (255, 150, 0)   # (B, G, R)

    #     target_dim_vis: int = 4  # d
    #     target_color: tuple = (0, 90, 250)
    #     target_thickness: int = 5

    class render:
        scale: int = 5
        robot_dim_vis: int = 4   # d
        # robot_color: tuple = (0, 150, 255)
        robot_color: tuple = (255, 150, 0)   # (B, G, R)

        target_dim_vis: int = 5  # d
        target_color: tuple = (0, 90, 250)
        target_thickness: int = 5

    class reward:
        class scales:
            # position_tracking: float = 0.
            # action_rate: float = -1e-2
            reach_target: float = 10
            collision: float = -10
            distance = 0.
            distance_change = 0.
            # distance_change: float = 0.5
            # time_accumulation: float = -1e-5
            # time_accumulation: float = -1e-2   # optimal value when it is the main objective
            # distance: float = -1e-4            # optimal value when it is the main objective
            # one_over_distance: float = 1e-1
            # pbrs_one_over_distance: float = 0.3

        position_tracking_sigma: float = 2500