class EnvCfg:

    num_envs: int = 1024

    num_channels: int = 3
    
    dt: float = 0.1

    # dynamics
    mu: float = 2. / 9
    frequency: float = 9
    noise_level: float = 0.2

    max_episode_length: int = 3000

    num_pos: int = 2
    num_vel: int = 2
    history_length: int = 1

    num_actions: int = 1
    action_distribution_type: str = 'beta'  # ['normal', 'beta']
    action_degree_type: str = 'radian'      # ['radian', 'degree']

    use_img: bool = False
    num_img_latent: int = 64
    if use_img: 
        num_obs = num_img_latent + num_actions
    else:
        num_obs = num_pos + num_actions

    reach_target_threshold: int = 2

    class sim:
        sim_width: int = 128
        sim_height: int = 128

        robot_dim: int = 3

    class render:
        scale: int = 3
        robot_dim_vis: int = 3
        target_dim_vis: int = 3

    class reward:
        class scales:
            # task
            reach_target: float = 10
            collision: float = -10
            
            # shape
            distance_change: float = 0.5

            # regularization
            action_rate: float = -1e-2