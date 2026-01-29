from .alg_cfg import ActorCriticCfg
from .env_cfg import EnvCfg


class FreespaceCfg(EnvCfg):

    num_envs: int = 8192

    num_channels: int = 3
    
    dt: float = 0.1

    # dynamics
    mu: float = 2. / 9
    frequency: float = 9
    noise_level: float = 0.2

    max_episode_length: int = 1500

    reach_target_threshold: int = 2

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

        robot_dim: int = 1

    class render:
        scale: int = 5
        robot_dim_vis: int = 4   # d
        robot_color: tuple = (255, 150, 0)   # (B, G, R)

        target_dim_vis: int = 5  # d
        target_color: tuple = (0, 90, 250)
        target_thickness: int = 5


class FreeSpaceActorCriticCfg(ActorCriticCfg):

    actor_hidden_dims: list[int] = [32, 16, 8]
    critic_hidden_dims: list[int] = [32, 16, 8]
    distribution_type: str = 'beta'  # ['normal', 'beta']