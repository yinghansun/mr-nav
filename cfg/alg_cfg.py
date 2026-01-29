from typing import List


class ActorCriticCfg:

    actor_hidden_dims: List[int] = [256, 128, 16]
    critic_hidden_dims: List[int] = [256, 128, 16]
    activation: str = 'elu'
    distribution_type: str = 'normal'  # ['normal', 'beta']


class PPOCfg:

    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2

    entropy_coef: float = 0.01

    num_learning_epochs: int = 5
    num_mini_batches: int = 4  # mini-batch size = num_envs * nsteps / num_mini_batches
    learning_rate: float = 1e-3
    schedule: str = 'adaptive'  # ['adaptive', 'fixed']

    gamma: float = 0.99
    gae_lambda: float = 0.95

    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


class RunnerCfg:

    batch_size: int = 196608      # 8192 * 24

    empirical_normalization: bool = True

    # log
    save_interval: int = 50
    log_dir: str = './log'