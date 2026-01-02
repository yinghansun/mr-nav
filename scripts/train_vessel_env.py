from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import ActorCriticCfg, PPOCfg, RunnerCfg
from cfg.vessel_env_cfg import VesselEnvCfg
from env.vessel_env import VesselEnv
from utils.log_utils import save_env_files, save_configs


# num_envs = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
# num_envs = [8192]

# num_envs = [64, 32, 16, 8]
# batch_sizes = [8192, 16384, 49152]

num_envs = [8192]
batch_size = 196608
# drs_weights = [-1, -10]
pbrs_weights = [0.5]


for num_env in num_envs:

    # for batch_size in batch_sizes:
    # for drs_weight in drs_weights:
    for pbrs_weight in pbrs_weights:

        for num_training_time in range(5):
            env_cfg = VesselEnvCfg()
            actor_critic_cfg = ActorCriticCfg()
            ppo_cfg = PPOCfg()
            runner_cfg = RunnerCfg()

            env_cfg.num_envs = num_env
            env_cfg.num_obs = 36 * 2 + 2 + 2 + 1
            env_cfg.action_distribution_type = 'normal'
            actor_critic_cfg.distribution_type = 'normal'
            env_cfg.action_degree_type = 'radian'

            # env_cfg.reward.scales.log_obstacle = 0.1
            # env_cfg.reward.scales.inverse_obstacle = 1.
            # env_cfg.reward.scales.distance = drs_weight
            env_cfg.reward.scales.distance_change = pbrs_weight

            actor_critic_cfg.actor_hidden_dims = [256, 128, 16]
            actor_critic_cfg.critic_hidden_dims = [256, 128, 16]

            runner_cfg.save_interval = 50
            # runner_cfg.num _steps_per_env = 1000
            # runner_cfg.batch_size =8192
            runner_cfg.batch_size = batch_size

            env = VesselEnv(env_cfg)

            runner = OnPolicyRunner(
                env,
                actor_critic_cfg,
                ppo_cfg,
                runner_cfg,
                visualize=False,
                render_idx=8191
            )

            log_dir = runner.log_dir
            save_configs(log_dir, env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg)
            save_env_files(log_dir, ['vessel_env.py'])

            runner.learn(500)

            del runner
            del env
            del env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg

            # 垃圾回收 + 释放 CUDA 缓存
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()