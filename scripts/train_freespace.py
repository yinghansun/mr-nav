from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import ActorCriticCfg, PPOCfg, RunnerCfg
from cfg.freespace_cfg import FreespaceCfg
from env.freespace import FreeSpace

from utils.log_utils import save_env_files, save_configs


map_sizes = [256, 512]
weights = ['None', 'DRS5e-4', 'PBRS0.5']
max_episode_lengths = [2000, 4000]

for map_size in map_sizes:
    for weight in weights:
        for num_training_time in range(5):
            env_cfg = FreespaceCfg()
            actor_critic_cfg = ActorCriticCfg()
            ppo_cfg = PPOCfg()
            runner_cfg = RunnerCfg()

            env_cfg.num_envs = 8192
            env_cfg.sim.sim_width = map_size
            env_cfg.sim.sim_height = map_size

            if map_size == 256:
                env_cfg.max_episode_length = 2000
            elif map_size == 512:
                env_cfg.max_episode_length = 4000

            if weight == 'None':
                env_cfg.reward.scales.distance = 0.
                env_cfg.reward.scales.distance_change = 0.
            elif weight == 'DRS5e-4':
                env_cfg.reward.scales.distance = -5e-4
                env_cfg.reward.scales.distance_change = 0.
            else:
                env_cfg.reward.scales.distance = 0.
                env_cfg.reward.scales.distance_change = 0.5

            actor_critic_cfg.actor_hidden_dims = [16, 8, 2]
            actor_critic_cfg.critic_hidden_dims = [16, 8, 2]

            env_cfg.action_distribution_type = 'beta'
            actor_critic_cfg.distribution_type = 'beta'
            env_cfg.action_degree_type = 'radian'

            env = FreeSpace(env_cfg)

            runner = OnPolicyRunner(
                env,
                actor_critic_cfg,
                ppo_cfg,
                runner_cfg, 
                visualize=False
            )

            log_dir = runner.log_dir
            save_configs(log_dir, env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg)
            save_env_files(log_dir, ['freespace.py'])

            runner.learn(500)

            del runner
            del env
            del env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg

            # 垃圾回收 + 释放 CUDA 缓存
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()