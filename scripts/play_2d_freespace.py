from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import ActorCriticCfg, PPOCfg, RunnerCfg
from cfg.freespace_cfg import FreespaceCfg

from env.freespace import FreeSpace

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


RECORD = False

env_cfg = FreespaceCfg()
actor_critic_cfg = ActorCriticCfg()
ppo_cfg = PPOCfg()
runner_cfg = RunnerCfg()

env_cfg.num_envs = 1

actor_critic_cfg.actor_hidden_dims = [16, 8, 2]
actor_critic_cfg.critic_hidden_dims = [16, 8, 2]

env_cfg.action_distribution_type = 'normal'
actor_critic_cfg.distribution_type = 'normal'
env_cfg.action_degree_type = 'radian'

env = FreeSpace(env_cfg)

runner = OnPolicyRunner(
    env,
    actor_critic_cfg,
    ppo_cfg,
    runner_cfg, 
    visualize=True               
)
# model_path = './saved_model/FreeSpace/128x128/model_999.pt'
model_path = './log/FreeSpaceAmp/20250828_196608_8192_0_yx/model_1999.pt'
runner.load(model_path)


if not RECORD:
    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(2999):
            obs = runner.obs_normalizer(obs)
            action = runner.alg.act(obs)
            obs, reward, done, _ = env.step(action)
            
            frame = env.render()

            if done:
                continue

else:
    fig, ax = plt.subplots()
    frames = []

    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(999):
            obs = runner.obs_normalizer(obs)
            action = runner.alg.act(obs)
            obs, reward, done, _ = env.step(action)
            
            frame = env.render()
            frames.append(frame)

            if done:
                frames.pop()
                break

    ani = animation.ArtistAnimation(
        fig, 
        [[ax.imshow(frame)] for frame in frames], 
        interval=50, 
        blit=True, 
        repeat_delay=1000
    )

    ani.save('./freespace_2d.gif', writer='pillow', fps=30)

    plt.close(fig)