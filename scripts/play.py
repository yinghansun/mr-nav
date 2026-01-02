from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import ActorCriticCfg, PPOCfg, RunnerCfg
from cfg.vessel_env_cfg import VesselEnvCfg
from cfg.visual_test_cfg import VisualTestCfg

# from env.vessel_env import VesselEnv
from env.vessel_env_evaluation import VesselEnv

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


RECORD = True

env_cfg = VesselEnvCfg()
actor_critic_cfg = ActorCriticCfg()
ppo_cfg = PPOCfg()
runner_cfg = RunnerCfg()

env_cfg.num_envs = 1

env_cfg.img_idx = 524

# env_cfg.img_idx = 0
# env_cfg.sim.sim_vessel2d_dataset_path = './dataset/manually_generated/converted'

env_cfg.num_obs = 36 * 2 + 2 + 2 + 1

env_cfg.action_distribution_type = 'normal'
actor_critic_cfg.distribution_type = 'normal'
env_cfg.action_degree_type = 'radian'

actor_critic_cfg.actor_hidden_dims = [256, 128, 16]
actor_critic_cfg.critic_hidden_dims = [256, 128, 16]

visual_cfg = VisualTestCfg()
visual_cfg.threshold = 0
env = VesselEnv(env_cfg, mode='test', visual_test_cfg=visual_cfg)

runner = OnPolicyRunner(
    env,
    actor_critic_cfg,
    ppo_cfg,
    runner_cfg, 
    visualize=True               
)
model_path = './saved_model/VesselEnv/model_1400.pt'
# model_path = './log/VesselAmpEnv/20250821_786432_8192_0/model_1550.pt'
runner.load(model_path)

manual_setting = False

if not RECORD:
    with torch.no_grad():
        # env.compute_observation()
        obs = env.get_observations()
        for i in range(2999):
            obs = runner.obs_normalizer(obs)
            action = runner.alg.act(obs)
            obs, reward, done, _ = env.step(action)

            if not manual_setting:
                env.pos = torch.tensor([[10., 110.]]).to(env.device)
                env.targets = torch.tensor([[10, 90]]).to(env.device)
                # env.pos = torch.tensor([[15., 110.]]).to(env.device)
                # env.targets = torch.tensor([[110, 70]]).to(env.device)
                manual_setting = True

            frame = env.render()

            if done:
                # model_path = './saved_model/VesselEnv/model_1400.pt'
                manual_setting = False
                continue

else:
    fig, ax = plt.subplots()
    frames = []

    with torch.no_grad():
        obs = env.get_observations()
        for _ in range(2999):
            obs = runner.obs_normalizer(obs)
            action = runner.alg.act(obs)
            obs, reward, done, _ = env.step(action)
            
            if not manual_setting:
                env.pos = torch.tensor([[125., 70.]]).to(env.device)
                env.targets = torch.tensor([[15, 5]]).to(env.device)
                manual_setting = True

            frame = env.render()
            frame = frame[..., ::-1]   # BGR to RGB
            frames.append(frame)

            if done:
                frames.pop()
                break

    # ani = animation.ArtistAnimation(
    #     fig, 
    #     [[ax.imshow(frame)] for frame in frames], 
    #     interval=50, 
    #     blit=True, 
    #     repeat_delay=1000
    # )

    # ani.save('./freespace_2d.gif', writer='pillow', fps=30)

    last_frame = frames[-1]
    plt.imsave('./last_frame.png', last_frame)

    plt.close(fig)