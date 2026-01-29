import argparse

from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import PPOCfg, RunnerCfg
from cfg.freespace_cfg import FreespaceCfg, FreeSpaceActorCriticCfg
from cfg.vessel_env_cfg import VesselEnvCfg, VesselEnvActorCriticCfg
from env.freespace_env import FreeSpace
from env.vessel_env import VesselEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="freespace_env", help="Type of the simulation environment")
    parser.add_argument("--model_path", type=str, default="./saved_model/FreeSpace/model_499.pt", help="Path to the model to load")
    parser.add_argument("--render_idx", type=int, default=0, help="Environment index to visualize during playing")
    return parser


def play(args: argparse.Namespace):
    assert args.env in ['freespace_env', 'vessel_env'], "Unsupported environment type"
    
    if args.env == 'freespace_env':
        env_cfg = FreespaceCfg()
        env_cfg.num_envs = 1
        env = FreeSpace(env_cfg)
        actor_critic_cfg = FreeSpaceActorCriticCfg()
    else:
        env_cfg = VesselEnvCfg()
        env_cfg.num_envs = 1
        env = VesselEnv(env_cfg)
        actor_critic_cfg = VesselEnvActorCriticCfg()

    ppo_cfg = PPOCfg()
    runner_cfg = RunnerCfg()

    runner = OnPolicyRunner(
        env,
        actor_critic_cfg,
        ppo_cfg,
        runner_cfg,
        visualize=False,
        render_idx=args.render_idx
    )

    model_path = args.model_path
    runner.load(model_path)

    obs = env.get_observations()
    for i in range(2999):
        obs = runner.obs_normalizer(obs)
        action = runner.alg.act(obs)
        obs, reward, done, _ = env.step(action)
        frame = env.render()

        if done:
            continue


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    play(args)