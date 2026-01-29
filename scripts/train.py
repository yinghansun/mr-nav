import argparse
import gc

import torch

from alg.on_policy_runner import OnPolicyRunner
from cfg.alg_cfg import PPOCfg, RunnerCfg
from cfg.freespace_cfg import FreespaceCfg, FreeSpaceActorCriticCfg
from cfg.vessel_env_cfg import VesselEnvCfg, VesselEnvActorCriticCfg
from env.freespace_env import FreeSpace
from env.vessel_env import VesselEnv
from utils.log_utils import save_configs, save_env_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", type=str, default="freespace_env", help="Type of the simulation environment")
    parser.add_argument("--num_run", type=int, default=1, help="Number of runs")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--vis", type=bool, default=False, help="Visualize the environment during training")
    parser.add_argument("--render_idx", type=int, default=0, help="Environment index to visualize during training")
    return parser


def train(args: argparse.Namespace):
    assert args.env in ['freespace_env', 'vessel_env'], "Unsupported environment type"
    assert args.num_run > 0, "Number of runs must be positive"
    assert args.num_epoch > 0, "Number of epochs must be positive"

    for _ in range(args.num_run):
        if args.env == 'freespace_env':
            env_cfg = FreespaceCfg()
            env = FreeSpace(env_cfg)
            actor_critic_cfg = FreeSpaceActorCriticCfg()
        else:
            env_cfg = VesselEnvCfg()
            env = VesselEnv(env_cfg)
            actor_critic_cfg = VesselEnvActorCriticCfg()

        ppo_cfg = PPOCfg()
        runner_cfg = RunnerCfg()

        runner = OnPolicyRunner(
            env,
            actor_critic_cfg,
            ppo_cfg,
            runner_cfg,
            visualize=args.vis,
            render_idx=args.render_idx
        )

        log_dir = runner.log_dir
        save_configs(log_dir, env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg)
        if args.env == 'freespace_env':
            save_env_files(log_dir, ['freespace_env.py'])
        else:
            save_env_files(log_dir, ['vessel_env.py'])

        runner.learn(args.num_epoch)

        del runner, env, env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    train(args)