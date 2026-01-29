# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from collections import deque
from datetime import datetime
import os
import statistics
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_critic import ActorCritic
from cfg.alg_cfg import ActorCriticCfg, RunnerCfg, PPOCfg
from env.base_env import BaseEnv
from .normalizer import EmpiricalNormalization
from .ppo import PPO


class OnPolicyRunner:

    def __init__(
        self,
        env: BaseEnv,
        actor_critic_cfg: ActorCriticCfg,
        ppo_cfg: PPOCfg,
        runner_cfg: RunnerCfg,
        visualize: bool = False,
        render_idx: int = 0,
    ):
        self.device = env.device
        self.env = env
        self.runner_cfg = runner_cfg
        
        self.visualize = visualize
        self.render_idx = render_idx

        obs = env.get_observations()

        actor_critic = ActorCritic(
            num_actor_input=env.num_actor_input,
            num_critic_input=env.num_critic_input,
            num_actions=env.num_actions,
            actor_hidden_dims=actor_critic_cfg.actor_hidden_dims,
            critic_hidden_dims=actor_critic_cfg.critic_hidden_dims,
            activation=actor_critic_cfg.activation,
            distribution_type=actor_critic_cfg.distribution_type,
        ).to(self.device)

        assert env.env_cfg.action_distribution_type == actor_critic_cfg.distribution_type, \
            f"Environment action distribution type {env.env_cfg.action_distribution_type} does not match " \
            f"ActorCritic distribution type {actor_critic_cfg.distribution_type}"

        self.alg = PPO(
            env=env,
            actor_critic=actor_critic,
            num_learning_epochs=ppo_cfg.num_learning_epochs,
            num_mini_batches=ppo_cfg.num_mini_batches,
            clip_param=ppo_cfg.clip_param,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.gae_lambda,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            learning_rate=ppo_cfg.learning_rate,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_clipped_value_loss=ppo_cfg.use_clipped_value_loss,
            schedule=ppo_cfg.schedule,
            desired_kl=ppo_cfg.desired_kl,
            device=self.device,
        )

        self.num_steps_per_env = runner_cfg.batch_size // self.env.num_envs
        self.save_interval = runner_cfg.save_interval

        if self.runner_cfg.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[env.num_obs], until=1.0e8).to(self.device)
        else:
            # identity mapping indicates no normalization
            self.obs_normalizer = torch.nn.Identity().to(self.device)

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=[self.env.num_actor_input],
            action_shape=[self.env.num_actions],
        )

        # log
        self.log_dir = runner_cfg.log_dir
        self.log_dir = os.path.join(runner_cfg.log_dir, type(env).__name__)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        current_date = datetime.now().strftime("%Y%m%d")

        log_dirs = [d for d in os.listdir(self.log_dir) if d.startswith(current_date)]
        existing_ids = []
        for d in log_dirs:
            try:
                id_str = d.split("_")[-1]
                if id_str.isdigit():
                    existing_ids.append(int(id_str))
            except IndexError:
                continue
        new_id = max(existing_ids) + 1 if existing_ids else 0

        self.log_dir = os.path.join(self.log_dir, f"{current_date}_{runner_cfg.batch_size}_{env.num_envs}_{new_id}")

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        obs = self.env.get_observations()
        obs = obs.to(self.device)           # (num_envs, obs_dim)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):   
                    actions = self.alg.act(obs)   # (num_envs, num_actions)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    if self.visualize:
                        self.env.render('selected', self.render_idx)

                    # Normalize observations
                    obs = self.obs_normalizer(obs)

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        
                        if "avg_success_rate_100" in infos:
                            avg_success_rate_100 = infos["avg_success_rate_100"]

                        if "total_num_trajs" in infos:
                            total_num_trajs = infos["total_num_trajs"]

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs)

            mean_value_loss, mean_surrogate_loss, mean_kl = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
            
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""

        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                        
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if locs["avg_success_rate_100"] is not None:
            self.writer.add_scalar("Evaluation/avg_success_rate_100", locs["avg_success_rate_100"], locs["it"])
            ep_string += f"""{f'Mean success rate (100):':>{pad}} {locs["avg_success_rate_100"]:.4f}\n"""

        if locs["total_num_trajs"] is not None:
            self.writer.add_scalar("Evaluation/total_num_trajs", locs["total_num_trajs"], locs["it"])
            ep_string += f"""{f'Total num trajs:':>{pad}} {locs["total_num_trajs"]:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.runner_cfg.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        if self.env.img_model is not None:
            saved_dict["img_model_state_dict"] = self.env.img_model.state_dict()
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.runner_cfg.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        if self.env.img_model is not None:
            self.env.img_model.load_state_dict(loaded_dict["img_model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]