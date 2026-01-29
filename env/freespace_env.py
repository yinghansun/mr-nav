from collections import deque

import cv2
import numpy as np
import torch

try:
    from .base_env import BaseEnv
except ImportError:
    from base_env import BaseEnv
from cfg.freespace_cfg import FreespaceCfg
from utils.class2dict import class_to_dict


class FreeSpace(BaseEnv):

    def __init__(
        self,
        env_cfg: FreespaceCfg
    ) -> None:
        super().__init__(env_cfg)

        self._init_buffers()
        self._create_sim()
        self._prepare_reward_functions()
        self.reset(reset_ids=[i for i in range(self.num_envs)])

    def _init_buffers(self):
        super()._init_buffers()

        self.rgb_imgs_tensor = torch.zeros((self.num_envs, 3, self.sim_width, self.sim_height), dtype=torch.uint8, device=self.device)
        self.safe_regions = torch.zeros((self.num_envs, 15000, 2), dtype=torch.long, device=self.device)

        self.success_record_buffer_100 = deque(maxlen=100)
        self.success_record_buffer_100.append(0)  # Initialize with 0 to avoid empty buffer issues
        self.extras["avg_success_rate_100"] = 0.
        self.extras["total_num_trajs"] = 0.

        self.last_delta_pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

    def _create_sim(self):
        # Create a white image tensor
        self.img_tensor = 255 * torch.ones(
            (self.sim_width, self.sim_height, 3), 
            dtype=torch.uint8, 
            device=self.device
        )
        self._get_pos_sampling_region()

    def reset(self, reset_ids):
        self._init_robot_pos(reset_ids)
        self._sample_cmds(reset_ids)

        self.step_counter[reset_ids] = 0
        self.last_actions[reset_ids] = 0.
        self.last_distance[reset_ids] = 0.
        # self.last_distance[reset_ids] = 9999999.

        self.last_delta_pos[reset_ids] = self.targets[reset_ids] - self.pos[reset_ids]

        self.pos_history[reset_ids, ...] = 0.

        if len(reset_ids) > 0:
            for key in self.reward_episode_sums.keys():
                self.extras["episode"]['rew_' + key] = torch.mean(self.reward_episode_sums[key][reset_ids])
                self.reward_episode_sums[key][reset_ids] = 0.
            if hasattr(self, 'targets_reached'):
                self.success_record_buffer_100.extend(self.targets_reached[reset_ids].float().cpu().tolist())
                self.extras["avg_success_rate_100"] = sum(self.success_record_buffer_100) / len(self.success_record_buffer_100)
            self.extras["total_num_trajs"] += len(reset_ids)


    def get_observations(self):
        return self.obs
    
    def compute_observation(self):
        self.obs = torch.cat((
            self.targets - self.pos,
            self.actions,
        ), dim=-1)
        self.last_delta_pos = self.targets - self.pos

    def step(self, actions: torch.Tensor):
        self.step_counter += 1

        self._process_actions(actions)
        
        # update dynamics
        alpha = self.actions[:, 0]
        disturbance = torch.randn_like(self.vel) * (self.mu * self.frequency * self.env_cfg.noise_level)
        self.vel[:, 0] = self.mu * self.frequency * torch.sin(alpha)
        self.vel[:, 1] = self.mu * self.frequency * torch.cos(alpha)
        self.vel += disturbance
        self.pos += self.vel * self.dt

        self.pos_history = self.pos_history[:, 1:, :]
        self.pos_history = torch.cat((self.pos_history, self.pos.unsqueeze(1)), dim=1)

        self.compute_observation()
        self.check_termination()
        self._compute_reward()

        reset_ids = self.reset_buffer.nonzero(as_tuple=False).flatten()
        self.reset(reset_ids)

        self.last_actions[:] = self.actions[:]

        return (
            self.obs,
            self.rewards,
            self.reset_buffer,
            self.extras,
        )
    
    def _process_actions(self, actions: torch.Tensor):
        if self.env_cfg.action_degree_type == 'radian':
            clip_actions = 3.1415926
        else:
            clip_actions = 180.

        if self.env_cfg.action_distribution_type == 'normal':
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        elif self.env_cfg.action_distribution_type == 'beta':
            self.actions = (actions * 2 - 1) * clip_actions
        else:
            raise ValueError(f"Unknown action distribution type: {self.env_cfg.action_distribution_type}")
        self.actions = self.actions.to(self.device)
        
        if self.env_cfg.action_degree_type == 'degree':
            self.actions = self._degree2radian(self.actions)

    def render(
        self, 
        mode: str = 'selected', 
        idx: int = 0,
        options: dict = {}
    ) -> np.ndarray:
        img = self.img_tensor.cpu().numpy().transpose(1, 0, 2).copy()

        if mode == 'selected':
            scale_factor = self.env_cfg.render.scale
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
            
            height, width, _ = img.shape
            robot_pos = self.pos[idx, :].cpu().numpy().astype(int)
            robot_pos = np.clip(robot_pos * scale_factor, 0, [width-1, height-1]).round().astype(int)
            robot_size_vis = scale_factor * self.env_cfg.render.robot_dim_vis
            
            target_pos = self.targets[idx, :].cpu().numpy().astype(int)
            target_pos = np.clip(target_pos * scale_factor, 0, [width-1, height-1]).round().astype(int)
            target_size_vis = scale_factor * self.env_cfg.render.target_dim_vis

            cv2.circle(img, tuple(robot_pos), robot_size_vis, self.env_cfg.render.robot_color, -1)
            cv2.circle(img, tuple(target_pos), target_size_vis, self.env_cfg.render.target_color, -1)
        else:
            raise NotImplementedError
        
        cv2.imshow('Blood Vessel Environment', img)
        cv2.waitKey(1)

        return img

    def _get_pos_sampling_region(self):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.robot_dim, self.sim_height), 
            torch.arange(self.robot_dim, self.sim_width), 
            indexing='ij'
        )
        self.safe_regions = torch.stack((x_coords.flatten(), y_coords.flatten()), dim=1).to(self.device)

    def _sample_cmds(self, reset_ids):
        num_reset = len(reset_ids)
        selected_indices = torch.randint(0, len(self.safe_regions), (num_reset,), device=self.device)
        new_targets = self.safe_regions[selected_indices].float()
        self.targets[reset_ids] = new_targets

    def _init_robot_pos(self, reset_ids):
        num_reset = len(reset_ids)
        selected_indices = torch.randint(0, len(self.safe_regions), (num_reset,), device=self.device)
        new_positions = self.safe_regions[selected_indices].float()
        self.pos[reset_ids] = new_positions

    def _check_in_vessel(self):
        pass

    def check_termination(self):
        time_out_buffer = (self.step_counter > self.max_episode_length)  # T: time out (need to be reset), F: not time out
        self.check_out_of_bound()     # T: out of bound (need to be reset) F: not out of bound
        self._check_in_vessel()
        fail_buffer = ~self.in_vessel # T: not in vessel (need to be reset), F: in vessel
        self._check_target_reached()  # T: target reached (need to be reset) F: not reach
        self.reset_buffer = time_out_buffer | fail_buffer | self.targets_reached | self.out_of_bound_buffer

    def check_out_of_bound(self):
        self.out_of_bound_buffer = \
            (self.pos[:, 0] < 0) | \
            (self.pos[:, 0] >= self.sim_width) | \
            (self.pos[:, 1] < 0) | \
            (self.pos[:, 1] >= self.sim_height)

    def _check_target_reached(self):
        threshold = self.env_cfg.reach_target_threshold
        distances = torch.norm(self.pos - self.targets, dim=1)
        self.targets_reached = distances <= threshold
        return self.targets_reached

    def _prepare_reward_functions(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        self.reward_scales = class_to_dict(self.env_cfg.reward.scales)

        self.extras["episode"] = {}

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

            self.extras["episode"]['rew_' + key] = 0.

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.reward_episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                                    for name in self.reward_scales.keys()}

    def _compute_reward(self):
        self.rewards[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            cur_reward_term = self.reward_functions[i]() * self.reward_scales[name]
            self.rewards += cur_reward_term
            self.reward_episode_sums[name] += cur_reward_term

            if torch.isinf(cur_reward_term).any():
                print(name)

    def _degree2radian(self, degree):
        return degree * np.pi / 180

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_reach_target(self):
        return self.targets_reached.float()
    
    def _reward_collision(self):
        return (~self.in_vessel).float()
    
    def _reward_distance_change(self):
        distance = torch.norm(self.pos - self.targets, dim=1)
        diff = self.last_distance - distance
        self.last_distance = distance
        return diff
    
    def _reward_distance(self):
        return torch.norm(self.pos - self.targets, dim=1)

    def _reward_time_accumulation(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
    
    def _reward_one_over_distance(self):
        return 1. / (torch.norm(self.pos - self.targets, dim=1) + 1e-6)


if __name__ == '__main__':
    env_cfg = FreespaceCfg()
    env_cfg.num_envs = 2
    env = FreeSpace(env_cfg)

    for _ in range(299):
        action = torch.tensor([80, 181,]).reshape(2, 1)
        obs, reward, done, _ = env.step(action)
        env.render()

    env.render()
    cv2.waitKey(0)
    cv2.destroyAllWindows()