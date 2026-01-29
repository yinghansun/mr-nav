from collections import deque
import os
import random
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from cfg.vessel_env_cfg import VesselEnvCfg
try:
    from .freespace_env import FreeSpace
except ImportError:
    from env.freespace_env import FreeSpace
from utils.vis_utils import draw_star


class VesselEnv(FreeSpace):

    def __init__(
        self,
        env_cfg: VesselEnvCfg,
    ) -> None:
        super().__init__(env_cfg)
        self.env_cfg = env_cfg
        self._init_virtual_lidar(env_cfg.lidar_max_depth, env_cfg.num_lidar_directions)

    def _init_buffers(self):
        super()._init_buffers()
        self.last_obstacle_distances = torch.zeros((self.num_envs, self.env_cfg.num_lidar_directions), dtype=torch.float32, device=self.device)

        self.rgb_imgs_tensor = torch.zeros((self.num_envs, 3, self.sim_width, self.sim_height), dtype=torch.uint8, device=self.device)
        self.safe_regions = torch.zeros((self.num_envs, 15000, 2), dtype=torch.long, device=self.device)

        self.success_record_buffer_100 = deque(maxlen=100)
        self.success_record_buffer_100.append(0)  # Initialize with 0 to avoid empty buffer issues
        self.extras["avg_success_rate_100"] = 0.
        self.extras["total_num_trajs"] = -self.num_envs
        self.extras["num_success_trajs"] = 0

    def _create_sim(self):        
        image_files = [f for f in os.listdir(self.env_cfg.sim.sim_vessel2d_dataset_path)]
        
        num_available_imgs = len(image_files)
        num_selected_imgs = min(self.num_envs, num_available_imgs)
        
        self.safe_regions = torch.zeros((self.num_envs, 15000, 2), dtype=torch.long, device=self.device)
        self.binary_imgs_tensor = torch.zeros((self.num_envs, self.sim_height, self.sim_width), dtype=torch.uint8, device=self.device)
        self.rgb_imgs_tensor = torch.zeros((self.num_envs, 3, self.sim_width, self.sim_height), dtype=torch.uint8, device=self.device)

        transform = transforms.Compose([transforms.ToTensor(),])

        selected_indices = random.sample(range(num_available_imgs), num_selected_imgs)
        for img_id, img_idx in tqdm(enumerate(selected_indices), total=num_selected_imgs, desc="Loading simulated envs"):
            img_file = image_files[img_idx]
            img_path = os.path.join(self.env_cfg.sim.sim_vessel2d_dataset_path, img_file)

            if img_path.endswith('.png') or img_path.endswith('.PNG'):
                img: torch.Tensor = 255 * transform(Image.open(img_path)).to(self.device).transpose(2, 1)
            else:
                img: torch.Tensor = torch.load(img_path).to(self.device)
                
            self.rgb_imgs_tensor[img_id, ...] = img
            safe_region_mask = (img[0, ...] == 255.)
            safe_region = torch.nonzero(safe_region_mask)
            self.safe_regions[img_id, :safe_region.shape[0], :] = safe_region

        for img_id in tqdm(range(num_selected_imgs, self.num_envs), total=self.num_envs - num_selected_imgs, desc="Reusing images"):
            reuse_id = img_id % num_selected_imgs
            self.rgb_imgs_tensor[img_id] = self.rgb_imgs_tensor[reuse_id]
            self.safe_regions[img_id] = self.safe_regions[reuse_id]

    def reset(self, reset_ids):
        self._init_robot_pos(reset_ids)
        self._sample_cmds(reset_ids)

        self.step_counter[reset_ids] = 0
        self.last_actions[reset_ids] = 0.
        self.last_distance[reset_ids] = 0.
        self.last_obstacle_distances[reset_ids] = 0.

        self.pos_history[reset_ids, ...] = 0.

        if len(reset_ids) > 0:
            for key in self.reward_episode_sums.keys():
                self.extras["episode"]['rew_' + key] = torch.mean(self.reward_episode_sums[key][reset_ids])
                self.reward_episode_sums[key][reset_ids] = 0.
            if hasattr(self, 'targets_reached'):
                self.success_record_buffer_100.extend(self.targets_reached[reset_ids].float().cpu().tolist())
                self.extras["avg_success_rate_100"] = sum(self.success_record_buffer_100) / len(self.success_record_buffer_100)
                self.extras["num_success_trajs"] += int(sum(self.targets_reached[reset_ids].float().cpu().tolist()))
            self.extras["total_num_trajs"] += len(reset_ids)

        if hasattr(self, 'render_id'):
            if not self.repeat_plot:
                if self.render_id in reset_ids:
                    if hasattr(self, 'img_rgb_numpy'):
                        delattr(self, 'img_rgb_numpy')

    def compute_observation(self):
        self.get_virtual_lidar_scans()
        self.obs = torch.cat((
            self.intersection_points.reshape(self.num_envs, -1),
            self.targets,
            self.pos,
            self.actions,
            # self.vel,
        ), dim=-1)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:

        self.step_counter += 1

        self._process_actions(actions)

        # update dynamics
        alpha = self.actions[:, 0]
        self.disturbance = torch.randn_like(self.vel) * (self.mu * self.frequency * self.env_cfg.noise_level)
        self.vel[:, 0] = self.mu * self.frequency * torch.sin(alpha)
        self.vel[:, 1] = self.mu * self.frequency * torch.cos(alpha)
        self.vel += self.disturbance
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
        self.render_id = idx
        self.repeat_plot = False if 'repeat_plot' not in options else options['repeat_plot']
        traj_color = (180, 200, 0) if 'traj_color' not in options else options['traj_color']
        non_vascular_area_color = [0, 255, 0] if 'non_vascular_area_color' not in options else options['non_vascular_area_color']

        if mode == 'selected':
            if not hasattr(self, 'img_rgb_numpy'):
                # (num_channels, width, height) -> (height, width, num_channels)
                img_rgb_numpy = self.rgb_imgs_tensor[idx].permute(2, 1, 0).cpu().numpy()

                non_vascular_area = np.all(img_rgb_numpy == non_vascular_area_color, axis=-1)
                img_rgb_numpy[non_vascular_area] = [70, 70, 70]

                scale_factor = self.env_cfg.render.scale
                self.img_rgb_numpy = cv2.resize(img_rgb_numpy, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

                height, width = self.img_rgb_numpy.shape[:2]
                target_pos = self.targets[idx].cpu().numpy()
                target_pos = np.clip(target_pos * scale_factor, 0, [width-1, height-1]).round().astype(int)
                target_size_vis = (scale_factor * self.env_cfg.render.target_dim_vis) / 2   # r = d / 2
                draw_star(self.img_rgb_numpy, tuple(target_pos), int(target_size_vis), self.env_cfg.render.target_color, self.env_cfg.render.target_thickness)

            scale_factor = self.env_cfg.render.scale
            img_rgb_numpy_ = self.img_rgb_numpy.copy()
            height, width = self.img_rgb_numpy.shape[:2]

            if self.repeat_plot:
                for idx, pos in enumerate(self.pos_history[idx]):
                    if idx > 0:
                        pos_np = pos.cpu().numpy()
                        pos_np = np.clip(pos_np * scale_factor, 0, [width-1, height-1]).round().astype(int)
                        cv2.circle(self.img_rgb_numpy, tuple(pos_np), 2, traj_color, -1)

            robot_pos = self.pos[idx].cpu().numpy()
            robot_pos = np.clip(robot_pos * scale_factor, 0, [width-1, height-1]).round().astype(int)
            robot_size_vis = (scale_factor * self.env_cfg.render.robot_dim_vis) / 2     # r = d / 2
            if self.step_counter[idx] > 0:
                cv2.circle(self.img_rgb_numpy, tuple(robot_pos), 2, traj_color, -1)
                cv2.circle(img_rgb_numpy_, tuple(robot_pos), int(robot_size_vis), self.env_cfg.render.robot_color, -1)

            intersection_points = self.intersection_points[idx]
            for point in intersection_points:
                point_np = point.cpu().numpy()
                point_np = np.clip(point_np * scale_factor, 0, [width-1, height-1]).round().astype(int)
                intersection_pts_size_vis = scale_factor * self.env_cfg.render.intersection_pts_dim_vis
                # cv2.circle(img_rgb_numpy, tuple(point_np), intersection_pts_size_vis, (255, 0, 180), -1)

                shadow_color = (100, 0, 120, 20)
                highlight_color = (255, 100, 220, 100)

                overlay = img_rgb_numpy_.copy()
                cv2.circle(overlay, (point_np[0]+1, point_np[1]+1), intersection_pts_size_vis + 2, shadow_color[:3], -1)
                cv2.circle(overlay, tuple(point_np), intersection_pts_size_vis, highlight_color[:3], -1)

                alpha_shadow = shadow_color[3] / 255
                alpha_highlight = highlight_color[3] / 255

                img_rgb_numpy_ = cv2.addWeighted(overlay, alpha_shadow, img_rgb_numpy_, 1 - alpha_shadow, 0)
                img_rgb_numpy_ = cv2.addWeighted(overlay, alpha_highlight, img_rgb_numpy_, 1 - alpha_highlight, 0)
        else:
            raise NotImplementedError

        cv2.imshow('Blood Vessel Environment', img_rgb_numpy_)
        cv2.waitKey(1)

        return self.img_rgb_numpy

    def _sample_cmds(self, reset_ids):
        for i, env_id in enumerate(reset_ids):
            safe_points = self.safe_regions[env_id]
            non_zero_mask = torch.any(safe_points != 0, dim=1)
            valid_safe_points = safe_points[non_zero_mask]

            sample_region_min = 5
            sample_region_max = self.sim_width - 5
            valid_safe_points = valid_safe_points[
                (valid_safe_points[:, 0] >= sample_region_min) & 
                (valid_safe_points[:, 0] <= sample_region_max) &
                (valid_safe_points[:, 1] >= sample_region_min) & 
                (valid_safe_points[:, 1] <= sample_region_max)
            ]

            if len(valid_safe_points) > 0:
                selected_index = torch.randint(0, len(valid_safe_points), (1,))
                new_target = valid_safe_points[selected_index].float()
                self.targets[env_id] = new_target.squeeze()
            else:
                print(f"Warning: No valid safe points for environment {env_id}")

    def _init_robot_pos(self, reset_ids):
        for i, env_id in enumerate(reset_ids):
            safe_points = self.safe_regions[env_id]
            non_zero_mask = torch.any(safe_points != 0, dim=1)
            valid_safe_points = safe_points[non_zero_mask]

            if len(valid_safe_points) > 0:
                selected_index = torch.randint(0, len(valid_safe_points), (1,))
                new_position = valid_safe_points[selected_index].float()
                self.pos[env_id] = new_position.squeeze()
            else:
                print(f"Warning: No valid safe points for environment {env_id}")

    def _check_in_vessel(self):
        if self.robot_dim == 1:
            env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 2)
            pos_x = self.pos[:, 0].long().clamp(max=self.sim_width-1)
            pos_y = self.pos[:, 1].long().clamp(max=self.sim_height-1)
            self.in_vessel = self.rgb_imgs_tensor[env_indices[:, 0], 0, pos_x, pos_y] > 0.5
        else:
            left_lower_bound_bias = -self.robot_dim / 2 * torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
            right_upper_bound_bias = self.robot_dim / 2 * torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
            left_upper_bound_bias = -self.robot_dim / 2 * torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
            left_upper_bound_bias[:, 1] = -left_upper_bound_bias[:, 1]
            right_lower_bound_bias = self.robot_dim / 2 * torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device)
            right_lower_bound_bias[:, 1] = -right_lower_bound_bias[:, 1]

            left_upper_bound_pos = (self.pos + left_upper_bound_bias).long().clamp(max=self.sim_width-1)
            left_lower_bound_pos = (self.pos + left_lower_bound_bias).long().clamp(max=self.sim_width-1)
            right_upper_bound_pos = (self.pos + right_upper_bound_bias).long().clamp(max=self.sim_width-1)
            right_lower_bound_pos = (self.pos + right_lower_bound_bias).long().clamp(max=self.sim_width-1)

            env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 2)

            left_upper_values = self.rgb_imgs_tensor[env_indices[:, 0], 0, left_upper_bound_pos[:, 0], left_upper_bound_pos[:, 1]]
            left_lower_values = self.rgb_imgs_tensor[env_indices[:, 0], 0, left_lower_bound_pos[:, 0], left_lower_bound_pos[:, 1]]
            right_upper_values = self.rgb_imgs_tensor[env_indices[:, 0], 0, right_upper_bound_pos[:, 0], right_upper_bound_pos[:, 1]]
            right_lower_values = self.rgb_imgs_tensor[env_indices[:, 0], 0, right_lower_bound_pos[:, 0], right_lower_bound_pos[:, 1]]

            left_upper_in_vessel = left_upper_values > 0.5
            left_lower_in_vessel = left_lower_values > 0.5
            right_upper_in_vessel = right_upper_values > 0.5
            right_lower_in_vessel = right_lower_values > 0.5

            self.in_vessel = left_upper_in_vessel & left_lower_in_vessel & right_upper_in_vessel & right_lower_in_vessel

            self.in_vessel = self.in_vessel.squeeze()      # (num_envs,)

    def _init_virtual_lidar(self, max_depth: int = 20, num_directions: int = 36):
        assert 360 % num_directions == 0, "num_directions must be a divisor of 360"

        self.num_lidar_directions = num_directions
        self.lidar_max_depth = max_depth

        angle_step = 360 / num_directions
        angles = torch.arange(0, 360, angle_step)
        directions = torch.stack([
            torch.cos(torch.deg2rad(angles)), torch.sin(torch.deg2rad(angles))
        ], dim=1).float().to(self.device)  # (num_directions, 2)

        self.lidar_directions = directions.unsqueeze(0).repeat(self.num_envs, 1, 1)   # (num_envs, num_directions, 2)

        self.env_idx_tensor = torch.arange(self.num_envs).unsqueeze(1).to(self.device)

    def get_virtual_lidar_scans(self) -> torch.Tensor:
        """
        Find all intersection points in the image from the robot's position.

        Args:
            max_depth (int, optional): The maximum depth of lidar scan. Defaults to 20.
            num_directions (int, optional): The number of directions of lidar scan 
                (virtual lidar's resolution). Defaults to 8.

        Returns:
            torch.Tensor: An array of lidar scanned points, with shape (num_env, num_directions, 2).
        """
        previous_pts = self.pos.unsqueeze(1).repeat(1, self.num_lidar_directions, 1)  # (num_envs, num_directions, 2)

        for _ in range(self.lidar_max_depth):
            current_pts = previous_pts + self.lidar_directions

            current_pts[:, :, 0] = torch.clamp(current_pts[:, :, 0], 0, self.sim_width - 1)
            current_pts[:, :, 1] = torch.clamp(current_pts[:, :, 1], 0, self.sim_height - 1)
            
            invalid_mask = self.rgb_imgs_tensor[
                self.env_idx_tensor,
                0,
                current_pts[:, :, 0].long(), 
                current_pts[:, :, 1].long(), 
            ] == 0
            current_pts[invalid_mask] = previous_pts[invalid_mask]

            previous_pts = current_pts.clone()

        self.intersection_points = current_pts.long()

    def _reward_obstacle(self):
        distances = torch.norm(self.intersection_points - self.pos.unsqueeze(1), dim=2)  # (num_envs, num_directions)
        mask = distances < 10
        filtered_distances = distances * mask.float()
        return filtered_distances.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _reward_log_obstacle(self):
        distances = torch.norm(self.intersection_points - self.pos.unsqueeze(1), dim=2)  # (num_envs, num_directions)
        eps=1e-6
        distances = torch.clamp(distances, min=eps)
        return torch.log(distances).mean(dim=1)

    def _reward_inverse_obstacle(self):
        distances = torch.norm(self.intersection_points - self.pos.unsqueeze(1), dim=2)  # (num_envs, num_directions)
        eps = 3.
        return 1. / (distances + eps).mean(dim=1)



if __name__ == '__main__':
    env_cfg = VesselEnvCfg()
    env = VesselEnv(env_cfg)
    
    for _ in range(299):
        action = torch.rand((env_cfg.num_envs, 1))
        obs, reward, done, _ = env.step(action)
        env.render()
    
    env.render()
    cv2.waitKey(0)
    cv2.destroyAllWindows()