from abc import ABC, abstractmethod

import numpy as np
import torch

from cfg.env_cfg import EnvCfg


class BaseEnv(ABC):

    def __init__(
        self,
        env_cfg: EnvCfg
    ) -> None:
        super(BaseEnv, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.env_cfg = env_cfg

        self.num_actions = env_cfg.num_actions
        self.img_latent_dim = env_cfg.num_img_latent
        self.num_obs = env_cfg.num_obs
        self.num_actor_input = self.num_obs
        self.num_critic_input = self.num_obs

        self.sim_width = env_cfg.sim.sim_width
        self.sim_height = env_cfg.sim.sim_height
        self.robot_dim = env_cfg.sim.robot_dim

        self.num_envs = env_cfg.num_envs

        self.dt = env_cfg.dt

        self.mu = env_cfg.mu
        self.frequency = env_cfg.frequency
           
        self.max_episode_length = env_cfg.max_episode_length

        self.img_model = None

    def _init_buffers(self):
        self.pos = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        history_length = self.env_cfg.history_length
        self.pos_history = torch.zeros((self.num_envs, history_length, 2), dtype=torch.float32, device=self.device)

        self.obs = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        
        self.targets = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.in_vessel = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)

        self.distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.last_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.step_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.reset_buffer = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.extras = {}

    def _create_sim(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, reset_ids):
        '''Reset the environments specified by reset_ids.
        '''
        raise NotImplementedError

    def get_observations(self):
        return self.obs
    
    @abstractmethod
    def compute_observation(self):
        '''Compute the observations for all environments and fill in self.obs.
        '''
        raise NotImplementedError
   
    @abstractmethod
    def step(self, actions: torch.Tensor):
        '''
        Process a simulation step in the environment.

        Args:
            actions (torch.Tensor): The actions to take, with shape (num_envs, num_actions).

        Returns:
            tuple: A tuple containing:
                - obs (torch.Tensor): The observations, with shape (num_envs, num_obs).
                - rewards (torch.Tensor): The rewards, with shape (num_envs,).
                - reset_buffer (torch.Tensor): The reset buffer, with shape (num_envs,).
                - extras (dict): Additional information.
        '''
        return (
            self.obs,
            self.rewards,
            self.reset_buffer,
            self.extras
        )

    @abstractmethod
    def render(
        self,
        mode: str = 'selected', 
        idx: int = 0,
        options: dict = {}
    ) -> np.ndarray:
        """
        Render the current state of the environment.

        Args:
            mode (str): Rendering mode, options are 'selected' (default) or 'all'.
            idx (int): Environment index to render, default is 0.
            options (dict): Additional rendering options, default is empty.

        Returns:
            np.ndarray: The rendered image as a NumPy array.
        """
        raise NotImplementedError

    def check_termination(self):
        raise NotImplementedError