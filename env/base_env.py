import torch

from cfg.env_cfg import EnvCfg


class BaseEnv:

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

    def reset(self, reset_ids):
        raise NotImplementedError

    def get_observations(self):
        return self.obs
    
    def compute_observation(self):
        raise NotImplementedError
   
    def step(self, actions: torch.Tensor):
        return (
            self.obs,
            self.rewards,
            self.reset_buffer,
            self.extras
        )

    def render(self):
        raise NotImplementedError

    def check_termination(self):
        raise NotImplementedError