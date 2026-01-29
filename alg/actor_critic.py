import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Beta, Normal

from utils.mlp import MLP


class ActorCritic(nn.Module):

    def __init__(
        self,
        num_actor_input,
        num_critic_input,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        distribution_type="beta"
    ):
        super(ActorCritic, self).__init__()

        assert distribution_type in ["normal", "beta"], \
            "distribution_type must be either 'normal' or 'beta'"

        self.num_actor_obs = num_actor_input
        self.num_critic_obs = num_critic_input
        self.distribution_type = distribution_type

        if distribution_type == "normal":
            self.actor = MLP(
                input_size=num_actor_input,
                output_size=num_actions,
                hidden_size=actor_hidden_dims,
                activate_func=activation,
                print_info=True,
                name='Actor'
            )
            self.logstd = nn.Parameter(torch.zeros(num_actions))
        elif distribution_type == "beta":
            self.actor = MLP(
                input_size=num_actor_input,
                output_size=actor_hidden_dims[-1],
                hidden_size=actor_hidden_dims[:-1],
                activate_func=activation,
                print_info=True,
                name='Actor'
            )
            self.alpha_layer = nn.Linear(actor_hidden_dims[-1], num_actions)
            self.beta_layer = nn.Linear(actor_hidden_dims[-1], num_actions)

        self.critic = MLP(
            input_size=num_critic_input,
            output_size=1,
            hidden_size=critic_hidden_dims,
            activate_func=activation,
            print_info=True,
            name='Critic'
        )

        self.distribution = None
        Normal.set_default_validate_args = False
        Beta.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        if self.distribution_type == "normal":
            mean = self.actor(observations)  # (num_envs, num_actions)
            self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))
        elif self.distribution_type == "beta":
            hidden = self.actor(observations)
            # alpha and beta need to be larger than 1
            alpha = F.softplus(self.alpha_layer(hidden)) + 1.0
            beta = F.softplus(self.beta_layer(hidden)) + 1.0
            self.distribution = Beta(alpha, beta)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor) -> torch.Tensor:
        value = self.critic(critic_observations)
        return value