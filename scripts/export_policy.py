import torch

from alg.actor_critic import ActorCritic
from alg.normalizer import EmpiricalNormalization
from utils.export_onnx_model import export_onnx_model


class NormalizedPolicy(torch.nn.Module):
    def __init__(
        self, 
        actor_critic_model: ActorCritic, 
        num_obs: int,
    ) -> None:
        super(NormalizedPolicy, self).__init__()
        self.actor_critic_model = actor_critic_model
        self.obs_normalizer = EmpiricalNormalization(
            shape=[num_obs], 
            until=1.0e8
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        return self.actor_critic_model.act_inference(obs)
    

def load_pt_model(pt_path: str, actor_critic_model: ActorCritic, num_obs: int) -> NormalizedPolicy:
    loaded_dict = torch.load(pt_path)
    model = NormalizedPolicy(actor_critic_model, num_obs)
    model.actor_critic_model.load_state_dict(loaded_dict['model_state_dict'])
    model.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
    return model


if __name__ == '__main__':
    pt_path = './saved_model/2D_constant_vel_freespace/random_target/130x130/model_999.pt'
    actor_critic = ActorCritic(
        num_actor_input=3,
        num_critic_input=3,
        num_actions=1,
        actor_hidden_dims=[16, 8, 2],
        critic_hidden_dims=[16, 8, 2],
        activation='elu'
    )
    model = load_pt_model(pt_path, actor_critic, 3)

    desired_onnx_path = './saved_model/2D_constant_vel_freespace/random_target/130x130/model_999.onnx'
    input_dim = (1, 3)  # (batch size, input dim)
    export_onnx_model(desired_onnx_path, model, input_dim)