import os
import shutil
import yaml

from .class2dict import class_to_dict


def save_env_files(log_dir, file_names):
    env_dir = os.path.join(log_dir, 'env')
    os.makedirs(env_dir, exist_ok=True)
    
    env_source_dir = 'env'
    for file in file_names:
        source_path = os.path.join(env_source_dir, file)
        if os.path.exists(source_path):
            shutil.copy(source_path, env_dir)
        else:
            print(f"Warning: File {file} not found in {env_source_dir}")
    
def save_configs(log_dir, env_cfg, actor_critic_cfg, ppo_cfg, runner_cfg):
    config_dir = os.path.join(log_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    configs = {
        'env_cfg.yaml': env_cfg,
        'actor_critic_cfg.yaml': actor_critic_cfg,
        'ppo_cfg.yaml': ppo_cfg,
        'runner_cfg.yaml': runner_cfg
    }
    
    for filename, cfg in configs.items():
        with open(os.path.join(config_dir, filename), 'w') as f:
            yaml.dump(class_to_dict(cfg), f, default_flow_style=False)