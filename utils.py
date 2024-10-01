from collections import OrderedDict

import torch
import gymnasium as gym
from nsfr.common import get_nsfr_model, get_blender_nsfr_model
from neumann.common import get_neumann_model, get_blender_neumann_model
from nsfr.utils.common import load_module
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from stable_baselines3 import PPO
# from huggingface_sb3 import load_from_hub, push_to_hub

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NeuralBlenderActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 2), std=0.01)
        
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs
    
    
class CNNActor(nn.Module):
    def __init__(self, n_actions=18, ):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

def get_blender(env, blender_rules, device, train=True, blender_mode='logic', reasoner='nsfr', explain=False):
    assert blender_mode in ['logic', 'neural']
    if blender_mode == 'logic':
        if reasoner == 'nsfr':
            return get_blender_nsfr_model(env.name, blender_rules, device, train=train, explain=explain)
        elif reasoner == 'neumann':
            return get_blender_neumann_model(env.name, blender_rules, device, train=train, explain=explain)
    if blender_mode == 'neural':
        net = NeuralBlenderActor()
        net.to(device)
        return net
        # mlp_module_path = f"in/envs/{env.name}/mlp.py"
        # module = load_module(mlp_module_path)
        # return module.MLP(out_size=1, has_sigmoid=True, device=device)
    
    
def extract_policy_probs(NSFR, V_T, device):
    """not to be used. blender actor already computes this in the forward function."""
    batch_size = V_T.size(0)
    # extract neural_agent(img) and logic_agent(image)
    indices = []
    for i, atom in enumerate(NSFR.atoms):
        if "neural_agent" in str(atom):
            indices.append(i)
    for i, atom in enumerate(NSFR.atoms):
        if "logic_agent" in str(atom):
            indices.append(i)
    
    indices = torch.tensor(indices).to(device).unsqueeze(0)
    indices = indices.repeat(batch_size, 1)
    
    policy_probs = torch.gather(V_T, 1, indices)
    return policy_probs


def load_pretrained_stable_baseline_ppo(env, device):
    agent_path = "rl-baselines3-zoo/rl-trained-agents/ppo/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4.zip"
    policy_path = "rl-baselines3-zoo/rl-trained-agents/ppo/SeaquestNoFrameskip-v4_1/policy.pth"
    policy = torch.load(policy_path) #, map_location=device)
    
    # checkpoint = load_from_hub("ThomasSimonini/ppo-SeaquestNoFrameskip-v4", "ppo-SeaquestNoFrameskip-v4.zip")

    # # Because we using 3.7 on Colab and this agent was trained with 3.8 to avoid Pickle errors:
    # custom_objects = {
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     }
    # baseline_ppo = PPO.load(checkpoint, custom_objects=custom_objects)

    # saved_variables = torch.load(agent_path, map_location=device, weights_only=False)

    # baseline_ppo = PPO("MlpPolicy", env.raw_env, verbose=1)
    # saved_variables = torch.load(agent_path, map_location=device)
    # saved_variables = torch.load(agent_path, map_location=device)
    # baseline_ppo.load_state_dict(saved_variables["state_dict"])
    # baseline_ppo.to(device)
    # return baseline_ppo
    # Create policy object
    # model = cls(**saved_variables["data"])
    # Load weights
    # model.load_state_dict(saved_variables["state_dict"])
    
    # baseline_ppo = PPO("MlpPolicy", env.raw_env, verbose=1)
    baseline_ppo = PPO.load(agent_path, env=env.raw_env, device=device)
    return baseline_ppo
    # neural_pixel_actor =baseline_ppo.policy #.mlp_extractor.policy_net
    # base_path = "rl-baselines3-zoo/rl-trained-agents/a2c/SeaquestNoFrameskip-v4_1"
    
    
def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from cleanrl.cleanrl.ppo_atari import make_env
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs
    
def load_cleanrl_agent(pretrained, device):
    # from cleanrl.cleanrl.ppo_atari import Agent
    agent = CNNActor(n_actions=18) #, device=device, verbose=1)
    if pretrained:
        try:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
            agent.to(device)
        except RuntimeError:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth", map_location=torch.device('cpu')))
    else:
        agent.to(device)
    return agent


def load_logic_ppo(agent, path):
    new_actor_dic = OrderedDict()
    new_critic_dic = OrderedDict()
    dic = torch.load(path)
    for name, value in dic.items():
        if 'actor.' in name:
            new_name = name.replace('actor.', '') 
            new_actor_dic[new_name] = value
        if 'critic.' in name:
            new_name = name.replace('critic.', '') 
            new_critic_dic[new_name] = value
    agent.logic_actor.load_state_dict(new_actor_dic)
    agent.logic_critic.load_state_dict(new_critic_dic)
    return agent