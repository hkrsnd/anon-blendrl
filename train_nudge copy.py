# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# added
# from nudge.agents.blender_agent import BlenderActorCritic
from nudge.agents.logic_agent import NsfrActorCritic
from nudge.env_vectorized import VectorizedNudgeBaseEnv
import csv
import os
import sys
import time
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Callable

from rtpt import RTPT


import torch
import random
import numpy as np
import yaml
from rtpt import RTPT
from torch.optim import Optimizer, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from nudge.agents.logic_agent import LogicPPO
# from nudge.agents.neural_agent import NeuralPPO
# from nudge.agents.deictic_agent import DeicticPPO
from nudge.env import NudgeBaseEnv
from nudge.utils import make_deterministic, save_hyperparams
from nudge.utils import exp_decay, get_action_stats
from nudge.utils import print_program

# Log in to your W&B account
import wandb
OUT_PATH = Path("out/")
IN_PATH = Path("in/")

torch.set_num_threads(6)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "NUDGE"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Seaquest-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 128 #128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # added
    env_name: str = "seaquest"
    """the name of the environment"""
    algorithm: str = "blender"
    """the algorithm used in the agent"""
    blender_mode: str = "logic"
    """the mode for the blend"""
    actor_mode: str = "hybrid"
    """the mode for the agent"""
    rules: str = "default"
    """the ruleset used in the agent"""
    save_steps: int = 500000
    """the number of steps to save models"""
    pretrained: bool = False
    """to use pretrained neural agent"""
    joint_training: bool = False
    """jointly train neural actor and logic actor and blender"""
    learning_rate: float = 2.5e-5
    """the learning rate of the optimizer (neural)"""
    logic_learning_rate: float = 2.5e-4
    """the learning rate of the optimizer (logic)"""


def main():
        
    args = tyro.cli(Args)
    rtpt = RTPT(name_initials='HS', experiment_name='NUDGE', max_iterations=args.total_timesteps)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    model_description = "nudge"
    learning_description = f"lr_{args.learning_rate}_numenvs_{args.num_envs}_steps_{args.num_steps}"
    run_name = f"{args.env_id}_{model_description}_{learning_description}_{args.seed}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name = run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=args.num_envs, mode=args.algorithm, seed=args.seed)#$, **env_kwargs)

    agent = NsfrActorCritic(envs, args.rules, device)
    # if args.pretrained:
    #     agent.visual_neural_actor.load_state_dict(torch.load("models/neural_ppo_agent_Seaquest-v4.pth"))
    #     agent.to(device)
    #     print("Pretrained neural agent loaded!!!")
    # agent._print()
    if args.track:
        wandb.watch(agent)
        
    rtpt.start()
    # parameters = list(agent.logic_actor.parameters()) + list(agent.blender.parameters())
    # parameters = list(agent.visual_neural_actor.parameters()) + list(agent.logic_actor.parameters()) + list(agent.blender.parameters())
    # optimizer = optim.Adam(parameters, lr=args.learning_rate, eps=1e-5)
    # parameters = agent.parameters()
    # optimizer = optim.Adam(parameters, lr=args.logic_learning_rate)
    optimizer = optim.Adam(
        [
            {"params": agent.actor.parameters(), "lr": args.logic_learning_rate},
            {"params": agent.critic.parameters(), "lr": args.learning_rate},
        ],
        eps = 1e-5
    )
    
    
    # for param in agent.actor.parameters():
    #     print(param)


    # ALGO Logic: Storage setup
    observation_space = (4, 84, 84)
    # logic_observation_space = (84, 51, 4)
    logic_observation_space = (43, 4)
    # logic_observation_space = (84, 43, 4)
    action_space = ()
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space).to(device)
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    save_step_bar = args.save_steps
    start_time = time.time()
    next_logic_obs, next_obs = envs.reset()#(seed=seed)
    # 1 env 
    next_logic_obs = next_logic_obs.to(device)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # # (1, 4, 84, 84)
    # for i in range(4):
    #     image = wandb.Image(next_obs_array[0][0], caption=f"State at global_step={global_step}_{i}")
        # wandb.log({"state_image": image})
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            rtpt.step()
            global_step += args.num_envs
            obs[step] = next_obs
            # print(logic_obs.shape)
            # print(next_logic_obs.shape)
            logic_obs[step] = next_logic_obs
            dones[step] = next_done
            

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # next_obs: (1, 4, 84, 84)
                # next_logic_obs: (1, 84, 51, 4)
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_logic_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (next_logic_obs, next_obs), reward, terminations, truncations, infos  = envs.step(action.cpu().numpy())
            terminations = np.array(terminations)
            truncations = np.array(truncations)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_logic_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_logic_obs).to(device), torch.Tensor(next_done).to(device)


            # Plot image
            # next_obs_array = next_obs.detach().cpu().numpy()
            # max_rgb = np.max(next_obs_array)
            # # (1, 4, 84, 84)
            # for i in range(4):
            #     image = wandb.Image(next_obs_array[0][i], caption=f"State at global_step={global_step}_{i}")
            #     wandb.log({"state_image": image})
        
            for info_ in infos:
                if "final_info" in info_: # or next_done.any():
                    info = info_['final_info']
                    # final_info = info['final_info']
                    if "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
              
            # Save the model      
            if global_step > save_step_bar:
                experiment_dir = OUT_PATH / "runs" / run_name # / now.strftime("%y-%m-%d-%H-%M")
                checkpoint_dir = experiment_dir / "checkpoints"
                image_dir = experiment_dir / "images"
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(image_dir, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"step_{save_step_bar}.pth"
                agent.save(checkpoint_path, checkpoint_dir, [], [], [])
                print("\nSaved model at:", checkpoint_path)
                
                # increase the updated bar
                save_step_bar += args.save_steps
                
                # save hyper params
                # save_hyperparams(signature=signature(main),
                #      local_scope=locals(),
                #      save_path=experiment_dir / "config.yaml",
                #      print_summary=True)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_logic_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + observation_space)
        b_logic_obs = logic_obs.reshape((-1,) + logic_observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print(b_obs[mb_inds])
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_logic_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # named_params = dict(agent.actor.named_parameters())
                # for name, param in agent.actor.named_parameters():
                #     print(name, param.grad)
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # agent._print()
        print_program(agent)
        

    envs.close()
    writer.close()




if __name__ == "__main__":
    main()