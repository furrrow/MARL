import os
import random
import time
import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.mpe import simple_v3, simple_speaker_listener_v4
from vmas.simulator.utils import save_video
from wandb import agent

""" cleanrl PPO implementation applied to VMAS
heavily referencing:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

"""


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 5
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    render_video: bool = False
    """render video"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    scenario_name: str = "simple_speaker_listener"
    """the scenario_name of the pettingzoo scenario"""
    n_agents: int = 2
    """number of agents"""
    num_envs: int = 1
    """number of environments"""
    env_max_steps: int = 25
    """environment steps before done"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.9
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    n_hidden: int = 64
    """number of hidden cells for Agent's NN"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QCritic(nn.Module):
    def __init__(self, envs, n_hidden=64):
        super().__init__()
        self.total_obs_size = sum([envs.observation_space(name).shape[0] for name in envs.agents])
        self.total_act_size = sum([envs.action_space(name).shape[0] for name in envs.agents])
        self.fc1 = layer_init(nn.Linear(np.array(self.total_obs_size).prod(), n_hidden))
        self.fc2 = layer_init(nn.Linear(n_hidden, n_hidden))
        self.fc3 = layer_init(nn.Linear(n_hidden, 1))

    def get_value(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.tanh((self.fc2(x)))
        x = torch.nn.functional.tanh((self.fc3(x)))
        return x

class Actor(nn.Module):
    def __init__(self, envs, agent_name, n_hidden=64):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(envs.observation_space(agent_name).shape).prod(), n_hidden))
        self.fc2 = layer_init(nn.Linear(n_hidden, n_hidden))
        self.fc_mu = layer_init(nn.Linear(n_hidden, np.prod(envs.action_space(agent_name).shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space(agent_name).shape)))

    def get_action(self, x, action=None):
        x = self.fc1(x)
        x = torch.nn.functional.tanh((self.fc2(x)))
        action_mean = torch.nn.functional.tanh((self.fc_mu(x)))  # [num_envs, 2]
        action_logstd = self.actor_logstd.expand_as(action_mean)  # [num_envs, 2]
        action_std = torch.exp(action_logstd)  # [num_envs, 2]
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

# class Agent(nn.Module):
#     def __init__(self, envs, agent_name, n_hidden=64):
#         super().__init__()
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(self.total_obs_size + self.total_act_size, n_hidden)),
#             nn.Tanh(),
#             layer_init(nn.Linear(n_hidden, n_hidden)),
#             nn.Tanh(),
#             layer_init(nn.Linear(n_hidden, 1), std=1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(np.array(envs.observation_space(agent_name).shape).prod(), n_hidden)),
#             nn.Tanh(),
#             layer_init(nn.Linear(n_hidden, n_hidden)),
#             nn.Tanh(),
#             layer_init(nn.Linear(n_hidden, np.prod(envs.action_space(agent_name).shape)), std=0.01),
#         )
#         self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space(agent_name).shape)))
#
#     def get_value(self, x):
#         return self.critic(x)
#
#     def get_action_and_value(self, x, action=None):
#         action_mean = self.actor_mean(x) # [num_envs, 2]
#         action_logstd = self.actor_logstd.expand_as(action_mean)  # [num_envs, 2]
#         action_std = torch.exp(action_logstd)  # [num_envs, 2]
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == '__main__':
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    current_time = datetime.datetime.now()
    run_name = f"{args.exp_name}__{args.seed}__{current_time.strftime('%m%d%y_%H%M')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{args.scenario_name}/{run_name}")
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
    print(f"using {device}")
    assert (args.num_envs == 1), "no parallel envs in pettingzoo"
    render_mode = "human" if args.render_video else "rgb_array"
    envs = simple_speaker_listener_v4.parallel_env(
        render_mode=render_mode, continuous_actions=True,
        max_cycles=args.env_max_steps  # max_cycle default 25
    )
    envs.reset()
    agent_names = envs.agents.copy()
    assert (args.n_agents == envs.num_agents), "arg n_agents mismatch env n_agents"
    actors = {}
    critics = {}
    actor_optimizers = {}
    critic_optimizers = {}
    buffers = {}
    total_reward = {}
    next_done = {}
    for agent_name in agent_names:
        actors[agent_name] = Actor(envs, agent_name, args.n_hidden).to(device)
        critics[agent_name] = QCritic(envs, args.n_hidden).to(device)
        actor_optimizers[agent_name] = optim.Adam(actors[agent_name].parameters(), lr=args.learning_rate, eps=1e-5)
        critic_optimizers[agent_name] = optim.Adam(critics[agent_name].parameters(), lr=args.learning_rate, eps=1e-5)
        buffer = TensorDict({
            "obs": torch.zeros((args.num_steps,) + envs.observation_space(agent_name).shape).to(device),
            "actions": torch.zeros((args.num_steps,) + envs.action_space(agent_name).shape).to(device),
            "logprobs": torch.zeros(args.num_steps).to(device),
            "rewards": torch.zeros(args.num_steps).to(device),
            "dones": torch.zeros(args.num_steps).to(device),
            "values": torch.zeros(args.num_steps).to(device),
        })
        buffers[agent_name] = buffer

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    frame_list = []  # For creating a gif

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for agent_name in agent_names:
                actor_optimizers[agent_name].param_groups[0]["lr"] = lrnow
                critic_optimizers[agent_name].param_groups[0]["lr"] = lrnow
        step = 0
        while step < args.num_steps:
            obs, infos = envs.reset(seed=args.seed)
            for agent_name in agent_names:
                next_done[agent_name] = 0
                total_reward[agent_name] = 0
            while envs.agents:
                actions = {}
                obs_all = [torch.Tensor(obs[agent_name]).to(device) for agent_name in agent_names]
                for i, agent_name in enumerate(agent_names):
                    obs_agent = obs_all[i]
                    buffers[agent_name]['obs'][step] = obs_agent
                    buffers[agent_name]['dones'][step] = next_done[agent_name]

                    # ALGO LOGIC: action logic
                    with torch.no_grad():
                        action, logprob, _ = actors[agent_name].get_action(obs_agent.unsqueeze(0))
                        value = critics[agent_name].get_value(torch.concat(obs_all).unsqueeze(0))
                        buffers[agent_name]["values"][step] = value.flatten()
                        low_limit = torch.tensor(envs.action_space(agent_name).low).to(device)
                        high_limit = torch.tensor(envs.action_space(agent_name).high).to(device)
                        action[0] = action[0].clip(low_limit, high_limit)
                    buffers[agent_name]["actions"][step], actions[agent_name] = action, action[0].cpu().numpy()
                    buffers[agent_name]["logprobs"][step] = logprob

                # execute the game and log data.
                next_obs, rewards, dones, truncated, infos = envs.step(actions)
                for agent_name in agent_names:
                    done_or_truncated = np.logical_or(dones[agent_name], truncated[agent_name]) * 1
                    next_done[agent_name] = torch.Tensor([done_or_truncated]).to(device)
                    buffers[agent_name]["rewards"][step] = torch.tensor(rewards[agent_name]).to(device)
                    total_reward[agent_name] += rewards[agent_name]
                global_step += 1
                step += 1
                if step >= args.num_steps:
                    break

                # CRUCIAL step easy to overlook
                obs = next_obs

                if args.render_video:
                    frame_list.append(
                        envs.render(
                            mode="rgb_array",
                            agent_index_focus=None,
                            visualize_when_rgb=True,
                        )
                    )
                frame_list = []

            # pettingzoo handles episode ending differently
            for agent_name in agent_names:
                writer.add_scalar(f"charts/episodic_return_{agent_name}", total_reward[agent_name], global_step)
                print(f"global_step {global_step} {agent_name} episodic_returns {total_reward[agent_name]:.3f}")

        b_inds = np.arange(args.batch_size)
        np.random.shuffle(b_inds)
        # bootstrap value if not done
        with torch.no_grad():
            next_obs_all = [torch.Tensor(next_obs[agent_name]).to(device) for agent_name in agent_names]
        for i_a, agent_name in enumerate(agent_names):
            with torch.no_grad():
                next_value = critics[agent_name].get_value(torch.concat(next_obs_all))
                advantages = torch.zeros_like(buffers[agent_name]["rewards"]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done[agent_name]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - buffers[agent_name]["dones"][t + 1]
                        nextvalues = buffers[agent_name]["values"][t + 1]
                    delta = buffers[agent_name]["rewards"][t] + args.gamma * nextvalues * nextnonterminal - \
                            buffers[agent_name]["values"][t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + buffers[agent_name]["values"]

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy = actors[agent_name].get_action(
                        buffers[agent_name]["obs"][mb_inds], buffers[agent_name]["actions"][mb_inds])
                    logratio = newlogprob - buffers[agent_name]["logprobs"][mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    actor_optimizers[agent_name].zero_grad()
                    pg_loss.backward()
                    nn.utils.clip_grad_norm_(actors[agent_name].parameters(), args.max_grad_norm)
                    actor_optimizers[agent_name].step()

                    # Value loss
                    _, newlogprob, entropy = actors[agent_name].get_action(
                        buffers[agent_name]["obs"][mb_inds], buffers[agent_name]["actions"][mb_inds])
                    all_buffer_obs = [buffers[agent_name]["obs"][mb_inds] for agent_name in agent_names]
                    newvalue = critics[agent_name].get_value(torch.concat(all_buffer_obs, dim=-1)).squeeze(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                        v_clipped = buffers[agent_name]["values"][mb_inds] + torch.clamp(
                            newvalue - buffers[agent_name]["values"][mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    loss = - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    critic_optimizers[agent_name].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(critics[agent_name].parameters(), args.max_grad_norm)
                    critic_optimizers[agent_name].step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = buffers[agent_name]["values"].cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(f"charts/{agent_name}_learning_rate", critic_optimizers[agent_name].param_groups[0]["lr"],
                              global_step)
            writer.add_scalar(f"losses/{agent_name}_value_loss", v_loss.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_entropy", entropy_loss.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_approx_kl", approx_kl.item(), global_step)
            writer.add_scalar(f"losses/{agent_name}_clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar(f"losses/{agent_name}_explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(f"charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model and iteration % 100:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            model_list = actors + critics
            model_weights = [model.state_dict() for model in model_list]
            torch.save(model_weights, model_path)
            # TODO: maybe not working yet...
            print(f"model saved to {model_path}")
