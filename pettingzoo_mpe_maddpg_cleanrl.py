# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.mpe import simple_speaker_listener_v4
from utils import show_render
from vmas.simulator.utils import save_video
import pygame


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
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
    """the scenario_name of the VMAS scenario"""
    n_agents: int = 2
    """number of agents"""
    num_envs: int = 1
    """number of environments"""
    env_max_steps: int = 25
    """environment steps before done"""
    total_timesteps: int = 1_000_000  # 2_500_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.01
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 100
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    n_hidden: int = 64
    """number of units per hidden layer"""


class QNetwork(nn.Module):
    def __init__(self, n_total_states, n_total_actions, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(n_total_states
                             + n_total_actions, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, envs, agent_name, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.observation_space(agent_name).shape).prod(), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, np.prod(envs.action_space(agent_name).shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space(agent_name).high[0]
                                          - envs.action_space(agent_name).low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space(agent_name).high[0]
                                         + envs.action_space(agent_name).low[0]) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def main():
    args = tyro.cli(Args)
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
        render_mode=render_mode, continuous_actions=True  # max_cycle default 25
    )
    obs, infos = envs.reset(seed=args.seed)
    assert (args.n_agents == envs.num_agents), "arg n_agents mismatch env n_agents"
    # CAVEAT: observation and action space must be same for all agents!!!
    agent_names = envs.agents.copy()
    critics = {}
    critic_targets = {}
    critic_optimizers = {}
    actors = {}
    actor_targets = {}
    actor_optimizers = {}
    n_total_actions = sum([envs.action_space(agent_name).shape[0] for agent_name in agent_names])
    n_total_states = sum([envs.observation_space(agent_name).shape[0] for agent_name in agent_names])
    total_reward = {}
    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device=device),
        batch_size=args.batch_size,
    )
    for agent_name in envs.agents:
        qf1 = QNetwork(n_total_states, n_total_actions, args.n_hidden).to(device)
        qf1_target = QNetwork(n_total_states, n_total_actions, args.n_hidden).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor = Actor(envs, agent_name, args.n_hidden).to(device)
        actor_target = Actor(envs, agent_name, args.n_hidden).to(device)
        actor_target.load_state_dict(actor.state_dict())
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

        critics[agent_name] = qf1
        critic_targets[agent_name] = qf1_target
        critic_optimizers[agent_name] = q_optimizer
        actors[agent_name] = actor
        actor_targets[agent_name] = actor_target
        actor_optimizers[agent_name] = actor_optimizer
        total_reward[agent_name] = 0

    start_time = time.time()

    frame_list = []  # For creating a gif
    init_time = time.time()
    global_step = 0
    episode = 0
    loop = 0
    # pygame.init()
    while global_step < args.total_timesteps:
        obs, infos = envs.reset()
        # pygame.event.get()
        while envs.agents:
            actions = {}
            for agent_name in envs.agents:
                if global_step < args.learning_starts:
                    actions[agent_name] = np.array(envs.action_space(agent_name).sample())
                else:
                    with torch.no_grad():
                        action = actors[agent_name](torch.Tensor(obs[agent_name]).to(device))
                        action += torch.normal(0, actors[agent_name].action_scale * args.exploration_noise)
                        actions[agent_name] = action.cpu().numpy().clip(
                            envs.action_space(agent_name).low, envs.action_space(agent_name).high)
            # execute the game and log data.
            next_obs, rewards, dones, truncated, infos = envs.step(actions)

            # save to replay buffer
            real_next_obs = next_obs.copy()
            data = TensorDict({
                "observations": TensorDict(obs).unsqueeze(0),
                "next_observations": TensorDict(real_next_obs).unsqueeze(0),
                "actions": TensorDict(actions).unsqueeze(0),
                "rewards": TensorDict(rewards).unsqueeze(0),
                "dones": TensorDict(dones).unsqueeze(0),
                # "infos": infos[agent_name],  # to save on RAM comment me out
            }, batch_size=[1]).to(device)
            rb.extend(data)
            # record rewards for plotting purposes
            for agent_name in agent_names:
                total_reward[agent_name] += rewards[agent_name]
            global_step += 1

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                sample = rb.sample(args.batch_size)
                # b_ means batch-size is first dim, and other dimensions are concatenated
                b_observations = torch.concat([sample["observations"][name] for name in agent_names], dim=1)
                b_next_observations = torch.concat([sample["next_observations"][name] for name in agent_names], dim=1)
                b_actions = torch.concat([sample["actions"][name] for name in agent_names], dim=1)

                next_q_values = {}
                with (torch.no_grad()):
                    next_state_actions = torch.concat([actor_targets[agent_name](sample["next_observations"][agent_name])
                                                       for agent_name in agent_names], dim=1)
                    # reshape into [batch_size, n_agents * dim]
                    for agent_name in agent_names:
                        qf1_next_target = critic_targets[agent_name](b_next_observations, next_state_actions)
                        next_q_values[agent_name] = sample["rewards"][agent_name] + (
                                1 - 1 * sample['dones'][agent_name].flatten()) * args.gamma * qf1_next_target.view(-1)
                for agent_name in agent_names:
                    qf1_a_values = critics[agent_name](b_observations, b_actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_values[agent_name].to(torch.float32))

                    # optimize the model
                    critic_optimizers[agent_name].zero_grad()
                    qf1_loss.backward()
                    critic_optimizers[agent_name].step()
                    if loop % 100 == 0:
                        writer.add_scalar(f"losses/q_{agent_name}_values", qf1_a_values.mean().item(), global_step)
                        writer.add_scalar(f"losses/q_{agent_name}_loss", qf1_loss.item(), global_step)

                if loop % args.policy_frequency == 0:
                    for agent_name in agent_names:
                        state_actions = torch.concat(
                            [actors[name](sample["observations"][name]) for name in agent_names], dim=1)
                        actor_loss = -critics[agent_name](b_observations, state_actions).mean()
                        actor_optimizers[agent_name].zero_grad()
                        actor_loss.backward()
                        actor_optimizers[agent_name].step()

                        # update the target network
                        for param, target_param in zip(actors[agent_name].parameters(), actor_targets[agent_name].parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(critics[agent_name].parameters(), critic_targets[agent_name].parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                        if loop % 100 == 0:
                            writer.add_scalar(f"losses/actor_{agent_name}_loss", actor_loss.item(), global_step)

                if loop % 100 == 0:
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # separate iterator from global_step due to multiple envs
                loop += 1

        # pettingzoo handles episode ending differently
        for agent_name in agent_names:
            writer.add_scalar(f"charts/episodic_return_{agent_name}", total_reward[agent_name], global_step)
            print(f"global_step {global_step} {agent_name} episodic_return {total_reward[agent_name]:.3f}")
            total_reward[agent_name] = 0
        episode += 1

        if args.save_model and loop+1 % 100:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            model_list = actors + critics
            model_weights = [model.state_dict() for model in model_list]
            torch.save(model_weights, model_path)
            # TODO: maybe not working yet...
            print(f"model saved to {model_path}")

        if args.capture_video and loop+1 % 100:
            frame = envs.render()
            frame_list.append(frame)
        frame_list = []

    total_time = time.time() - init_time
    if args.capture_video:
        save_video(args.scenario_name, frame_list, 1 / envs.scenario.world.dt)

    print(
        f"It took: {total_time:.2f}s for {global_step} steps of {args.num_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )
    pass


if __name__ == "__main__":
    main()
