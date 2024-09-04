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
from vmas import make_env
from vmas.simulator.utils import save_video


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
    scenario_name: str = "simple"
    """the scenario_name of the VMAS scenario"""
    n_agents: int = 1
    """number of agents"""
    num_envs: int = 12
    """number of environments"""
    env_max_steps: int = 100
    """environment steps before done"""
    total_timesteps: int = 2_500_000  # 2_500_000
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
    def __init__(self, envs, agent_idx, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.observation_space[agent_idx].shape).prod()
                             + np.prod(envs.action_space[agent_idx].shape), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, envs, agent_idx, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(np.array(envs.observation_space[agent_idx].shape).prod(), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc_mu = nn.Linear(n_hidden, np.prod(envs.action_space[agent_idx].shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((envs.action_space[agent_idx].high[0]
                                          - envs.action_space[agent_idx].low[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((envs.action_space[agent_idx].high[0]
                                         + envs.action_space[agent_idx].low[0]) / 2.0, dtype=torch.float32)
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
    assert not (args.capture_video and not args.render_video), "To save the video you have to render it"

    envs = make_env(
        scenario=args.scenario_name,
        num_envs=args.num_envs,
        device=device,
        continuous_actions=True,
        max_steps=args.env_max_steps,
        seed=args.seed,
        # Scenario specific
        n_agents=args.n_agents,
    )
    assert (args.n_agents == envs.n_agents), "arg n_agents mismatch env n_agents"
    assert envs.continuous_actions, "only continuous action space is supported"
    # CAVEAT: observation and action space must be same for all agents!!!

    critic_list = []
    critic_target_list = []
    critic_optimizer_list = []
    actor_list = []
    actor_target_list = []
    actor_optimizer_list = []
    buffer_list = []
    for idx in range(args.n_agents):
        qf1 = QNetwork(envs, idx, args.n_hidden).to(device)
        qf1_target = QNetwork(envs, idx, args.n_hidden).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
        actor = Actor(envs, idx, args.n_hidden).to(device)
        target_actor = Actor(envs, idx, args.n_hidden).to(device)
        target_actor.load_state_dict(actor.state_dict())
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(max_size=args.buffer_size, device=device),
            batch_size=args.batch_size,
        )

        critic_list.append(qf1)
        critic_target_list.append(qf1_target)
        critic_optimizer_list.append(q_optimizer)
        actor_list.append(actor)
        actor_target_list.append(target_actor)
        actor_optimizer_list.append(actor_optimizer)
        buffer_list.append(rb)

    start_time = time.time()

    frame_list = []  # For creating a gif
    init_time = time.time()
    global_step = 0
    loop = 0
    obs = envs.reset(seed=args.seed)
    total_reward = torch.zeros(envs.num_envs).to(device)
    # policy = RandomPolicy(continuous_action=True)
    while global_step < args.total_timesteps:
        actions = [None] * envs.n_agents
        for i in range(envs.n_agents):
            if global_step < args.learning_starts:
                actions[i] = np.array([envs.action_space[i].sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():
                    actions[i] = actor_list[i](torch.Tensor(obs[i]).to(device))
                    actions[i] += torch.normal(0, actor_list[i].action_scale * args.exploration_noise)
                    actions[i] = actions[i].cpu().numpy().clip(envs.action_space[i].low, envs.action_space[i].high)

        # execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # save to replay buffer
        real_next_obs = next_obs.copy()
        for i in range(envs.n_agents):
            data = TensorDict({
                "observations": obs[i],
                "next_observations": real_next_obs[i],
                "actions": actions[i],
                "rewards": rewards[i],
                "dones": dones,
                # "infos": infos[i],  # to save on RAM comment me out
            }, batch_size=[envs.num_envs]).to(device)
            buffer_list[i].extend(data)

        # record rewards for plotting purposes
        global_reward = torch.stack(rewards, dim=1).mean(dim=1)
        total_reward += global_reward
        for idx, item in enumerate(dones):
            if bool(item) is True:
                writer.add_scalar("charts/episodic_return", total_reward[idx], global_step)
                print(f"global_step {global_step} done detected at idx {idx} "
                      f"rewards {rewards[0][idx]:.3f} episodic_returns {total_reward[idx]:.3f}")
                next_obs = envs.reset_at(index=idx)
                total_reward[idx] = 0
            global_step += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for i in range(args.n_agents):
                data = buffer_list[i].sample(args.batch_size)
                with (torch.no_grad()):
                    next_state_actions = actor_target_list[i](data["next_observations"])
                    qf1_next_target = critic_target_list[i](data["next_observations"], next_state_actions)
                    next_q_value = data["rewards"].flatten() + (
                                1 - 1 * data["dones"].flatten()) * args.gamma * qf1_next_target.view(-1)

                qf1_a_values = critic_list[i](data["observations"], data["actions"]).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                critic_optimizer_list[i].zero_grad()
                qf1_loss.backward()
                critic_optimizer_list[i].step()

                if loop % args.policy_frequency == 0:
                    actor_loss = -critic_list[i](data["observations"], actor_list[i](data["observations"])).mean()
                    actor_optimizer_list[i].zero_grad()
                    actor_loss.backward()
                    actor_optimizer_list[i].step()

                    # update the target network
                    for param, target_param in zip(actor_list[i].parameters(), actor_target_list[i].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(critic_list[i].parameters(), critic_target_list[i].parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if loop % 100 == 0:
                    writer.add_scalar(f"losses/qf{i}_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar(f"losses/qf{i}_loss", qf1_loss.item(), global_step)
                    writer.add_scalar(f"losses/actor{i}_loss", actor_loss.item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    print(f"global step {global_step} qf1_loss {qf1_loss.item():.3f} actor_loss {actor_loss.item():.3f} "
                          f"buffer size {len(buffer_list[i].storage)}")
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            # separate iterator from global_step due to multiple envs
            loop += 1

        if args.save_model and loop % 100:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            model_list = actor_list + critic_list
            model_weights = [model.state_dict() for model in model_list]
            torch.save(model_weights, model_path)
            # TODO: maybe not working yet...
            print(f"model saved to {model_path}")

        if args.render_video:
            frame_list.append(
                envs.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )
        frame_list = []

    total_time = time.time() - init_time
    if args.render_video and args.capture_video:
        save_video(args.scenario_name, frame_list, 1 / envs.scenario.world.dt)

    print(
        f"It took: {total_time:.2f}s for {global_step} steps of {args.num_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward}"
    )
    pass


if __name__ == "__main__":
    main()
