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
from vmas.scenarios.balance import HeuristicPolicy as BalanceHeuristic
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
    scenario_name: str = "balance"
    """the scenario_name of the VMAS scenario"""
    n_agents: int = 4
    """number of agents"""
    num_envs: int = 12
    """number of environments"""
    env_max_steps: int = 200
    """environment steps before done"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""


def main():
    args = tyro.cli(Args)
    current_time = datetime.datetime.now()
    run_name = f"{args.scenario_name}__{args.exp_name}__{args.seed}__{current_time.strftime('%m%d%y_%H%M')}"
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
    assert envs.continuous_actions, "only continuous action space is supported"
    # CAVEAT: observation and action space must be same for all agents!!!

    policy = BalanceHeuristic(continuous_action=True)

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
            actions[i] = policy.compute_action(obs[i], u_range=envs.agents[i].u_range)
        # execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

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

        if loop % 100 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # separate iterator from global_step due to multiple envs
        loop += 1

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
