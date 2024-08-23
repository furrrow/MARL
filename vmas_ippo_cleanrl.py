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
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from vmas import make_env
from vmas.simulator.utils import save_video
""" cleanrl PPO implementation applied to VMAS
heavily referencing:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

"""


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
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
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, n_hidden=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space[0].shape).prod(), n_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden, n_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.action_space[0].shape).prod(), n_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden, n_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(n_hidden, np.prod(envs.action_space[0].shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space[0].shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def main():
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

    agent_list = []
    optimizer_list = []
    buffer_list = []
    obs_list = []
    actions_list = []
    logprobs_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    for idx in range(args.n_agents):
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs_list.append(torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device))
        actions_list.append(torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device))
        logprobs_list.append(torch.zeros((args.num_steps, args.num_envs)).to(device))
        rewards_list.append(torch.zeros((args.num_steps, args.num_envs)).to(device))
        dones_list.append(torch.zeros((args.num_steps, args.num_envs)).to(device))
        values_list.append(torch.zeros((args.num_steps, args.num_envs)).to(device))

        agent_list.append(agent)
        optimizer_list.append(optimizer)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    a_next_obs = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(a_next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    frame_list = []  # For creating a gif
    total_reward = torch.zeros(envs.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        actions = [None] * envs.n_agents
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for optimizer in optimizer_list:
                optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            actions = [None] * envs.n_agents
            logprobs = [None] * envs.n_agents
            values = [None] * envs.n_agents
            for i in range(envs.num_agents):
                with torch.no_grad():
                    actions[i], logprobs[i], _, values[i] = agent_list[i].get_action_and_value(next_obs[i])

            # execute the game and log data.
            next_obs, rewards, terminations, infos = envs.step(actions)
            next_dones = dones
            for i in range(envs.num_agents):
                rewards_list[i][step] = torch.tensor(rewards[i]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

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

        # bootstrap value if not done
        with torch.no_grad():
            for i in range(envs.n_agents):
                next_value = agent_list[i].get_value(next_obs[i]).reshape(1, -1)
                advantages = torch.zeros_like(rewards[i]).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done[i]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[i][t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

        if args.save_model and iteration % 100:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            model_weights = [agent.state_dict() for agent in agent_list]
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


if __name__ == '__main__':
    main()
