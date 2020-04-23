from torch.optim import Adam

from ddpg_utils import OUNoise, ReplayMemory
from model import Actor, Critic

import torch.nn.functional as F

import numpy as np
import torch


class CompetetiveAgent:
    def __init__(self, state_size, action_size, num_agents, args, device="cpu"):
        self.device = device
        self.args = args
        self.action_size = action_size
        self.state_size = state_size

        self.agents = []

        for _ in range(num_agents):
            self.agents.append(Agent(state_size, action_size, args, device))

        self.replay_buffer = ReplayMemory(args["memory_size"], args["batch_size"])

    def act(self, states, add_noise=True):
        actions = []
        for agent, state in zip(self.agents, states):
            actions.append(agent.act(state, add_noise))
        return actions

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()

    def step(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.append(states, actions, rewards, next_states, dones)

        losses = []
        for agent in self.agents:
            batch = self.replay_buffer.sample(device=self.device)
            curr_loss = agent.learn(batch)
            losses.append(curr_loss)

        if losses[0] is None:
            return None

        return np.mean(losses, axis=0)

    def save(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_target, "{}th_player_agent_target.pth".format(i + 1))
            torch.save(agent.critic_target, "{}th_player_agent_critic.pth".format(i + 1))


class Agent:
    def __init__(self, state_size, action_size, args, device="cpu"):
        self.action_size = action_size
        self.state_size = state_size
        self.device = device
        self.discount_factor = args["discount_factor"]
        self.tau = args["tau"]

        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=args["lr_actor"])

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=args["lr_critic"],
                                     weight_decay=args["weight_decay"])

        self.hard_update_actor(self.actor_local)
        self.hard_update_critic(self.critic_local)

        self.noise = OUNoise(action_size)

    def learn(self, batch):
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.discount_factor * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update_critic(self.critic_local)
        self.soft_update_actor(self.actor_local)

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def hard_update_actor(self, model):
        for target_param, param in zip(self.actor_target.parameters(), model.parameters()):
            target_param.data.copy_(param.data)

    def hard_update_critic(self, model):
        for target_param, param in zip(self.critic_target.parameters(), model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update_actor(self, model):
        for target_param, param in zip(self.actor_target.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def soft_update_critic(self, model):
        for target_param, param in zip(self.critic_target.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
