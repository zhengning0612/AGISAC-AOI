"""TD3 agent implementation."""

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .models import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient agent."""

    def __init__(self, state_dim, action_dim, max_action, config: Config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.max_action = max_action
        self.config = config
        self.total_it = 0

    def select_action(self, state, noise=0.0):
        state_t = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state_t).cpu().data.numpy().flatten()

        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
            action[0] = np.clip(action[0], 0, 2 * np.pi)
            action[1] = np.clip(action[1], 0, self.max_action[1])
            action[2] = np.clip(action[2], 0.01, self.max_action[2])
            action[3] = np.clip(action[3], 0.01, self.max_action[3])

        return action

    def train(self, replay_buffer: ReplayBuffer):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(self.config.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.config.noise_std).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = self.actor_target(next_state) + noise
            next_action[:, 0] = next_action[:, 0].clamp(0, 2 * np.pi)
            next_action[:, 1] = next_action[:, 1].clamp(0, self.max_action[1])
            next_action[:, 2] = next_action[:, 2].clamp(0.01, self.max_action[2])
            next_action[:, 3] = next_action[:, 3].clamp(0.01, self.max_action[3])

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.config.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.config.policy_delay == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

    def save(self, path):
        """Save actor and critic state dictionaries."""

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "total_it": self.total_it,
            },
            path,
        )

    def load(self, path, map_location=None):
        """Load state dictionaries saved by :meth:`save`."""

        payload = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.total_it = int(payload.get("total_it", 0))

