"""Training loop for the TD3 AoI experiment."""

import numpy as np

from .agent import TD3Agent
from .config import config
from .environment import AGISACEnv
from .replay_buffer import ReplayBuffer


def train_td3(service_time=600, episodes=3000):
    """Train TD3 with the same loop as the original script.

    Returns:
        tuple: ``agent, env, episode_rewards, episode_AoIs, episode_tasks,
        episode_rounds``.
    """

    env = AGISACEnv(config, service_time)

    state_dim = len(env.reset())
    action_dim = 4
    max_action = np.array([2 * np.pi, config.v_max, config.P_max, config.P_max], dtype=np.float32)

    agent = TD3Agent(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(config.buffer_size)

    episode_rewards = []
    episode_AoIs = []
    episode_tasks = []
    episode_rounds = []
    noise = config.noise_std

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        if episode % 10 == 0:
            print(f"\n{'=' * 60}")
            print(f"Episode {episode + 1} 开始")
            print(f"{'=' * 60}")

        while not done:
            action = agent.select_action(state, noise)
            next_state, reward, done, info = env.step(action)

            replay_buffer.add((state, action, reward, next_state, float(done)))

            state = next_state
            episode_reward += reward

            if replay_buffer.size() > 1500:
                agent.train(replay_buffer)

        noise *= config.noise_decay
        noise = max(noise, 0.1)

        episode_rewards.append(episode_reward)
        episode_AoIs.append(float(np.mean(info["AoI"])))
        episode_tasks.append(info["total_tasks"])
        episode_rounds.append(info["total_rounds"])

        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(episode_rewards[-100:]))
            avg_AoI = float(np.mean(episode_AoIs[-100:]))
            avg_tasks = float(np.mean(episode_tasks[-100:]))
            avg_rounds = float(np.mean(episode_rounds[-100:]))
            print(f"\n{'=' * 60}")
            print(f"Episode {episode + 1} 统计:")
            print(f"  平均Reward: {avg_reward:.2f}")
            print(f"  平均AoI: {avg_AoI:.2f}")
            print(f"  平均完成任务数: {avg_tasks:.1f}")
            print(f"  平均完成回合数: {avg_rounds:.2f}")
            print(f"{'=' * 60}\n")

    return agent, env, episode_rewards, episode_AoIs, episode_tasks, episode_rounds


def train_td3_with_config(cfg, service_time=600, episodes=3000):
    """Variant useful for experiments with a custom config object.

    The default public training function keeps the exact original signature.
    This helper exists for project-level experimentation and mirrors the same
    steps while accepting a supplied configuration instance.
    """

    env = AGISACEnv(cfg, service_time)
    state_dim = len(env.reset())
    action_dim = 4
    max_action = np.array([2 * np.pi, cfg.v_max, cfg.P_max, cfg.P_max], dtype=np.float32)
    agent = TD3Agent(state_dim, action_dim, max_action, cfg)
    replay_buffer = ReplayBuffer(cfg.buffer_size)

    episode_rewards = []
    episode_AoIs = []
    episode_tasks = []
    episode_rounds = []
    noise = cfg.noise_std

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, noise)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            episode_reward += reward
            if replay_buffer.size() > 1500:
                agent.train(replay_buffer)

        noise *= cfg.noise_decay
        noise = max(noise, 0.1)
        episode_rewards.append(episode_reward)
        episode_AoIs.append(float(np.mean(info["AoI"])))
        episode_tasks.append(info["total_tasks"])
        episode_rounds.append(info["total_rounds"])

    return agent, env, episode_rewards, episode_AoIs, episode_tasks, episode_rounds

