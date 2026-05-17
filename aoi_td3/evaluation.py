"""Evaluation helpers for trained TD3 agents."""

import numpy as np

from .config import config
from .environment import AGISACEnv


def evaluate_agent(agent, service_time=600, cfg=None):
    """Run one deterministic evaluation episode.

    The loop is the same test loop that originally lived under
    ``if __name__ == "__main__"``.
    """

    if cfg is None:
        cfg = config

    test_env = AGISACEnv(cfg, service_time=service_time)
    state = test_env.reset()
    done = False
    total_reward = 0.0
    info = None

    while not done:
        action = agent.select_action(state, noise=0.0)
        state, reward, done, info = test_env.step(action)
        total_reward += reward

    if info is None:
        info = {
            "AoI": test_env.AoI.copy(),
            "total_tasks": test_env.total_tasks_completed,
            "total_rounds": test_env.total_rounds_completed,
        }

    result = {
        "env": test_env,
        "total_reward": float(total_reward),
        "avg_aoi": float(np.mean(info["AoI"])),
        "total_tasks": int(info["total_tasks"]),
        "total_rounds": int(info["total_rounds"]),
        "info": info,
    }
    return result


def print_evaluation_result(result):
    """Print the evaluation block used by the runner."""

    print(f"测试完成:")
    print(f"  总奖励: {result['total_reward']:.2f}")
    print(f"  平均AoI: {result['avg_aoi']:.2f}")
    print(f"  完成任务数: {result['total_tasks']}")
    print(f"  完成回合数: {result['total_rounds']}")

