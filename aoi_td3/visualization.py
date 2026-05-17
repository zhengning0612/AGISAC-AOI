"""Plotting functions for trajectories and training curves."""

import numpy as np
import matplotlib.pyplot as plt

from .config import config


def plot_trajectories(env, title="UAV Trajectories"):
    plt.figure(figsize=(12, 10))

    for i, task in enumerate(env.tasks):
        circle = plt.Circle((task[0], task[1]), 50, color="gray", alpha=0.3)
        plt.gca().add_patch(circle)
        plt.plot(task[0], task[1], "ko", markersize=8)
        plt.text(task[0], task[1] - 80, f"T{i + 1}", ha="center", fontsize=9)

    plt.plot(config.x_B, config.y_B, "b^", markersize=15, label="Base Station")

    if env.A_trajectory:
        A_positions = np.array([point["pos"] for point in env.A_trajectory])
        plt.plot(A_positions[:, 0], A_positions[:, 1], "b-", linewidth=2, alpha=0.6, label="UAV-A Trajectory")

        for i, wp in enumerate(env.A_waypoints[: min(5, len(env.A_waypoints))]):
            plt.plot(wp["pos"][0], wp["pos"][1], "c^", markersize=10)
            plt.text(wp["pos"][0], wp["pos"][1] + 50, f"{wp['time']:.0f}s", color="blue", fontsize=8, ha="center")

    U_positions = np.array(env.U_trajectory)
    plt.plot(U_positions[:, 0], U_positions[:, 1], "r-", linewidth=2, label="UAV-U Trajectory")

    step_interval = max(len(env.U_timestamps) // 10, 1)
    for i in range(0, len(env.U_timestamps), step_interval):
        if i < len(env.U_trajectory):
            plt.plot(env.U_trajectory[i][0], env.U_trajectory[i][1], "r^", markersize=8)
            plt.text(
                env.U_trajectory[i][0],
                env.U_trajectory[i][1] + 50,
                f"{env.U_timestamps[i]:.0f}s",
                color="red",
                fontsize=8,
                ha="center",
            )

    plt.xlim(config.S_min_x - 100, config.S_max_x + 100)
    plt.ylim(config.S_min_y - 100, config.S_max_y + 100)
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    return plt.gcf()


def plot_training_curves(episode_rewards, episode_AoIs, episode_tasks, episode_rounds):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    window = 50

    ax1.plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(episode_rewards)), smoothed, linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episode_AoIs, alpha=0.3)
    if len(episode_AoIs) >= window:
        smoothed = np.convolve(episode_AoIs, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(episode_AoIs)), smoothed, linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average AoI")
    ax2.set_title("Episode AoI")
    ax2.grid(True, alpha=0.3)

    ax3.plot(episode_tasks, alpha=0.3)
    if len(episode_tasks) >= window:
        smoothed = np.convolve(episode_tasks, np.ones(window) / window, mode="valid")
        ax3.plot(range(window - 1, len(episode_tasks)), smoothed, linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Tasks Completed")
    ax3.set_title("Episode Tasks Completed")
    ax3.grid(True, alpha=0.3)

    ax4.plot(episode_rounds, alpha=0.3)
    if len(episode_rounds) >= window:
        smoothed = np.convolve(episode_rounds, np.ones(window) / window, mode="valid")
        ax4.plot(range(window - 1, len(episode_rounds)), smoothed, linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Rounds Completed")
    ax4.set_title("Episode Rounds Completed")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(fig, path, dpi=300):
    """Save a matplotlib figure with the same defaults as the original script."""

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

