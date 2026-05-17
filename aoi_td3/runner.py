"""Command-line runner for the AOI TD3 project."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .config import config
from .evaluation import evaluate_agent, print_evaluation_result
from .reports import format_startup_message
from .training import train_td3
from .utils import OutputPaths, set_seed
from .visualization import plot_training_curves, plot_trajectories, save_figure


def build_parser():
    parser = argparse.ArgumentParser(description="Train TD3 for the AoI AG-ISAC UAV environment.")
    parser.add_argument("--service-time", type=float, default=600)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--no-show", action="store_true")
    return parser


def run(args=None):
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    set_seed(parsed.seed)
    paths = OutputPaths.build(parsed.output_dir)

    print(format_startup_message(config, parsed.service_time))
    print()

    agent, env, episode_rewards, episode_AoIs, episode_tasks, episode_rounds = train_td3(
        service_time=parsed.service_time,
        episodes=parsed.episodes,
    )

    print("\n测试训练好的模型...")
    result = evaluate_agent(agent, service_time=parsed.service_time, cfg=config)
    print_evaluation_result(result)

    test_env = result["env"]

    print("\n生成轨迹图...")
    fig1 = plot_trajectories(
        test_env,
        "UAV Trajectories (T={}s, Rounds={})".format(parsed.service_time, result["total_rounds"]),
    )
    save_figure(fig1, paths.trajectory_png)
    print("轨迹图已保存")

    print("生成训练曲线...")
    fig2 = plot_training_curves(episode_rewards, episode_AoIs, episode_tasks, episode_rounds)
    save_figure(fig2, paths.training_png)
    print("训练曲线已保存")

    if not parsed.no_show:
        plt.show()

    print("\n完成！")
    return {
        "agent": agent,
        "env": env,
        "test_env": test_env,
        "episode_rewards": episode_rewards,
        "episode_AoIs": episode_AoIs,
        "episode_tasks": episode_tasks,
        "episode_rounds": episode_rounds,
        "evaluation": result,
        "paths": paths,
    }


def main():
    run()

