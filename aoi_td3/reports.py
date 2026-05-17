"""Console and text-report helpers."""

from __future__ import print_function

from pathlib import Path

import torch

from .config import config_items
from .utils import current_device_name


def project_banner():
    return "\n".join(
        [
            "AOI TD3 AG-ISAC Experiment",
            "=========================",
            "This run trains a TD3 policy for UAV sensing and communication.",
        ]
    )


def format_startup_message(cfg, service_time):
    lines = [
        "开始训练TD3-TP算法...",
        "服务时间: {}s".format(service_time),
        "任务数量: {}".format(cfg.K),
        "检测概率阈值: {}".format(cfg.gamma_min),
        "SINR阈值: {}dB".format(cfg.tau_th),
    ]
    return "\n".join(lines)


def format_config_table(cfg):
    lines = ["Config values:"]
    for name, value in config_items(cfg):
        lines.append("  {:<20} {}".format(name, value))
    return "\n".join(lines)


def format_device_report():
    lines = [
        "Runtime device:",
        "  selected: {}".format(current_device_name()),
        "  torch: {}".format(torch.__version__),
        "  cuda available: {}".format(torch.cuda.is_available()),
    ]
    if torch.cuda.is_available():
        lines.append("  cuda device count: {}".format(torch.cuda.device_count()))
    return "\n".join(lines)


def write_text_report(path, sections):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(str(section) for section in sections)
    path.write_text(text, encoding="utf-8")
    return path


def format_task_report(env):
    lines = ["Tasks:"]
    for row in env.task_table():
        lines.append("  T{task}: ({x:.1f}, {y:.1f})".format(**row))
    return "\n".join(lines)


def format_evaluation_report(result):
    return "\n".join(
        [
            "Evaluation:",
            "  total reward: {:.2f}".format(result["total_reward"]),
            "  average AoI: {:.2f}".format(result["avg_aoi"]),
            "  total tasks: {}".format(result["total_tasks"]),
            "  total rounds: {}".format(result["total_rounds"]),
        ]
    )

