"""General utilities shared by the project modules."""

from __future__ import print_function

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy, and PyTorch exactly as the original script did."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def ensure_dir(path):
    """Create a directory if it does not exist and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path, base=None):
    """Resolve a path without changing current working directory."""

    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is None:
        base = Path.cwd()
    return (Path(base) / path).resolve()


def moving_average(values, window):
    """Compute a trailing moving average as a Python list."""

    values = [float(v) for v in values]
    if window <= 1 or not values:
        return values
    output = []
    running = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        output.append(running / len(queue))
    return output


def count_parameters(model, trainable_only=False):
    """Count model parameters for reports."""

    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return int(sum(p.numel() for p in params))


def current_device_name():
    """Return a friendly device description."""

    if torch.cuda.is_available():
        return "cuda: {}".format(torch.cuda.get_device_name(0))
    return "cpu"


def format_seconds(seconds):
    """Format elapsed time for console output."""

    seconds = float(seconds)
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    minutes = int(seconds // 60)
    rest = seconds - 60 * minutes
    return "{}m {:.1f}s".format(minutes, rest)


@dataclass
class Stopwatch:
    """Small elapsed-time helper."""

    start_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.perf_counter()

    def reset(self):
        self.start_time = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.start_time


@dataclass
class OutputPaths:
    """Output file locations used by the default runner."""

    output_dir: Path
    trajectory_png: Path
    training_png: Path

    @classmethod
    def build(cls, output_dir="."):
        output_dir = ensure_dir(output_dir)
        return cls(
            output_dir=output_dir,
            trajectory_png=output_dir / "uav_trajectories_fixed.png",
            training_png=output_dir / "training_curves_fixed.png",
        )


def env_flag(name, default=False):
    """Read a boolean environment variable."""

    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def safe_mean(values, default=0.0):
    """Mean helper that accepts empty iterables."""

    values = list(values)
    if not values:
        return float(default)
    return float(np.mean(values))

