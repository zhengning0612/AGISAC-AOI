"""AOI TD3 project package.

The original experiment lived in a single ``main.py`` file.  This package keeps
the same training logic but separates the code into focused modules:

- configuration
- AG-ISAC environment dynamics
- TD3 neural networks
- replay buffer
- TD3 agent update logic
- training loop
- evaluation
- plotting

The public imports below are intentionally small and familiar.  Scripts can
still import the central objects from ``aoi_td3`` directly, while deeper modules
remain available for project-style maintenance.
"""

from .config import Config, config
from .environment import AGISACEnv
from .models import Actor, Critic
from .replay_buffer import ReplayBuffer
from .agent import TD3Agent
from .training import train_td3
from .evaluation import evaluate_agent
from .visualization import plot_training_curves, plot_trajectories

__all__ = [
    "Config",
    "config",
    "AGISACEnv",
    "Actor",
    "Critic",
    "ReplayBuffer",
    "TD3Agent",
    "train_td3",
    "evaluate_agent",
    "plot_training_curves",
    "plot_trajectories",
]

