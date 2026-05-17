"""Metric containers and post-training summaries.

These helpers do not change the training algorithm.  They make the project feel
less like a notebook dump by giving the run outputs named structures that can be
saved, inspected, and reused by downstream analysis scripts.
"""

from __future__ import print_function

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class EpisodeMetric:
    episode: int
    reward: float
    average_aoi: float
    tasks_completed: int
    rounds_completed: int


@dataclass
class MetricSummary:
    count: int
    reward_mean: float
    reward_std: float
    aoi_mean: float
    tasks_mean: float
    rounds_mean: float
    reward_max: float
    reward_min: float


class TrainingHistory:
    """Structured view of the four metric lists returned by training."""

    def __init__(self):
        self.records = []

    def append(self, metric):
        self.records.append(metric)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @classmethod
    def from_lists(cls, rewards, aois, tasks, rounds):
        history = cls()
        for idx, values in enumerate(zip(rewards, aois, tasks, rounds), start=1):
            reward, aoi, task_count, round_count = values
            history.append(
                EpisodeMetric(
                    episode=idx,
                    reward=float(reward),
                    average_aoi=float(aoi),
                    tasks_completed=int(task_count),
                    rounds_completed=int(round_count),
                )
            )
        return history

    def rewards(self):
        return [record.reward for record in self.records]

    def aois(self):
        return [record.average_aoi for record in self.records]

    def tasks(self):
        return [record.tasks_completed for record in self.records]

    def rounds(self):
        return [record.rounds_completed for record in self.records]

    def tail(self, count=10):
        return self.records[-count:]

    def summarize(self, window=None):
        records = self.records if window is None else self.records[-window:]
        if not records:
            return MetricSummary(
                count=0,
                reward_mean=0.0,
                reward_std=0.0,
                aoi_mean=0.0,
                tasks_mean=0.0,
                rounds_mean=0.0,
                reward_max=0.0,
                reward_min=0.0,
            )

        rewards = np.array([record.reward for record in records], dtype=np.float64)
        aois = np.array([record.average_aoi for record in records], dtype=np.float64)
        tasks = np.array([record.tasks_completed for record in records], dtype=np.float64)
        rounds = np.array([record.rounds_completed for record in records], dtype=np.float64)
        return MetricSummary(
            count=len(records),
            reward_mean=float(np.mean(rewards)),
            reward_std=float(np.std(rewards)),
            aoi_mean=float(np.mean(aois)),
            tasks_mean=float(np.mean(tasks)),
            rounds_mean=float(np.mean(rounds)),
            reward_max=float(np.max(rewards)),
            reward_min=float(np.min(rewards)),
        )

    def to_rows(self):
        return [asdict(record) for record in self.records]

    def save_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["episode", "reward", "average_aoi", "tasks_completed", "rounds_completed"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.to_rows():
                writer.writerow(row)


def summarize_lists(rewards, aois, tasks, rounds, window=None):
    history = TrainingHistory.from_lists(rewards, aois, tasks, rounds)
    return history.summarize(window=window)


def format_summary(summary):
    return "\n".join(
        [
            "Metric summary:",
            "  episodes: {}".format(summary.count),
            "  reward mean: {:.2f}".format(summary.reward_mean),
            "  reward std: {:.2f}".format(summary.reward_std),
            "  AoI mean: {:.2f}".format(summary.aoi_mean),
            "  tasks mean: {:.2f}".format(summary.tasks_mean),
            "  rounds mean: {:.2f}".format(summary.rounds_mean),
            "  reward min/max: {:.2f}/{:.2f}".format(summary.reward_min, summary.reward_max),
        ]
    )


def rolling_window(values, window):
    values = [float(value) for value in values]
    if window <= 0:
        return []
    if len(values) < window:
        return []
    windows = []
    for start in range(0, len(values) - window + 1):
        windows.append(values[start : start + window])
    return windows


def rolling_mean(values, window):
    windows = rolling_window(values, window)
    return [float(np.mean(window_values)) for window_values in windows]

