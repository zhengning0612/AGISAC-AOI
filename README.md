# AGISAC-AOI

《AoI Optimization for UAV-Assisted AG-ISAC》的仿真代码。项目基于空地一体化感知通信场景，构建无人机用户、对抗无人机、地面任务点与基站组成的动态环境，并使用 TD3 强化学习算法优化无人机飞行方向、速度、感知功率和通信功率，以降低系统信息时效（Age of Information, AoI）并提升任务完成效率。

## 文件结构

- `main.py`: 主函数入口。
- `aoi_td3/config.py`: 参数配置。
- `aoi_td3/environment.py`: 空地一体化感知通信仿真环境。
- `aoi_td3/models.py`: TD3 的 Actor 与 Critic 网络结构。
- `aoi_td3/replay_buffer.py`: 经验回放池。
- `aoi_td3/agent.py`: TD3 智能体及策略更新逻辑。
- `aoi_td3/training.py`: 模型训练流程。
- `aoi_td3/evaluation.py`: 训练后模型评估。
- `aoi_td3/visualization.py`: 无人机轨迹与训练曲线可视化。
- `aoi_td3/metrics.py`: 训练指标统计与历史记录。
- `aoi_td3/reports.py`: 运行报告与控制台输出。
- `aoi_td3/utils.py`: 随机种子、路径、计时等通用工具。

## 运行方式

安装依赖后，在项目目录下运行：

```bash
python main.py
```

## 输出结果

训练和测试结束后，程序会生成：

- `uav_trajectories_fixed.png`: 无人机轨迹图。
- `training_curves_fixed.png`: 训练奖励、AoI、任务完成数和回合完成数曲线。

## 项目声明

本项目的作者及单位：

```text
项目名称 (Project Name): AGISAC-AOI
项目作者 (Author): Ning Zheng, Zhen Chen
作者单位 (Affiliation): 暨南大学网络空间安全学院(College of cyber Security,Jinan University)
```
