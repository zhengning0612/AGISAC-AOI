"""AG-ISAC environment used by the TD3 agent.

This file contains the environment logic from the original single-file script.
The method bodies are kept algorithmically equivalent: task locations, attacker
movement, sensing probability, communication SINR, AoI state construction, and
reward shaping all follow the same formulas.
"""

import random

import numpy as np
from scipy.stats import norm

from .config import Config


class AGISACEnv:
    """Age-of-Information aware AG-ISAC simulation environment."""

    def __init__(self, config: Config, service_time=600):
        self.config = config
        self.T = float(service_time)

        self.tasks = self._generate_tasks()

        self.q_U = np.array([600.0, 2000.0], dtype=np.float32)
        self.q_A0 = np.array([2400.0, 240.0], dtype=np.float32)

        self.A_trajectory = []
        self.A_waypoints = []

        self.current_time = 0.0
        self.step_count = 0
        self.stage = "fly_sense"
        self.beta = 0.0
        self.sense_start_time = 0.0

        self.task_generation_time = np.zeros(self.config.K, dtype=np.float32)
        self.task_last_update_time = np.zeros(self.config.K, dtype=np.float32)
        self.AoI = self.current_time - self.task_last_update_time

        self.task_updates = np.zeros(self.config.K, dtype=int)
        self.round_done_mask = np.zeros(self.config.K, dtype=np.float32)

        self.current_round = 0
        self.current_task = 0

        self.U_trajectory = [self.q_U.copy()]
        self.U_timestamps = [0.0]

        self.total_tasks_completed = 0
        self.total_rounds_completed = 0

        self.reset()

        self._dbg_gamma = 0.0
        self._dbg_Pr_echo = 0.0
        self._dbg_Pr_jam = 0.0
        self._dbg_d_UK = 0.0
        self._dbg_d_UA = 0.0

    def _generate_tasks(self):
        tasks = [
            [750, 2000],
            [270, 2100],
            [1500, 2250],
            [2200, 2100],
            [460, 1380],
            [2250, 1500],
            [350, 1000],
            [2250, 900],
            [1000, 400],
            [1800, 500],
        ]
        return np.array(tasks, dtype=np.float32)

    def _to_3d(self, q2d: np.ndarray, z: float):
        return np.array([float(q2d[0]), float(q2d[1]), float(z)], dtype=np.float64)

    def _distance_3d(self, p3: np.ndarray, q3: np.ndarray) -> float:
        return float(np.linalg.norm(p3 - q3))

    def _elevation_angle_deg(self, tx3: np.ndarray, rx3: np.ndarray) -> float:
        dx = tx3[0] - rx3[0]
        dy = tx3[1] - rx3[1]
        dz = tx3[2] - rx3[2]
        d_h = np.sqrt(dx * dx + dy * dy) + 1e-9
        return float(np.degrees(np.arctan2(abs(dz), d_h)))

    def _p_los(self, theta_deg: float) -> float:
        a = self.config.a_los
        b = self.config.b_los
        return float(1.0 / (1.0 + a * np.exp(-b * (theta_deg - a))))

    def _g_avg_paper(self, tx3: np.ndarray, rx3: np.ndarray) -> float:
        d = max(self._distance_3d(tx3, rx3), 1.0)
        theta = self._elevation_angle_deg(tx3, rx3)
        p_los = self._p_los(theta)
        p_nlos = 1.0 - p_los
        loss = p_los * self.config.eta_LOS + p_nlos * self.config.eta_NLOS
        loss = max(loss, 1e-12)
        g = (1.0 / self.config.K0) * (1.0 / (d ** 2)) * (1.0 / loss)
        return float(max(g, 0.0))

    def _generate_A_trajectory(self):
        self.A_trajectory = []
        self.A_waypoints = []

        current_pos = self.q_A0.copy().astype(np.float32)
        current_time = 0.0

        self.A_trajectory.append({"pos": current_pos.copy(), "time": current_time})
        self.A_waypoints.append({"pos": current_pos.copy(), "time": current_time})

        while current_time < self.T:
            next_pos = np.array(
                [
                    np.random.uniform(self.config.S_min_x, self.config.S_max_x),
                    np.random.uniform(self.config.S_min_y, self.config.S_max_y),
                ],
                dtype=np.float32,
            )

            distance = float(np.linalg.norm(next_pos - current_pos))
            v_A = float(np.random.uniform(self.config.v_min_A, self.config.v_max_A))
            fly_time = distance / max(v_A, 1e-6)

            num_points = max(int(fly_time / self.config.t_f), 1)
            for i in range(num_points):
                t = (i + 1) / num_points
                pos = current_pos + t * (next_pos - current_pos)
                current_time += self.config.t_f
                if current_time > self.T:
                    break
                self.A_trajectory.append({"pos": pos.copy(), "time": current_time})

            if current_time >= self.T:
                break

            hover_time = float(np.random.uniform(0, self.config.t_A_max))
            current_time += hover_time
            if current_time > self.T:
                break

            self.A_waypoints.append({"pos": next_pos.copy(), "time": current_time})
            current_pos = next_pos.copy()

    def _get_A_position(self, time_now: float):
        if not self.A_trajectory:
            return self.q_A0.copy()
        for i, point in enumerate(self.A_trajectory):
            if point["time"] >= time_now:
                if i == 0:
                    return point["pos"].copy()
                prev = self.A_trajectory[i - 1]
                t = (time_now - prev["time"]) / (point["time"] - prev["time"] + 1e-9)
                pos = prev["pos"] + t * (point["pos"] - prev["pos"])
                return pos.copy()
        return self.A_trajectory[-1]["pos"].copy()

    def _Q(self, x: float) -> float:
        return float(norm.sf(x))

    def _Qinv(self, p: float) -> float:
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        return float(norm.isf(p))

    def _calculate_detection_probability(self, task_idx: int, p_s: float) -> float:
        qU3 = self._to_3d(self.q_U, self.config.z_U)
        qA3 = self._to_3d(self.q_A, self.config.z_A)
        qk3 = self._to_3d(self.tasks[task_idx], 0.0)

        d_UK = max(self._distance_3d(qU3, qk3), 1.0)
        d_UA = max(self._distance_3d(qU3, qA3), 1.0)

        Pr_echo = (
            float(p_s)
            * self.config.G_t_lin
            * self.config.G_r_lin
            * self.config.sigma_rcs
            / (self.config.K0 * (d_UK ** 4) * 4 * np.pi)
        )

        Pr_jam = (
            self.config.p_s_A
            * self.config.G_j_lin
            * self.config.G_rj_lin
            / (self.config.K0 * (d_UA ** 2))
        ) * self.config.chi

        denom = max(self.config.N_s + Pr_jam, 1e-18)
        eta = self._Qinv(self.config.P_F) * np.sqrt(denom)

        x = (eta - np.sqrt(max(self.config.G_p * Pr_echo, 0.0))) / np.sqrt(denom)
        gamma = self._Q(float(x))

        self._dbg_gamma = gamma
        self._dbg_Pr_echo = Pr_echo
        self._dbg_Pr_jam = Pr_jam
        self._dbg_d_UK = d_UK
        self._dbg_d_UA = d_UA

        return float(np.clip(gamma, 0.0, 1.0))

    def _calculate_SINR(self, p_c: float) -> float:
        qU3 = self._to_3d(self.q_U, self.config.z_U)
        qA3 = self._to_3d(self.q_A, self.config.z_A)
        qB3 = np.array([self.config.x_B, self.config.y_B, self.config.H_B], dtype=np.float64)

        g_UB = self._g_avg_paper(qU3, qB3)
        g_AB = self._g_avg_paper(qA3, qB3)

        denom = max(g_AB * self.config.p_c_A + self.config.N_s, 1e-18)
        sinr = g_UB * float(p_c) / denom
        return float(max(sinr, 0.0))

    def _calculate_data_rate(self, sinr: float) -> float:
        return float(self.config.B * np.log2(1.0 + max(sinr, 0.0)))

    def reset(self):
        self.current_time = 0.0
        self.step_count = 0

        self.q_U = np.array([600.0, 2000.0], dtype=np.float32)

        self._generate_A_trajectory()
        self.q_A = self._get_A_position(0.0).astype(np.float32)

        self.task_generation_time = np.zeros(self.config.K, dtype=np.float32)
        self.task_last_update_time = np.zeros(self.config.K, dtype=np.float32)
        self.AoI = self.current_time - self.task_last_update_time

        self.task_updates = np.zeros(self.config.K, dtype=int)
        self.round_done_mask = np.zeros(self.config.K, dtype=np.float32)

        self.current_round = 0
        self.current_task = 0

        self.stage = "fly_sense"
        self.beta = 0.0
        self.sense_start_time = 0.0

        self.U_trajectory = [self.q_U.copy()]
        self.U_timestamps = [0.0]

        self.total_tasks_completed = 0
        self.total_rounds_completed = 0

        return self._get_state()

    def _get_state(self):
        AoI_norm = np.tanh(self.AoI / 100.0).astype(np.float32)

        a_k = self.task_updates.astype(np.float32)
        a_k_norm = np.tanh(a_k / 5.0).astype(np.float32)

        b_k = self.round_done_mask.astype(np.float32)

        task_pos = (self.tasks[self.current_task] / 2500.0).astype(np.float32)
        q_U_norm = (self.q_U / 2500.0).astype(np.float32)
        q_A_norm = (self.q_A / 2500.0).astype(np.float32)

        beta_norm = float(np.tanh(self.beta / 100.0))

        d_to_task = float(np.linalg.norm(self.q_U - self.tasks[self.current_task]) / 2500.0)
        d_to_BS = float(
            np.linalg.norm(self.q_U - np.array([self.config.x_B, self.config.y_B], dtype=np.float32)) / 2500.0
        )
        d_to_A = float(np.linalg.norm(self.q_U - self.q_A) / 2500.0)

        state = np.concatenate(
            [
                AoI_norm,
                a_k_norm,
                b_k,
                task_pos,
                q_U_norm,
                q_A_norm,
                np.array([beta_norm, d_to_task, d_to_BS, d_to_A], dtype=np.float32),
                np.array([self.current_task / self.config.K, self.current_round / 10.0], dtype=np.float32),
            ],
            axis=0,
        )

        return state.astype(np.float32)

    def step(self, action):
        time_before = float(self.current_time)
        AoI_before = self.AoI.copy()

        theta, v, p_s, p_c = action
        theta = float(np.clip(theta, 0.0, 2.0 * np.pi))
        v = float(np.clip(v, 0.0, self.config.v_max))
        p_s = float(np.clip(p_s, 0.01, self.config.P_max))
        p_c = float(np.clip(p_c, 0.01, self.config.P_max))

        new_x = float(self.q_U[0] + v * np.cos(theta) * self.config.t_f)
        new_y = float(self.q_U[1] + v * np.sin(theta) * self.config.t_f)

        boundary_penalty = 0.0
        if (
            (new_x < self.config.S_min_x)
            or (new_x > self.config.S_max_x)
            or (new_y < self.config.S_min_y)
            or (new_y > self.config.S_max_y)
        ):
            boundary_penalty = float(self.config.P_fl)
        else:
            self.q_U[0] = new_x
            self.q_U[1] = new_y

        self.current_time += self.config.t_f
        self.q_A = self._get_A_position(self.current_time).astype(np.float32)

        self.U_trajectory.append(self.q_U.copy())
        self.U_timestamps.append(float(self.current_time))

        d_U_A = float(np.linalg.norm(self.q_U - self.q_A))

        reward = 0.0
        b_exc = 0.0
        task_completed = False
        round_completed = False

        if self.stage == "fly_sense":
            gamma = self._calculate_detection_probability(self.current_task, p_s)
            if gamma >= self.config.gamma_min:
                self.sense_start_time = float(self.current_time)
                self.current_time += self.config.T_s

                self.task_generation_time[self.current_task] = float(self.current_time)

                b_exc = 1.0
                print(f"感知成功! Task {self.current_task}, Gamma: {gamma:.3f}")

                self.stage = "fly_comm"

        elif self.stage == "fly_comm":
            sinr = self._calculate_SINR(p_c)
            sinr_dB = 10.0 * np.log10(sinr) if sinr > 0 else -np.inf

            if sinr_dB >= self.config.tau_th:
                rate = self._calculate_data_rate(sinr)
                data_size = self.config.T_s * self.config.R_s
                T_c = data_size / max(rate, 1e-12)
                T_c = float(min(T_c, 20.0))

                k = self.current_task

                old_aoi = float(self.current_time - self.task_last_update_time[k])

                self.current_time += T_c

                self.task_last_update_time[k] = float(self.task_generation_time[k])

                new_aoi = float(self.current_time - self.task_last_update_time[k])
                self.AoI[k] = new_aoi

                aoi_gain = old_aoi - new_aoi
                reward += aoi_gain

                print(f" 通信成功! Task {k}, AoI: {old_aoi:.1f}→{new_aoi:.1f}, SINR: {sinr_dB:.1f}dB")

                self.task_updates[k] += 1
                self.round_done_mask[k] = 1.0

                task_completed = True
                self.total_tasks_completed += 1

                self.current_task += 1
                if self.current_task >= self.config.K:
                    self.current_task = 0
                    self.current_round += 1
                    round_completed = True
                    reward += float(self.config.P_r)
                    self.total_rounds_completed += 1
                    self.round_done_mask[:] = 0.0

                self.stage = "fly_sense"

            else:
                pass

        self.AoI = self.current_time - self.task_last_update_time

        delta_beta = self.config.w_dis * d_U_A + self.config.w_exc * b_exc - boundary_penalty
        self.beta += float(delta_beta)

        r_shap = 2.0 / (1.0 + np.exp(-self.beta / (self.config.w_exc * self.config.K))) - 1.0

        delta_t = float(max(0.0, self.current_time - time_before))
        mean_before = float(np.mean(AoI_before))
        mean_after = float(np.mean(self.AoI))

        reward += float(r_shap)

        done = bool(self.current_time >= self.T)
        self.step_count += 1

        info = {
            "AoI": self.AoI.copy(),
            "round": self.current_round,
            "task_completed": task_completed,
            "total_tasks": self.total_tasks_completed,
            "total_rounds": self.total_rounds_completed,
            "stage": self.stage,
        }

        return self._get_state(), float(reward), done, info

    def debug_snapshot(self):
        """Return the latest radio-debug values without changing state."""

        return {
            "gamma": self._dbg_gamma,
            "Pr_echo": self._dbg_Pr_echo,
            "Pr_jam": self._dbg_Pr_jam,
            "d_UK": self._dbg_d_UK,
            "d_UA": self._dbg_d_UA,
        }

    def task_table(self):
        """Return task coordinates as ordinary dictionaries for reports."""

        rows = []
        for index, task in enumerate(self.tasks):
            rows.append({"task": index, "x": float(task[0]), "y": float(task[1])})
        return rows

