"""Configuration for the AoI-aware AG-ISAC TD3 experiment.

The numeric values in this file are copied from the original merged script.
They are grouped by system component so that the rest of the project can import
a single ``config`` object without carrying large blocks of constants around.

No algorithmic behavior is implemented here.  The class only stores values and
pre-computes constants that the original code derived once at construction time.
"""

import numpy as np


class Config:
    """Container for all experiment constants.

    The class deliberately keeps mutable attributes instead of using a frozen
    dataclass because the original script used an ordinary class instance.  This
    makes it easy to override fields from experiments or notebooks without
    changing the rest of the code.
    """

    def __init__(self):
        # -------------------- Area --------------------
        self.S_min_x = 0
        self.S_max_x = 2500
        self.S_min_y = 0
        self.S_max_y = 2500

        # -------------------- UAV --------------------
        self.z_U = 120.0
        self.z_A = 100.0
        self.v_max = 40.0
        self.v_min_A = 20.0
        self.v_max_A = 30.0
        self.t_f = 5.0
        self.t_A_max = 5.0

        # -------------------- Base station --------------------
        self.x_B = 1500.0
        self.y_B = 1500.0
        self.H_B = 20.0

        # -------------------- Sensing --------------------
        self.K = 10
        self.T_s = 0.5
        self.R_s = 1e6
        self.gamma_min = 0.9
        self.P_F = 1e-6

        # -------------------- Communication --------------------
        self.tau_th = 1.0
        self.B = 1e6

        # -------------------- Power --------------------
        self.P_max = 1.0
        self.N_s = 10 ** ((-96) / 10) * 1e-3

        # -------------------- Radar gains --------------------
        self.G_t_dB = 35
        self.G_r_dB = 35
        self.G_j_dB = 21
        self.G_rj_dB = 21

        # -------------------- Carrier and radar constants --------------------
        self.f_c = 3e9
        self.c0 = 3e8
        self.sigma_rcs = 1.0
        self.chi = 0.2
        self.G_p = 1024

        # -------------------- Air-to-ground channel --------------------
        self.a_los = 12.0
        self.b_los = 0.135
        self.eta_LOS = 1.0
        self.eta_NLOS = 20.0

        # -------------------- Jammer power --------------------
        self.p_s_A = 0.01
        self.p_c_A = 0.01

        # -------------------- TD3 hyperparameters --------------------
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.gamma = 0.95
        self.tau = 0.05
        self.buffer_size = int(1e6)
        self.batch_size = 256
        self.noise_std = 0.36
        self.noise_decay = 0.999
        self.noise_clip = 0.5
        self.policy_delay = 2

        # -------------------- Reward shaping --------------------
        self.w_dis = 1.0
        self.w_exc = 20.0
        self.P_fl = 10.0
        self.P_r = 100.0

        # -------------------- Main objective weights --------------------
        self.lambda_aoi = 1.0
        self.lambda_gain = 1.0
        self.lambda_shap = 0.01

        # -------------------- Derived constants --------------------
        self.G_t_lin = 10 ** (self.G_t_dB / 10)
        self.G_r_lin = 10 ** (self.G_r_dB / 10)
        self.G_j_lin = 10 ** (self.G_j_dB / 10)
        self.G_rj_lin = 10 ** (self.G_rj_dB / 10)

        self.lambda_c = self.c0 / self.f_c
        self.K0 = (4.0 * np.pi * self.f_c / self.c0) ** 2


config = Config()


def config_items(cfg):
    """Yield public scalar-like configuration values.

    This helper is used by reporting code and intentionally avoids importing any
    training dependency.  It does not mutate the configuration object.
    """

    for name in sorted(dir(cfg)):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        if isinstance(value, (int, float, str, bool, list, tuple)):
            yield name, value


def format_config(cfg):
    """Return a readable block of configuration values."""

    lines = ["Configuration:"]
    for name, value in config_items(cfg):
        lines.append("  {}: {}".format(name, value))
    return "\n".join(lines)

