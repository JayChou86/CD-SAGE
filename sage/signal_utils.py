"""
信号处理工具模块
=================
- 基于 FFT 的分数延迟 (fractional delay)
- 复基带导频信号生成
- 多径接收信号仿真
- CRB (Cramér-Rao Bound) 理论下界计算
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Tuple


def fractional_delay_shift(
    s_base: np.ndarray,
    delay: float,
    N_total: int,
) -> np.ndarray:
    """
    严格基于 FFT 的分数延迟平移。

    使用频域相移实现任意小数时延，避免时域插值带来的频谱泄露。

    Parameters
    ----------
    s_base : 1D complex ndarray
        发射基带信号 (长度为 N_orig)。
    delay : float
        时延 (采样点，支持小数)。
    N_total : int
        输出信号总长度。

    Returns
    -------
    s_shifted : 1D complex ndarray, shape (N_total,)
        延迟后的信号，在 floor(delay) 位置开始放置。
    """
    N_orig = len(s_base)
    S_freq = fft(s_base)

    # 对称频率向量 (MATLAB 风格: [0:ceil(N/2)-1, -floor(N/2):-1])
    k = np.concatenate([
        np.arange(0, int(np.ceil(N_orig / 2))),
        np.arange(-int(np.floor(N_orig / 2)), 0),
    ])

    # 分数部分相移
    frac_part = delay - np.floor(delay)
    phase_shift = np.exp(-1j * 2 * np.pi * k * frac_part / N_orig)
    s_temp = ifft(S_freq * phase_shift)

    # 放置到输出数组
    s_shifted = np.zeros(N_total, dtype=complex)
    start_idx = int(np.floor(delay))
    end_idx = start_idx + N_orig
    if start_idx >= 0 and end_idx <= N_total:
        s_shifted[start_idx:end_idx] = s_temp

    return s_shifted


def generate_tx_signal(
    N_seq: int = 511,
    seed: int = 42,
) -> np.ndarray:
    """
    生成复基带 QPSK 导频/M序列发射信号。

    Parameters
    ----------
    N_seq : int
        序列长度。
    seed : int
        随机种子 (与 MATLAB rng(42) 保持一致的复现需要相同生成方式)。

    Returns
    -------
    s_tx : 1D complex ndarray, shape (N_seq,)
        能量归一化的发射信号。
    """
    rng = np.random.RandomState(seed)
    s_tx = (np.sign(rng.randn(N_seq)) + 1j * np.sign(rng.randn(N_seq)))
    s_tx = s_tx / np.linalg.norm(s_tx)
    return s_tx


def generate_rx_signal(
    s_tx: np.ndarray,
    true_delays: np.ndarray,
    true_amps: np.ndarray,
    N_rx: int,
    noise_std: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成含噪多径接收信号。

    Parameters
    ----------
    s_tx : 1D complex ndarray
        发射信号。
    true_delays : 1D ndarray
        真实时延列表 (采样点，可为小数)。
    true_amps : 1D complex ndarray
        真实复数幅度列表。
    N_rx : int
        接收信号总长度。
    noise_std : float
        复噪声标准差 (实部虚部各 noise_std)。
    seed : int
        噪声随机种子。

    Returns
    -------
    y_rx : 1D complex ndarray
        含噪接收信号。
    y_clean : 1D complex ndarray
        无噪接收信号。
    """
    rng = np.random.RandomState(seed)
    y_clean = np.zeros(N_rx, dtype=complex)

    for delay, amp in zip(true_delays, true_amps):
        s_frac = fractional_delay_shift(s_tx, delay, N_rx)
        y_clean += amp * s_frac

    noise = noise_std * (rng.randn(N_rx) + 1j * rng.randn(N_rx))
    y_rx = y_clean + noise

    return y_rx, y_clean


def compute_crb(
    s_tx: np.ndarray,
    true_amps: np.ndarray,
    noise_var: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算多径时延/幅度/相位的 Cramér-Rao Bound 理论下界。

    基于离散信号的一阶导数能量。

    Parameters
    ----------
    s_tx : 1D complex ndarray
        发射信号 (已能量归一化)。
    true_amps : 1D complex ndarray
        真实复数幅度。
    noise_var : float
        复噪声方差 N0。

    Returns
    -------
    crb_delay_std : ndarray
        各路径时延 CRB 标准差。
    crb_mag_std : ndarray
        各路径幅度 CRB 标准差。
    crb_phase_std : ndarray
        各路径相位 CRB 标准差。
    """
    N_seq = len(s_tx)
    S_freq = fft(s_tx)

    # 对称频率向量
    k = np.concatenate([
        np.arange(0, int(np.ceil(N_seq / 2))),
        np.arange(-int(np.floor(N_seq / 2)), 0),
    ])
    omega = 2 * np.pi * k / N_seq

    # 一阶导数信号
    s_prime = ifft((1j * omega) * S_freq)
    energy_s_prime = np.sum(np.abs(s_prime) ** 2)
    energy_s = np.sum(np.abs(s_tx) ** 2)  # 归一化后应为 1

    # CRB 方差
    crb_var_delay = noise_var / (2 * np.abs(true_amps) ** 2 * energy_s_prime)
    crb_var_mag = noise_var / (2 * energy_s) * np.ones_like(true_amps, dtype=float)
    crb_var_phase = noise_var / (2 * np.abs(true_amps) ** 2 * energy_s)

    crb_delay_std = np.sqrt(crb_var_delay)
    crb_mag_std = np.sqrt(crb_var_mag)
    crb_phase_std = np.sqrt(crb_var_phase)

    return crb_delay_std, crb_mag_std, crb_phase_std


def build_dictionary(
    s_tx: np.ndarray,
    tau_grid: np.ndarray,
    N_rx: int,
) -> np.ndarray:
    """
    构建细网格字典矩阵 (供 MF, OMP, SBL 使用)。

    Parameters
    ----------
    s_tx : 1D complex ndarray
        发射信号。
    tau_grid : 1D ndarray
        时延候选网格。
    N_rx : int
        接收信号长度。

    Returns
    -------
    A_dict : 2D complex ndarray, shape (N_rx, N_grid)
        字典矩阵，每列为一个时延对应的接收波形。
    """
    N_grid = len(tau_grid)
    A_dict = np.zeros((N_rx, N_grid), dtype=complex)
    for g, tau in enumerate(tau_grid):
        A_dict[:, g] = fractional_delay_shift(s_tx, tau, N_rx)
    return A_dict
