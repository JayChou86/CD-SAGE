"""
插值模块
========
在 SAGE M-step 的整数互相关峰值处进行亚采样插值，
突破整数采样网格的精度限制。

方法:
- parabolic_delta:  经典三点抛物线插值 (基于 |corr| 或 |corr|²)
- parabolic_interp:  完整插值流程 (含边界保护、分母保护)
- complex_interp:    对复数互相关分别插值实/虚部
"""

import numpy as np
from typing import Tuple


def parabolic_delta(y_m1: float, y_0: float, y_p1: float, eps: float = 1e-10) -> float:
    """
    三点抛物线插值，返回峰值偏移量 delta。

    假设三点 (x=-1, y_m1), (x=0, y_0), (x=1, y_p1) 构成的抛物线
    顶点位于 x = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1)。

    Parameters
    ----------
    y_m1 : float
        k_opt-1 处的值。
    y_0 : float
        k_opt 处的值。
    y_p1 : float
        k_opt+1 处的值。
    eps : float
        分母保护阈值。

    Returns
    -------
    delta : float
        峰值相对整数的偏移量，约束在 [-0.5, 0.5] 内。
    """
    denom = y_m1 - 2 * y_0 + y_p1
    if abs(denom) < eps:
        return 0.0
    delta = 0.5 * (y_m1 - y_p1) / denom
    # 边界约束：抛物线峰值不应超出相邻采样点之间
    return np.clip(delta, -0.5, 0.5)


def parabolic_interp(
    corr_vals: np.ndarray,
    k_opt: int,
    use_power: bool = True,
) -> float:
    """
    在互相关序列的峰值处执行抛物线插值，返回亚采样精度的时延。

    Parameters
    ----------
    corr_vals : 1D ndarray (real)
        互相关幅度序列 (或幅度平方序列)。
    k_opt : int
        整数峰值索引 (0-indexed)。
    use_power : bool
        True:  对 |corr|² 插值 (峰值更尖锐，推荐)。
        False: 对 |corr| 插值 (与原始 MATLAB 一致)。

    Returns
    -------
    tau_frac : float
        亚采样精度的时延 (k_opt + delta)。
    """
    search_len = len(corr_vals)

    if k_opt <= 0 or k_opt >= search_len - 1:
        return float(k_opt)

    y = corr_vals
    if use_power:
        # 对幅度平方插值 → 峰值更尖锐
        y_m1 = abs(y[k_opt - 1]) ** 2
        y_0 = abs(y[k_opt]) ** 2
        y_p1 = abs(y[k_opt + 1]) ** 2
    else:
        y_m1 = abs(y[k_opt - 1])
        y_0 = abs(y[k_opt])
        y_p1 = abs(y[k_opt + 1])

    delta = parabolic_delta(y_m1, y_0, y_p1)
    return float(k_opt) + delta


def complex_parabolic_interp(
    corr_complex: np.ndarray,
    k_opt: int,
) -> Tuple[float, float]:
    """
    对复数互相关分别插值实部和虚部，返回两种策略的时延。

    这比仅在幅度上插值保留了相位信息。

    Parameters
    ----------
    corr_complex : 1D complex ndarray
        复数互相关序列。
    k_opt : int
        整数峰值索引。

    Returns
    -------
    tau_real : float
        对实部插值得到的时延。
    tau_imag : float
        对虚部插值得到的时延。
    """
    search_len = len(corr_complex)
    if k_opt <= 0 or k_opt >= search_len - 1:
        return float(k_opt), float(k_opt)

    delta_real = parabolic_delta(
        corr_complex[k_opt - 1].real,
        corr_complex[k_opt].real,
        corr_complex[k_opt + 1].real,
    )
    delta_imag = parabolic_delta(
        corr_complex[k_opt - 1].imag,
        corr_complex[k_opt].imag,
        corr_complex[k_opt + 1].imag,
    )

    return float(k_opt) + delta_real, float(k_opt) + delta_imag


def local_refine_search(
    x_l: np.ndarray,
    s_base: np.ndarray,
    tau_coarse: float,
    N_rx: int,
    fractional_delay_shift_fn,
    step: float = 0.005,
    radius: float = 0.5,
) -> Tuple[float, complex]:
    """
    在插值得到的粗估计附近进行局部细搜索 (二阶段精化)。

    在 [tau_coarse - radius, tau_coarse + radius] 范围内以 step 为步长
    搜索使 |⟨s(τ), x_l⟩| 最大的时延和对应的复幅度。

    Parameters
    ----------
    x_l : 1D complex ndarray
        E-step 信号。
    s_base : 1D complex ndarray
        发射模板信号。
    tau_coarse : float
        插值得到的粗时延估计。
    N_rx : int
        接收信号长度。
    fractional_delay_shift_fn : callable
        分数延迟函数。
    step : float
        搜索步长 (采样点)。
    radius : float
        搜索半径 (采样点)。

    Returns
    -------
    tau_best : float
        精化后的时延。
    amp_best : complex
        精化后的复幅度。
    """
    Es = np.vdot(s_base, s_base).real
    best_obj = -np.inf
    tau_best = tau_coarse
    amp_best = 0.0

    tau_start = max(0, tau_coarse - radius)
    tau_end = tau_coarse + radius
    n_steps = max(2, int((tau_end - tau_start) / step))
    tau_candidates = np.linspace(tau_start, tau_end, n_steps)

    for tau_test in tau_candidates:
        s_test = fractional_delay_shift_fn(s_base, tau_test, N_rx)
        amp_test = np.vdot(s_test, x_l) / Es
        residual_energy = np.sum(np.abs(x_l - amp_test * s_test) ** 2)
        obj = -residual_energy  # 最大化 → 最小化残差能量

        if obj > best_obj:
            best_obj = obj
            tau_best = tau_test
            amp_best = amp_test

    return tau_best, amp_best
