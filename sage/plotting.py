"""
绘图模块
========
学术级图表:
- RMSE vs SNR 三子图 (时延/幅度/相位)
- 信道冲激响应 stem 图
- 残差信号对比图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合脚本批量出图
import matplotlib.pyplot as plt
from typing import List, Optional

# Sub-panel label utility — adds (a),(b),(c) to top-left of axes
def _add_sublabels(axes, fmt='(%s)', fontsize=16, fontweight='bold',
                   ox=-0.08, oy=1.02):
    for i, ax in enumerate(axes):
        ax.text(ox, oy, fmt % chr(ord('a')+i), transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, ha='left', va='bottom')


# 全局样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'mathtext.fontset': 'stix',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def plot_rmse_vs_snr(
    snr_dB_vec: np.ndarray,
    rmse_delay: np.ndarray,
    rmse_mag: np.ndarray,
    rmse_phase: np.ndarray,
    crb_delay: np.ndarray,
    crb_mag: np.ndarray,
    crb_phase: np.ndarray,
    algo_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    RMSE vs SNR 三子图 (时延 / 幅度 / 相位)。

    Parameters
    ----------
    snr_dB_vec : 1D ndarray, shape (num_SNR,)
    rmse_delay/mag/phase : 2D ndarray, shape (num_SNR, num_algos)
    crb_delay/mag/phase : 1D ndarray, shape (num_SNR,)
    algo_names : list of str
    save_path : str or None

    Returns
    -------
    fig : matplotlib Figure
    """
    colors = ['k', 'b', 'g', 'm', 'orange', 'c']
    markers = ['s', '^', '*', 'd', 'v', 'x']
    line_styles = ['-', '-', '--', '--', '-.', '-']
    num_algos = len(algo_names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _add_sublabels(axes)

    titles = [
        ('Delay RMSE vs CRB', 'Delay RMSE (ms)'),
        ('Magnitude RMSE vs CRB', 'Magnitude RMSE'),
        ('Phase RMSE vs CRB', 'Phase RMSE (rad)'),
    ]
    data = [
        (rmse_delay, crb_delay),
        (rmse_mag, crb_mag),
        (rmse_phase, crb_phase),
    ]

    for ax_idx, (ax, (title, ylabel), (rmse, crb)) in enumerate(
        zip(axes, titles, data)
    ):
        ax.semilogy(snr_dB_vec, crb, 'r-', linewidth=2, label='CRB')
        for alg in range(num_algos):
            ax.semilogy(
                snr_dB_vec, rmse[:, alg],
                color=colors[alg],
                marker=markers[alg],
                linestyle=line_styles[alg],
                linewidth=1.5,
                markersize=7,
                label=algo_names[alg],
            )
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        if ax_idx == 0:
            ax.set_ylim([1e-3, 200])

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_channel_stem(
    true_delays: np.ndarray,
    true_amps: np.ndarray,
    est_delays_list: List[np.ndarray],
    est_amps_list: List[np.ndarray],
    labels: List[str],
    xlim: tuple = (30, 220),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    多径信道冲激响应 stem 对比图（含放大子图）。

    主图显示全部时延范围，三个放大子图分别放大每条多径附近。
    """
    colors = ['r', 'k', 'b', 'g', 'm', 'c']
    markers = ['o', 's', '^', '*', 'd', 'x']

    fig = plt.figure(figsize=(14, 8))

    # --- 主图 (占据左 60%) ---
    ax_main = fig.add_axes([0.06, 0.12, 0.52, 0.82])
    _draw_stems(ax_main, true_delays, true_amps, est_delays_list, est_amps_list, colors, markers, labels)
    ax_main.set_xlabel('Delay (ms)', fontsize=11)
    ax_main.set_ylabel(r'$|\alpha|$', fontsize=11)
    ax_main.legend(fontsize=12, loc='upper right', ncol=2)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(xlim)
    ax_main.set_ylim([0, None])

    # 在每条真实多径附近画虚线框
    for i, (td, ta) in enumerate(zip(true_delays, true_amps)):
        box_w = 4.0
        box_h = np.abs(ta) * 1.3
        rect = plt.Rectangle((td-box_w/2, 0), box_w, box_h,
                             fill=False, linestyle='--', linewidth=1,
                             edgecolor='gray', alpha=0.6)
        ax_main.add_patch(rect)

    # --- 三条放大子图 (占据右 40%) ---
    sub_positions = [
        [0.63, 0.58, 0.34, 0.35],  # 路径 1
        [0.63, 0.32, 0.34, 0.22],  # 路径 2
        [0.63, 0.06, 0.34, 0.22],  # 路径 3
    ]

    for i, (td, ta, pos) in enumerate(zip(true_delays, true_amps, sub_positions)):
        ax_sub = fig.add_axes(pos)
        _draw_stems(ax_sub, true_delays, true_amps, est_delays_list, est_amps_list,
                    colors, markers, labels=None)
        ax_sub.set_xlim([td - 2, td + 2])
        ax_sub.set_ylim([0, np.abs(ta) * 1.3])
        ax_sub.set_title(f'Path {i+1}: $\\tau \\approx$ {td:.1f}', fontsize=11)
        ax_sub.grid(True, alpha=0.3)
        ax_sub.tick_params(labelsize=7)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _draw_stems(ax, true_delays, true_amps, est_delays_list, est_amps_list,
                colors, markers, labels=None):
    """在指定轴上绘制 stem 图"""
    ax.stem(true_delays, np.abs(true_amps), linefmt='r-', markerfmt='ro',
            basefmt=' ', label='True' if labels else None)
    offsets = np.linspace(-0.25, 0.25, len(est_delays_list))
    for j, (delays, amps) in enumerate(zip(est_delays_list, est_amps_list)):
        ax.stem(delays + offsets[j], np.abs(amps),
                linefmt=f'{colors[j+1]}-', markerfmt=f'{colors[j+1]}{markers[j+1]}',
                basefmt=' ', label=labels[j] if labels else None)


def plot_residual_comparison(
    residual_list: List[np.ndarray],
    labels: List[str],
    xlim: tuple = (30, 250),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    残差信号对比图。

    Parameters
    ----------
    residual_list : 各算法残差信号列表
    labels : 对应图例标签
    """
    colors = ['k', 'm', 'c', 'b', 'g', 'r']
    linewidths = [1.0, 1.0, 1.0, 1.5, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(12, 4))

    for i, (res, label) in enumerate(zip(residual_list, labels)):
        energy = np.sum(np.abs(res) ** 2)
        ax.plot(
            np.abs(res),
            color=colors[i],
            linewidth=linewidths[i],
            label=f'{label} (energy={energy:.1f})',
        )

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('|Residual|')
    ax.set_title('Residual Error Comparison')
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_grid_mismatch(
    down_ratios: np.ndarray,
    mse_int: np.ndarray,
    mse_frac: np.ndarray,
    improvement: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    栅格失配敏感度双轴图。
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1, color2 = 'b', 'r'
    ax1.semilogy(down_ratios, mse_int, 'b-s', linewidth=1.5, markersize=6,
                 label='Standard SAGE (Integer Grid)')
    ax1.semilogy(down_ratios, mse_frac, 'r-o', linewidth=1.5, markersize=6,
                 label='Proposed SAGE (Fractional Grid)')
    ax1.set_xlabel('Decimation Factor')
    ax1.set_ylabel('Equalization MSE', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    bars = ax2.bar(down_ratios, improvement, width=0.2, color='gray',
                   alpha=0.4, edgecolor='none', label='Improvement (%)')
    ax2.set_ylabel('Performance Improvement (%)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)

    ax1.set_title('Impact of Sampling Grid Misalignment')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
