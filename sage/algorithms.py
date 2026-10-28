"""
核心算法模块
============
实现六种多径时延估计算法:
1. algo_SAGE  — SAGE (整数模式 + 亚采样分数模式)
2. algo_MF    — 匹配滤波器
3. algo_OMP   — 正交匹配追踪
4. algo_NOMP  — Newtonized OMP (连续参数精化)
5. algo_SBL   — 稀疏贝叶斯学习 (含 Woodbury 加速)

所有算法返回 (est_delays, est_amps) 元组。
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Tuple

from sage.signal_utils import fractional_delay_shift
from sage.interpolation import parabolic_delta


# =========================================================================
# 1. SAGE 算法
# =========================================================================

def algo_SAGE(
    y_rx: np.ndarray, s_base: np.ndarray, N_rx: int, L_paths: int,
    max_iter: int = 15, do_frac: bool = True, use_power: bool = True,
    use_fft_corr: bool = True, local_refine: bool = True,
    refine_step: float = 0.005,
    accelerate: str = 'full',  # 'full' | 'coarse' | 'newton'
) -> Tuple[np.ndarray, np.ndarray]:
    N_seq = len(s_base)
    Es = np.vdot(s_base, s_base).real
    search_len = N_rx - N_seq
    est_delays = np.zeros(L_paths)
    est_amps = np.zeros(L_paths, dtype=complex)
    residual = y_rx.copy()

    for it in range(max_iter):
        for l in range(L_paths):
            if it > 0:
                s_old = est_amps[l] * fractional_delay_shift(s_base, est_delays[l], N_rx)
                x_l = residual + s_old
            else:
                x_l = residual.copy()

            if use_fft_corr:
                X_f = fft(x_l)
                S_f_conj = np.conj(fft(s_base, n=N_rx))
                corr_abs = np.abs(ifft(X_f * S_f_conj) / Es)[:search_len]
            else:
                corr_abs = np.zeros(search_len)
                for tau in range(search_len):
                    corr_abs[tau] = np.abs(np.vdot(s_base, x_l[tau:tau+N_seq]))

            if it == 0:
                k_opt = int(np.argmax(corr_abs))
            else:
                search_radius = 10
                old_k = int(np.floor(est_delays[l] + 0.5))
                lo = max(0, old_k - search_radius)
                hi = min(search_len - 1, old_k + search_radius)
                if lo < hi:
                    k_opt = lo + int(np.argmax(corr_abs[lo:hi+1]))
                else:
                    k_opt = int(np.argmax(corr_abs))

            if do_frac and 0 < k_opt < search_len - 1:
                if use_power:
                    y_m1 = corr_abs[k_opt-1]**2
                    y_0 = corr_abs[k_opt]**2
                    y_p1 = corr_abs[k_opt+1]**2
                else:
                    y_m1, y_0, y_p1 = corr_abs[k_opt-1], corr_abs[k_opt], corr_abs[k_opt+1]
                delta = parabolic_delta(y_m1, y_0, y_p1)
                tau_est = float(k_opt) + delta
            else:
                tau_est = float(k_opt)

            if local_refine and do_frac:
                tau_est, amp_est = _local_refine(x_l, s_base, tau_est, N_rx, refine_step, accelerate)
            else:
                s_temp = fractional_delay_shift(s_base, tau_est, N_rx)
                amp_est = np.vdot(s_temp, x_l) / Es

            est_delays[l] = tau_est
            est_amps[l] = amp_est
            residual = x_l - amp_est * fractional_delay_shift(s_base, tau_est, N_rx)

        if it == 0:
            sort_idx = np.argsort(-np.abs(est_amps))
            est_delays = est_delays[sort_idx]
            est_amps = est_amps[sort_idx]
            for i in range(L_paths):
                for j in range(i+1, L_paths):
                    if abs(est_delays[i] - est_delays[j]) < 2.0:
                        if np.abs(est_amps[i]) >= np.abs(est_amps[j]):
                            est_delays[j] += 3.0
                        else:
                            est_delays[i] += 3.0

    return est_delays, est_amps


def _local_refine(x_l, s_base, tau_coarse, N_rx, step=0.005, accelerate='full'):
    """
    Stage 2 局部精化 — 支持三种模式:
      'full'   : 全网格搜索 (200 次, 安全)
      'coarse' : 粗细搜索 (40 次, 5× 加速)
      'newton' : Newton 精化 (9 次, 22× 加速)
    """
    Es = np.vdot(s_base, s_base).real

    if accelerate == 'newton':
        # Newton 精化: 频域导数
        return _refine_newton(x_l, s_base, tau_coarse, N_rx, Es)

    if accelerate == 'coarse':
        # 粗细两阶段
        # Pass 1: coarse step = 0.05
        coarse_step = 0.05
        taus_c = np.arange(max(0, tau_coarse - 0.5), tau_coarse + 0.5 + coarse_step, coarse_step)
        best_obj = -np.inf; tau_c = tau_coarse
        for t in taus_c:
            s_t = fractional_delay_shift(s_base, t, N_rx)
            amp_t = np.vdot(s_t, x_l) / Es
            obj = -np.sum(np.abs(x_l - amp_t * s_t) ** 2)
            if obj > best_obj:
                best_obj, tau_c = obj, t
        # Pass 2: fine step = 0.005 around coarse peak
        fine_range = 0.05
        taus_f = np.arange(max(0, tau_c - fine_range), tau_c + fine_range + step, step)
        best_obj = -np.inf; tau_best = tau_c; amp_best = 0j
        for t in taus_f:
            s_t = fractional_delay_shift(s_base, t, N_rx)
            amp_t = np.vdot(s_t, x_l) / Es
            obj = -np.sum(np.abs(x_l - amp_t * s_t) ** 2)
            if obj > best_obj:
                best_obj, tau_best, amp_best = obj, t, amp_t
        return tau_best, amp_best

    # 默认 'full': 全网格搜索
    best_obj, tau_best, amp_best = -np.inf, tau_coarse, 0j
    for tau_test in np.linspace(max(0, tau_coarse - 0.5), tau_coarse + 0.5, max(3, int(1.0 / step))):
        s_test = fractional_delay_shift(s_base, tau_test, N_rx)
        amp_test = np.vdot(s_test, x_l) / Es
        obj = -np.sum(np.abs(x_l - amp_test * s_test) ** 2)
        if obj > best_obj:
            best_obj, tau_best, amp_best = obj, tau_test, amp_test
    return tau_best, amp_best


def _refine_newton(x_l, s_base, tau_coarse, N_rx, Es, max_iter=3):
    """Newton 精化: 最大化 f(τ)=|⟨s(τ), x_l⟩|²"""
    N_seq = len(s_base)
    S_f = np.fft.fft(s_base, n=N_rx)
    X_f = np.fft.fft(x_l, n=N_rx)
    k_vec = np.concatenate([
        np.arange(0, int(np.ceil(N_rx / 2))),
        np.arange(-int(np.floor(N_rx / 2)), 0),
    ])
    w = 2 * np.pi * k_vec / N_rx
    Sp = (-1j * w) * S_f     # conj(Sp) = jω·conj(S) → correct C'(τ)
    Sd = -(w ** 2) * S_f

    tau = tau_coarse
    for _ in range(max_iter):
        ph = np.exp(1j * w * tau)
        c = np.sum(np.conj(S_f) * X_f * ph) / (N_rx * Es)
        cp = np.sum(np.conj(Sp) * X_f * ph) / (N_rx * Es)
        cd = np.sum(np.conj(Sd) * X_f * ph) / (N_rx * Es)
        fp = 2 * np.real(np.conj(c) * cp)
        fd = 2 * np.real(np.conj(c) * cd + np.abs(cp) ** 2)
        if abs(fd) < 1e-12:
            break
        delta = np.clip(fp / fd, -0.5, 0.5)
        tau = tau - delta
        tau = max(0.0, min(float(N_rx), tau))
        if abs(delta) < 1e-8:
            break

    s_t = fractional_delay_shift(s_base, tau, N_rx)
    amp = np.vdot(s_t, x_l) / Es
    return tau, amp


# =========================================================================
# 2. 匹配滤波器
# =========================================================================

def algo_MF(y_rx, A_dict, tau_grid, L_paths):
    corr_mf = np.abs(A_dict.conj().T @ y_rx)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(corr_mf, distance=3)
        if len(peaks) >= L_paths:
            idx_mf = peaks[np.argsort(corr_mf[peaks])[::-1]][:L_paths]
        else:
            idx_mf = np.argsort(corr_mf)[::-1][:L_paths]
    except ImportError:
        idx_mf = np.argsort(corr_mf)[::-1][:L_paths]
    est_delays = tau_grid[idx_mf]
    est_amps, _, _, _ = np.linalg.lstsq(A_dict[:, idx_mf], y_rx, rcond=None)
    return est_delays, est_amps


# =========================================================================
# 1b. SAGE with configurable M-step solver (baselines for ablation study)
# =========================================================================

def algo_SAGE_baseline(
    y_rx: np.ndarray, s_base: np.ndarray, N_rx: int, L_paths: int,
    max_iter: int = 15, mstep: str = 'two_stage',
    refine_step: float = 0.005, use_fft_corr: bool = True,
    search_radius: int = 10, eval_counter: list = None,
    collision_on: bool = True, trace: dict = None,
    zpfft_factor: int = 200, gg_step: float = 0.02,
    refine_radius: float = 0.5, refine_center: str = 'integer',
    refine_points: int = None, safeguard: bool = True,
    window_profile=None, snapshot_steps=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SAGE with configurable M-step solver. All variants share identical
    E-step, initialization, path sorting, and collision handling — the only
    difference is the M-step delay refinement.

    mstep options:
      'integer'     : integer-grid argmax only (IG-SAGE baseline)
      'stage1'      : integer peak + three-point parabolic interpolation
      'stage2_int'  : integer peak + local fine grid search centered at k_opt
      'two_stage'   : Stage 1 (parabolic) + Stage 2 (local grid) [proposed]
      'brent'       : continuous M-step via Brent's method on [k_opt±0.5]
      'golden'      : continuous M-step via golden-section search
      'newton'      : continuous M-step via 3-iter Newton refinement
      'window_grid'/'window_bounded': instrumented radius/center study;
                      center='integer' or 'stage1', optional fixed point budget.
      'zpfft'       : oversampled correlation via ZERO-PADDED FFT with factor
                      zpfft_factor (grid step 1/zpfft_factor); argmax over the
                      same search region as integer mode (full range at it=0,
                      ±search_radius window afterwards). [MC7 baseline]
      'global_grid' : template-based exhaustive fine grid with step gg_step
                      over the same search region (full range at it=0, ±10
                      window afterwards). [MC7 baseline]

    eval_counter (optional list of length 1): counts fractional-delay
    template generations + inner products per M-step (complexity metric).

    collision_on (default True): toggles the ad-hoc "weak path perturbed by
    +3 samples when two paths are closer than 2 samples" heuristic (must be
    ablated separately per Reviewer Major Comment 7).

    trace (optional dict): if given, appends (it, l, k_opt) to trace['k_opt']
    and (it, l, tau_est) to trace['tau_est'] for every M-step.
    """
    from scipy.optimize import minimize_scalar
    from sage.interpolation import parabolic_delta
    study_mode = mstep in ('window_grid', 'window_bounded')
    if study_mode:
        from sage.window_refinement import DelayProfile, refine_window
        if refine_center not in ('integer', 'stage1'):
            raise ValueError('refine_center must be integer or stage1')
        if collision_on:
            raise ValueError('window study requires collision_on=False')
        profile = window_profile or DelayProfile(s_base, N_rx)

    N_seq = len(s_base)
    Es = np.vdot(s_base, s_base).real
    search_len = N_rx - N_seq + 1
    est_delays = np.zeros(L_paths)
    est_amps = np.zeros(L_paths, dtype=complex)
    residual = y_rx.copy()
    if eval_counter is None:
        eval_counter = [0]

    def _count(n=1):
        eval_counter[0] += n

    def _obj(tau):
        """Objective f(tau) = |<s(tau), x_l>|^2, maximized over tau."""
        _count(1)
        s_t = fractional_delay_shift(s_base, tau, N_rx)
        return np.abs(np.vdot(s_t, x_l)) ** 2

    def _grid_search(center, lo, hi, step):
        """Exhaustive local grid search with fixed step."""
        best_tau, best_obj = center, -np.inf
        taus = np.arange(max(0, lo), min(hi, hi + step), step)
        for t in taus:
            o = _obj(t)
            if o > best_obj:
                best_obj, best_tau = o, t
        return best_tau

    for it in range(max_iter):
        for l in range(L_paths):
            if it > 0:
                s_old = est_amps[l] * fractional_delay_shift(s_base, est_delays[l], N_rx)
                x_l = residual + s_old
            else:
                x_l = residual.copy()

            if study_mode and use_fft_corr:
                corr_abs = profile.integer_correlation(x_l) / Es
            elif use_fft_corr:
                X_f = fft(x_l)
                S_f_conj = np.conj(fft(s_base, n=N_rx))
                corr_abs = np.abs(ifft(X_f * S_f_conj) / Es)[:search_len]
            else:
                corr_abs = np.zeros(search_len)
                for tau in range(search_len):
                    corr_abs[tau] = np.abs(np.vdot(s_base, x_l[tau:tau + N_seq]))

            if it == 0:
                k_opt = int(np.argmax(corr_abs))
            else:
                old_k = int(np.floor(est_delays[l] + 0.5))
                lo = max(0, old_k - search_radius)
                hi = min(search_len - 1, old_k + search_radius)
                if lo < hi:
                    k_opt = lo + int(np.argmax(corr_abs[lo:hi + 1]))
                else:
                    k_opt = int(np.argmax(corr_abs))

            if trace is not None:
                trace.setdefault('k_opt', []).append((it, l, k_opt))
                if snapshot_steps is not None and (it, l) in snapshot_steps:
                    trace.setdefault('snapshots', {})[(it, l)] = x_l.copy()

            # ---------- M-step: delay refinement per variant ----------
            if study_mode:
                from time import perf_counter
                t_stage1 = perf_counter()
                if refine_center == 'stage1' and 0 < k_opt < search_len - 1:
                    delta = float(parabolic_delta(*(corr_abs[k_opt-1:k_opt+2] ** 2)))
                else:
                    delta = 0.0
                center = float(k_opt) + delta
                stage1_s = perf_counter() - t_stage1 if refine_center == 'stage1' else 0.0
                wr = refine_window(profile, x_l, center, refine_radius,
                                   solver=mstep.removeprefix('window_'), step=refine_step,
                                   points=refine_points,
                                   old_tau=float(est_delays[l]) if safeguard and it > 0 else None)
                _count(wr.nfev)
                tau_est = wr.tau
                if trace is not None:
                    trace.setdefault('window', []).append((
                        it, l, k_opt, delta, center, wr.lo, wr.hi, wr.nfev,
                        wr.window_nfev, wr.objective, wr.old_objective,
                        wr.window_tau, float(wr.used_old), float(wr.success),
                        stage1_s, wr.elapsed_s))
            elif mstep == 'integer':
                tau_est = float(k_opt)
            elif mstep == 'stage1':
                if 0 < k_opt < search_len - 1:
                    y_m1 = corr_abs[k_opt - 1] ** 2
                    y_0 = corr_abs[k_opt] ** 2
                    y_p1 = corr_abs[k_opt + 1] ** 2
                    tau_est = float(k_opt) + parabolic_delta(y_m1, y_0, y_p1)
                else:
                    tau_est = float(k_opt)
            elif mstep == 'stage2_int':
                # Direct local grid search centered at the INTEGER peak —
                # tests whether Stage 1 provides any independent benefit.
                tau_est = _grid_search(float(k_opt), k_opt - 0.5, k_opt + 0.5, refine_step)
            elif mstep == 'two_stage':
                if 0 < k_opt < search_len - 1:
                    y_m1 = corr_abs[k_opt - 1] ** 2
                    y_0 = corr_abs[k_opt] ** 2
                    y_p1 = corr_abs[k_opt + 1] ** 2
                    tau_c = float(k_opt) + parabolic_delta(y_m1, y_0, y_p1)
                else:
                    tau_c = float(k_opt)
                tau_est = _grid_search(tau_c, tau_c - 0.5, tau_c + 0.5, refine_step)
            elif mstep == 'brent':
                res = minimize_scalar(lambda t: -_obj(t),
                                      bounds=(max(0, k_opt - 0.5), k_opt + 0.5),
                                      method='bounded', options={'xatol': 1e-8})
                tau_est = float(res.x)
            elif mstep == 'golden':
                # Golden-section on fixed interval [k_opt-0.5, k_opt+0.5]
                # (unimodal objective; no bracket validation needed)
                a, b = max(0, k_opt - 0.5), float(k_opt + 0.5)
                gr = (np.sqrt(5) - 1) / 2
                c, d = b - gr * (b - a), a + gr * (b - a)
                fc_, fd_ = _obj(c), _obj(d)
                for _ in range(60):  # (0.618)^60 << 1e-10
                    if fc_ > fd_:
                        b, d, fd_ = d, c, fc_
                        c = b - gr * (b - a)
                        fc_ = _obj(c)
                    else:
                        a, c, fc_ = c, d, fd_
                        d = a + gr * (b - a)
                        fd_ = _obj(d)
                tau_est = 0.5 * (a + b)
            elif mstep == 'newton':
                # Newton-SAGE baseline: plain frequency-domain Newton on
                # f(tau)=|<s(tau), x_l>|^2 starting from the integer peak.
                # NOTE: converges on clean single-path signals (err ~0.003)
                # but drifts/mislocks inside the SAGE loop (RMSE >= 20):
                # the E-step signal carries residual interference whose
                # gradient bias accumulates across iterations. This is a
                # real property of Newton-based M-step refinement and is
                # reported as such in the ablation.
                tau_est, _ = _refine_newton(x_l, s_base, float(k_opt), N_rx, Es)
            elif mstep in ('zpfft', 'global_grid'):
                # Same search region as the integer argmax (full range at
                # it=0, ±search_radius window afterwards), evaluated on a
                # fine grid instead of the integer grid.
                lo_r = 0 if it == 0 else max(0, old_k - search_radius)
                hi_r = (search_len - 1) if it == 0 else min(search_len - 1,
                                                            old_k + search_radius)
                if mstep == 'zpfft':
                    # zero-padded-FFT oversampled correlation (spectral
                    # zero-padding = bandlimited sinc interpolation of the
                    # integer-lag correlation, evaluated on grid step 1/Z).
                    # NOTE: time-domain zero-padding does NOT produce a
                    # fractional-lag grid (verified: peak stays at integer
                    # index); the spectrum must be zero-padded instead.
                    Z = zpfft_factor
                    X_f = fft(x_l)
                    S_f = fft(s_base, n=N_rx)
                    C = X_f * np.conj(S_f)          # spectrum of int-lag corr
                    C_pad = np.zeros(Z * N_rx, dtype=complex)
                    n2 = N_rx // 2
                    C_pad[:n2] = C[:n2]
                    C_pad[Z * N_rx - (N_rx - n2):] = C[n2:]
                    corr_f = np.abs(ifft(C_pad))
                    lo_k = Z * lo_r
                    hi_k = min(Z * search_len - 1, Z * hi_r)
                    k_best = lo_k + int(np.argmax(corr_f[lo_k:hi_k + 1]))
                    tau_est = k_best / Z
                else:
                    # template-based global fine grid (costly, honest baseline)
                    taus = np.arange(lo_r, hi_r + gg_step / 2, gg_step)
                    best_t, best_o = float(k_opt), -np.inf
                    for t in taus:
                        _count(1)
                        s_t = fractional_delay_shift(s_base, t, N_rx)
                        o = np.abs(np.vdot(s_t, x_l)) ** 2
                        if o > best_o:
                            best_o, best_t = o, t
                    tau_est = best_t
            else:
                raise ValueError(f'Unknown mstep: {mstep}')

            if study_mode:
                s_temp = profile.template(tau_est)
                amp_est = wr.amp
            else:
                s_temp = fractional_delay_shift(s_base, tau_est, N_rx)
                amp_est = np.vdot(s_temp, x_l) / Es

            if trace is not None:
                trace.setdefault('tau_est', []).append((it, l, tau_est))

            est_delays[l] = tau_est
            est_amps[l] = amp_est
            residual = x_l - amp_est * s_temp

        if it == 0:
            sort_idx = np.argsort(-np.abs(est_amps))
            if trace is not None:
                trace['initial_sort'] = sort_idx.copy()
            est_delays = est_delays[sort_idx]
            est_amps = est_amps[sort_idx]
            if collision_on:
                for i in range(L_paths):
                    for j in range(i + 1, L_paths):
                        if abs(est_delays[i] - est_delays[j]) < 2.0:
                            if np.abs(est_amps[i]) >= np.abs(est_amps[j]):
                                est_delays[j] += 3.0
                            else:
                                est_delays[i] += 3.0

    return est_delays, est_amps

def algo_OMP(y_rx, A_dict, tau_grid, L_paths):
    residual = y_rx.copy()
    idx_omp = []
    for _ in range(L_paths):
        corr_omp = np.abs(A_dict.conj().T @ residual)
        best_idx = int(np.argmax(corr_omp))
        idx_omp.append(best_idx)
        A_sub = A_dict[:, idx_omp]
        amps_omp, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)
        residual = y_rx - A_sub @ amps_omp
    idx_omp = np.array(idx_omp)
    est_delays = tau_grid[idx_omp]
    est_amps, _, _, _ = np.linalg.lstsq(A_dict[:, idx_omp], y_rx, rcond=None)
    return est_delays, est_amps


# =========================================================================
# 4. Newtonized OMP (NOMP)
# =========================================================================

def algo_NOMP(
    y_rx, s_base, N_rx, L_paths, tau_grid,
    newton_iters=3, cyclic_iters=1,
):
    N_seq = len(s_base)
    search_len = N_rx - N_seq

    # 频域预计算
    S_f = fft(s_base)

    residual = y_rx.copy()
    active_delays = []

    for l in range(L_paths):
        # 1. 时域粗搜索
        corr_abs = np.zeros(search_len)
        for tau in range(search_len):
            corr_abs[tau] = np.abs(np.vdot(s_base, residual[tau:tau+N_seq]))
        k_opt = int(np.argmax(corr_abs))
        tau_current = float(k_opt)

        # 2. Newton 精化 (窗内)
        win_start = max(0, k_opt - 1)
        win_end = min(N_rx, k_opt + N_seq + 1)
        r_win = residual[win_start:win_end]
        tau_local = tau_current - win_start

        tau_local = _newton_refine(r_win, s_base, tau_local, newton_iters)
        tau_current = win_start + tau_local

        # 3. 冲突检测
        for _retry in range(3):
            too_close = any(abs(tau_current - t) < 3.0 for t in active_delays)
            if not too_close:
                break
            lo, hi = max(0, k_opt-5), min(search_len-1, k_opt+5)
            corr_abs[lo:hi+1] = 0
            if np.max(corr_abs) <= 0:
                tau_current = float(k_opt) + 10.0
                break
            k_opt = int(np.argmax(corr_abs))
            tau_current = float(k_opt)
            w0 = max(0, k_opt-1)
            w1 = min(N_rx, k_opt+N_seq+1)
            r_win = residual[w0:w1]
            tau_current = w0 + _newton_refine(r_win, s_base, tau_current-w0, newton_iters)

        active_delays.append(tau_current)

        # 4. LS + cyclic refinement
        for cyc in range(cyclic_iters + 1):
            A_sub = np.zeros((N_rx, len(active_delays)), dtype=complex)
            for a, t in enumerate(active_delays):
                A_sub[:, a] = fractional_delay_shift(s_base, t, N_rx)
            amps_all, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)

            if cyc == cyclic_iters:
                break

            for a in range(len(active_delays)):
                r_other = y_rx.copy()
                for b in range(len(active_delays)):
                    if b != a:
                        r_other -= amps_all[b] * fractional_delay_shift(s_base, active_delays[b], N_rx)
                ka = int(np.floor(active_delays[a]+0.5))
                w0, w1 = max(0, ka-1), min(N_rx, ka+N_seq+1)
                active_delays[a] = w0 + _newton_refine(r_other[w0:w1], s_base, active_delays[a]-w0, newton_iters)

        # 5. 残差更新
        A_sub = np.zeros((N_rx, len(active_delays)), dtype=complex)
        for a, t in enumerate(active_delays):
            A_sub[:, a] = fractional_delay_shift(s_base, t, N_rx)
        amps_all, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)
        residual = y_rx - A_sub @ amps_all

    return np.array(active_delays), amps_all


def _newton_refine(r_win, s_base, tau, max_iters=3):
    """Hann-窗 Newton 精化: 最大化 |⟨s(τ), r_win⟩|²

    消除循环FFT伪影的关键: 对截断窗口 r_win 和补零后的 s_base 同时施加
    Hann 窗, 使信号在窗口边界平滑衰减至零。这消除了 FFT 视图中由截断引入的
    虚假高频分量, 是 NOMP 原文 (Mamandipoor 2016) 推荐的标准做法。
    """
    N_win = len(r_win)
    N_seq = len(s_base)

    # Hann 窗: 消除边界不连续性 → 消除循环FFT高频伪影
    window = np.hanning(N_win)

    R_f = fft(r_win * window)
    S_w = fft(s_base, n=N_win)           # 补零到窗口长度
    S_w = S_w * np.sqrt(np.mean(window**2))  # 能量补偿

    k_w = np.concatenate([
        np.arange(0, int(np.ceil(N_win / 2))),
        np.arange(-int(np.floor(N_win / 2)), 0),
    ])
    omega = 2 * np.pi * k_w / N_win
    Sp = (-1j * omega) * S_w        # conj(Sp)=jω·conj(S) → C'(τ)
    Sd = -(omega ** 2) * S_w        # conj(Sd)=-ω²·conj(S) → C''(τ)

    for _ in range(max_iters):
        ph = np.exp(1j * omega * tau)
        c  = np.sum(np.conj(S_w) * R_f * ph) / N_win
        cp = np.sum(np.conj(Sp)  * R_f * ph) / N_win
        cd = np.sum(np.conj(Sd)  * R_f * ph) / N_win

        fp = 2 * np.real(np.conj(c) * cp)
        fd = 2 * (np.abs(cp) ** 2 + np.real(np.conj(c) * cd))

        if abs(fd) < 1e-12:
            break
        delta = np.clip(fp / fd, -0.5, 0.5)
        tau = max(0.0, min(float(N_win - N_seq), tau - delta))
        if abs(delta) < 1e-6:
            break
    return tau


# =========================================================================
# 5. 稀疏贝叶斯学习 (SBL)
# =========================================================================

def algo_SBL(y_rx, A_dict, tau_grid, L_paths, N0=None, max_iter=50, use_woodbury=True):
    """
    稀疏贝叶斯学习 (SBL) — 自适应噪声方差版。

    修复:
      1. N0=None → 自适应估计噪声方差 σ²
      2. gamma 初始化使用 MF 相关而非均匀值
      3. 收敛条件同时检查 gamma 和 σ²

    Parameters
    ----------
    N0 : float or None
        噪声方差。None 时自适应估计 (推荐)。
    """
    N_rx, N_grid = A_dict.shape
    gamma_tol = 1e-6
    eps = 1e-6

    # ---- 初始化 ----
    # gamma: 使用 MF 相关能量 (归一化)
    gamma = np.abs(A_dict.conj().T @ y_rx)**2
    gamma = gamma / np.max(gamma) * 0.1  # 缩放到合理范围
    gamma = np.maximum(gamma, gamma_tol)

    # 噪声方差: 若未指定, 初始化为残差方差的 50%
    if N0 is None:
        N0 = 0.5 * np.var(y_rx)
    sigma2 = N0

    for it in range(max_iter):
        # ---- E-step: 计算后验均值 μ 和对角协方差 ----
        if use_woodbury:
            inv_gamma = np.where(gamma > gamma_tol, 1.0/gamma, 0.0)
            M_mat = sigma2*np.diag(inv_gamma) + A_dict.conj().T @ A_dict + eps*np.eye(N_grid)
            try:
                inv_M = np.linalg.solve(M_mat, np.eye(N_grid))
            except np.linalg.LinAlgError:
                inv_M = np.linalg.pinv(M_mat)
            Ahy = A_dict.conj().T @ y_rx
            B = A_dict.conj().T @ A_dict
            mu = (gamma/sigma2)*(Ahy - B @ (inv_M @ Ahy))
            # Σ_diag[k] = γ_k - γ_k²·a_k^H·Σ_y^{-1}·a_k
            diag_term = np.zeros(N_grid)
            for k in range(N_grid):
                col_k = B[:, k]
                diag_term[k] = (B[k,k].real - np.real(col_k.conj().T @ (inv_M @ col_k)))/sigma2
        else:
            Gamma = np.diag(gamma)
            Sigma_y = A_dict @ Gamma @ A_dict.conj().T + sigma2*np.eye(N_rx)
            try:
                inv_Sigma_y = np.linalg.solve(Sigma_y+eps*np.eye(N_rx), np.eye(N_rx))
            except np.linalg.LinAlgError:
                inv_Sigma_y = np.linalg.pinv(Sigma_y+eps*np.eye(N_rx))
            mu = Gamma @ A_dict.conj().T @ inv_Sigma_y @ y_rx
            diag_term = np.sum(A_dict.conj().T*(A_dict.conj().T @ inv_Sigma_y).T, axis=1).real

        # Σ_diag = gamma - gamma²·diag_term
        Sigma_diag = gamma - gamma**2 * diag_term
        Sigma_diag = np.maximum(Sigma_diag, 0.0)  # 非负约束

        # ---- M-step: 更新 gamma ----
        gamma_new = np.abs(mu)**2 + Sigma_diag
        gamma_new = np.maximum(gamma_new, gamma_tol)

        # ---- 自适应噪声方差更新 (Type II ML) ----
        residual = y_rx - A_dict @ mu
        # 有效参数数
        gamma_eff = np.sum(gamma / (gamma + sigma2))
        sigma2_new = (np.linalg.norm(residual)**2) / max(N_rx - gamma_eff, 1.0)
        # 平滑更新 (防止震荡)
        sigma2 = 0.8 * sigma2 + 0.2 * sigma2_new
        sigma2 = max(sigma2, 1e-10)

        # ---- 收敛检查 ----
        gamma_change = np.linalg.norm(gamma_new - gamma) / (np.linalg.norm(gamma) + 1e-10)
        if gamma_change < 1e-3 and it > 5:
            gamma = gamma_new
            break
        gamma = gamma_new

    # ---- 寻峰 ----
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(gamma, height=0.01*np.max(gamma), distance=5)
        if len(peaks) >= L_paths:
            idx_sbl = peaks[np.argsort(gamma[peaks])[::-1]][:L_paths]
        else:
            idx_sbl = np.argsort(gamma)[::-1][:L_paths]
    except ImportError:
        idx_sbl = np.argsort(gamma)[::-1][:L_paths]

    return tau_grid[idx_sbl], mu[idx_sbl]


# =========================================================================
# 6. Root-MUSIC — 子空间方法 / 连续时延估计
# =========================================================================

def algo_ESPRIT(
    y_rx: np.ndarray,
    s_base: np.ndarray,
    L_paths: int,
    N_rx: int = None,
    M_smooth: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ESPRIT 时延估计 (Frequency-Domain TLS-ESPRIT)。

    比 Root-MUSIC 更鲁棒: 无需多项式求根,
    利用信号子空间的旋转不变性直接估计时延。

    原理:
      H(f) = Y(f)/S(f) = Σ a_k·e^{-j2πfτ_k}
      → 选择矩阵 J₁, J₂ → 最小二乘 → 特征值 → 时延
    """
    if N_rx is None:
        N_rx = len(y_rx)

    S_f = np.fft.fft(s_base, n=N_rx)
    Y_f = np.fft.fft(y_rx, n=N_rx)

    # 正半频带: 连续单调频率, Vandermonde 结构正确
    half = N_rx // 2
    H_raw = Y_f[:half] / (S_f[:half] + 1e-12)
    M = len(H_raw)
    delta_f = 1.0 / N_rx

    if M_smooth is None:
        M_smooth = M // 2
    L = M - M_smooth + 1
    if L <= L_paths:
        L = L_paths + 1
        M_smooth = M - L + 1

    # Hankel 矩阵
    H_hankel = np.zeros((L, M_smooth), dtype=complex)
    for i in range(L):
        H_hankel[i, :] = H_raw[i:i + M_smooth]

    # SVD
    U, s, Vh = np.linalg.svd(H_hankel, full_matrices=False)
    Us = U[:, :L_paths]  # 信号子空间

    # TLS-ESPRIT: J₁Us·Ψ ≈ J₂Us
    J1 = Us[:-1, :]  # 去掉最后一行
    J2 = Us[1:, :]   # 去掉第一行

    # 总体最小二乘 (TLS)
    UV = np.hstack([J1, J2])
    _, _, V_tls = np.linalg.svd(UV, full_matrices=False)
    V12 = V_tls[:L_paths, L_paths:]  # 右上块
    V22 = V_tls[L_paths:, L_paths:]  # 右下块
    Psi = -V12 @ np.linalg.pinv(V22)

    # 特征值 → 时延
    eigvals = np.linalg.eigvals(Psi)
    # 只保留单位圆内的特征值
    valid = np.abs(eigvals) <= 1.0
    if np.sum(valid) < L_paths:
        valid = np.abs(np.abs(eigvals) - 1.0) < 0.5

    z_k = eigvals[valid]
    # 按幅值排序, 取前 K 个
    idx = np.argsort(-np.abs(z_k))[:L_paths]
    z_k = z_k[idx]

    delays = np.mod(-np.angle(z_k) / (2 * np.pi * delta_f), N_rx)
    delays = np.sort(delays)

    # 最小二乘幅度
    from sage.signal_utils import fractional_delay_shift
    A = np.zeros((N_rx, L_paths), dtype=complex)
    for k in range(L_paths):
        A[:, k] = fractional_delay_shift(s_base, delays[k], N_rx)
    amps, _, _, _ = np.linalg.lstsq(A, y_rx, rcond=None)

    return delays, amps


# 保留 Root-MUSIC 作为别名 (向后兼容)
def algo_ROOTMUSIC(*args, **kwargs):
    return algo_ESPRIT(*args, **kwargs)
