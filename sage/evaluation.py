"""One-to-one evaluation. Delay in samples, phase in radians.

Known-L RMSE requires equal counts; detection supports unequal counts.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


def _vector(value, name, complex_ok=False):
    a = np.asarray(value, dtype=complex if complex_ok else float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return a


@dataclass(frozen=True)
class PathAssignment:
    true_indices: np.ndarray
    estimate_indices: np.ndarray
    unmatched_true: np.ndarray
    unmatched_estimates: np.ndarray
    squared_delay_cost: float


def assign_paths(true_delays, est_delays, *, max_delay_error=None):
    """One-to-one minimum squared delay error; gated matching maximizes TP first.

Do not reject edges of an ungated optimum afterwards: that can lose valid TPs.
"""
    td, ed = _vector(true_delays, 'true_delays'), _vector(est_delays, 'est_delays')
    n, m = len(td), len(ed)
    if max_delay_error is not None and (not np.isfinite(max_delay_error) or max_delay_error <= 0):
        raise ValueError('max_delay_error must be finite and positive')
    if not n or not m:
        ri, ci = np.array([], dtype=int), np.array([], dtype=int)
    else:
        distance = np.abs(td[:, None]-ed[None, :])
        if max_delay_error is None:
            ri, ci = linear_sum_assignment(distance**2)
        else:
            gate, penalty = float(max_delay_error), min(n, m)+1.
            cost = np.zeros((n+m, n+m))
            cost[:n, :m] = 4*penalty*(n+m+1)
            valid = distance <= gate
            cost[:n, :m][valid] = (distance[valid]/gate)**2
            cost[:n, m:], cost[n:, :m] = penalty, penalty
            rows, cols = linear_sum_assignment(cost)
            take = (rows < n) & (cols < m)
            ri, ci = rows[take], cols[take]
            if np.any(distance[ri, ci] > gate):
                raise RuntimeError('invalid gated assignment')
    return PathAssignment(ri, ci, np.setdiff1d(np.arange(n), ri),
                          np.setdiff1d(np.arange(m), ci),
                          float(np.sum((td[ri]-ed[ci])**2)))


def match_paths(true_delays, true_amps, est_delays, est_amps):
    """Bijection in truth order; compatibility return (delays, amplitudes)."""
    td, ed = _vector(true_delays, 'true_delays'), _vector(est_delays, 'est_delays')
    ta, ea = _vector(true_amps, 'true_amps', True), _vector(est_amps, 'est_amps', True)
    if len(td) != len(ta) or len(ed) != len(ea):
        raise ValueError('delay/amplitude lengths differ')
    if len(td) != len(ed) or not len(td):
        raise ValueError('RMSE requires nonempty equal L; use detection_metrics for unequal L')
    assignment = assign_paths(td, ed)
    md, ma = np.empty_like(td), np.empty_like(ta)
    md[assignment.true_indices], ma[assignment.true_indices] = ed[assignment.estimate_indices], ea[assignment.estimate_indices]
    return md, ma


def _pair(a, b, complex_ok=False):
    a, b = _vector(a, 'truth', complex_ok), _vector(b, 'estimate', complex_ok)
    if a.shape != b.shape or not a.size:
        raise ValueError('truth and estimate require identical nonempty shapes')
    return a, b


def compute_rmse_delay(true_delays, matched_delays):
    a, b = _pair(true_delays, matched_delays)
    return float(np.sqrt(np.mean((b-a)**2)))


def compute_rmse_magnitude(true_amps, matched_amps):
    a, b = _pair(true_amps, matched_amps, True)
    return float(np.sqrt(np.mean((abs(b)-abs(a))**2)))


def phase_errors(true_amps, matched_amps):
    """Wrapped phase and validity; angle of zero amplitude is undefined."""
    a, b = _pair(true_amps, matched_amps, True)
    valid = (abs(a) > 0) & (abs(b) > 0)
    errors = np.full(a.shape, np.nan)
    delta = np.angle(b[valid])-np.angle(a[valid])
    errors[valid] = np.arctan2(np.sin(delta), np.cos(delta))
    return errors, valid


def compute_rmse_phase(true_amps, matched_amps):
    errors, valid = phase_errors(true_amps, matched_amps)
    return float(np.sqrt(np.mean(errors**2))) if valid.all() else float('nan')


def compute_rmse_complex_amplitude(true_amps, matched_amps):
    a, b = _pair(true_amps, matched_amps, True)
    return float(np.sqrt(np.mean(abs(b-a)**2)))


def compute_all_rmse(true_delays, true_amps, est_delays, est_amps):
    md, ma = match_paths(true_delays, true_amps, est_delays, est_amps)
    return (compute_rmse_delay(true_delays, md), compute_rmse_magnitude(true_amps, ma),
            compute_rmse_phase(true_amps, ma))


def error_statistics(errors, *, axis=0):
    """Trial-level bias/variance/MSE; never average per-trial RMSEs or omit NaNs.

Population variance: MSE = variance + |bias|². Nonfinite errors require explicit
failed/unavailable-trial reporting by the caller, not a silently filtered RMSE.
"""
    e = np.asarray(errors)
    if not e.size or not np.isfinite(e).all():
        raise ValueError('all errors must be finite and nonempty')
    bias = np.mean(e, axis=axis)
    variance = np.mean(abs(e-np.expand_dims(bias, axis))**2, axis=axis)
    mse = np.mean(abs(e)**2, axis=axis)
    return dict(bias=bias, variance=variance, mse=mse, rmse=np.sqrt(mse),
                overall_rmse=float(np.sqrt(np.mean(mse))))


def detection_metrics(true_delays, est_delays, *, max_delay_error=.1):
    """Gated TP/FN/FP, recall and false discovery proportion (not noise-cell FAR)."""
    result = assign_paths(true_delays, est_delays, max_delay_error=max_delay_error)
    tp, fn, fp = len(result.true_indices), len(result.unmatched_true), len(result.unmatched_estimates)
    return dict(true_positive=tp, missed=fn, false_positive=fp,
                detection_rate=tp/(tp+fn) if tp+fn else float('nan'),
                precision=tp/(tp+fp) if tp+fp else float('nan'),
                false_discovery_rate=fp/(tp+fp) if tp+fp else float('nan'), assignment=result)


def normalized_squared_error(reference, estimate):
    """Prediction NMSE: ||estimate-reference||² / ||reference||², linear units."""
    a, b = _pair(reference, estimate, True)
    energy = float(np.vdot(a, a).real)
    if energy <= 0:
        raise ValueError('NMSE reference energy must be positive')
    return float(np.vdot(b-a, b-a).real/energy)


def reconstruct_signal(est_delays, est_amps, s_base, N_rx, fractional_delay_shift_fn):
    delays, amps = _vector(est_delays, 'est_delays'), _vector(est_amps, 'est_amps', True)
    if delays.shape != amps.shape:
        raise ValueError('delay/amplitude shapes differ')
    signal = np.zeros(N_rx, dtype=complex)
    for delay, amp in zip(delays, amps):
        signal += amp*fractional_delay_shift_fn(s_base, delay, N_rx)
    return signal


def compute_residual_energy(y_rx, est_delays, est_amps, s_base, N_rx, fractional_delay_shift_fn):
    y_est = reconstruct_signal(est_delays, est_amps, s_base, N_rx, fractional_delay_shift_fn)
    a, b = _pair(y_rx, y_est, True)
    return float(np.vdot(a-b, a-b).real)
