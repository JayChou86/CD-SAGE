"""Local coupled CRB for the original finite-block fractional FFT template.

y ~ CN(mu(theta), N0 I), E|w[n]|²=N0; theta=[tau,Re(alpha),Im(alpha)].
F=(2/N0) Re(J.H J). Support changes make integer-delay points nonregular.
Singular/ill-conditioned cases never use an unqualified pseudoinverse.
"""
from dataclasses import dataclass
import numpy as np
from sage.signal_utils import fractional_delay_shift


@dataclass(frozen=True)
class CRBResult:
    delay_std: np.ndarray
    magnitude_std: np.ndarray
    phase_std: np.ndarray
    fim: object
    covariance_reim: object
    covariance_delay_magnitude_phase: object
    status: str
    rank: int
    condition_number: float
    scaled_condition_number: float
    eigenvalues: np.ndarray
    nonregular_paths: np.ndarray

    @property
    def valid(self):
        return self.status == 'ok'


def _inputs(s_tx, delays, amps, n_rx):
    s, td, ta = np.asarray(s_tx, complex), np.asarray(delays, float), np.asarray(amps, complex)
    if s.ndim != 1 or not s.size or not np.isfinite(s).all() or np.vdot(s, s).real <= 0:
        raise ValueError('waveform requires positive energy and finite vector samples')
    if td.ndim != 1 or ta.shape != td.shape or not td.size or not np.isfinite(td).all() or not np.isfinite(ta).all():
        raise ValueError('truth delays/amplitudes must be finite, nonempty, equal-length vectors')
    if not isinstance(n_rx, (int, np.integer)) or n_rx < len(s):
        raise ValueError('N_rx must be an integer at least waveform length')
    if np.any(td < 0) or np.any(td > n_rx-len(s)):
        raise ValueError('truth delays are outside the full-support domain')
    return s, td, ta


def nonregular_delay_indices(delays, *, tolerance=1e-10):
    td = np.asarray(delays, float)
    return np.flatnonzero(abs(td-np.round(td)) <= tolerance)


def mean_jacobian(s_tx, true_delays, true_amps, N_rx, *, method='analytic', step=1e-5):
    """Exact local mean derivative; finite differences never cross a support jump."""
    s, td, ta = _inputs(s_tx, true_delays, true_amps, N_rx)
    if nonregular_delay_indices(td).size:
        raise ValueError('template is not differentiable at integer delays')
    if method not in ('analytic', 'finite_difference') or not np.isfinite(step) or step <= 0:
        raise ValueError('invalid derivative method/step')
    L, ns = len(td), len(s)
    J = np.empty((N_rx, 3*L), complex)
    spectrum, omega = np.fft.fft(s), 2*np.pi*np.fft.fftfreq(ns)
    for i, (delay, amp) in enumerate(zip(td, ta)):
        template = fractional_delay_shift(s, delay, N_rx)
        if method == 'analytic':
            start = int(np.floor(delay))
            derivative = np.zeros(N_rx, complex)
            derivative[start:start+ns] = np.fft.ifft((-1j*omega)*spectrum*np.exp(-1j*omega*(delay-start)))
        else:
            fraction = delay-np.floor(delay)
            h = min(step, .25*min(fraction, 1-fraction))
            if delay+h == delay or delay-h == delay:
                raise ValueError('finite-difference step is below floating-point resolution')
            derivative = (fractional_delay_shift(s, delay+h, N_rx)-fractional_delay_shift(s, delay-h, N_rx))/(2*h)
        J[:, i], J[:, L+i], J[:, 2*L+i] = amp*derivative, template, 1j*template
    return J


def compute_joint_crb(s_tx, true_delays, true_amps, noise_var, N_rx, *, rcond=1e-12,
                      derivative='analytic', step=1e-5):
    """Joint unknown-delay/unknown-complex-amplitude CRB in samples/magnitude/rad.

Only status='ok' yields usable bounds. Other statuses' Inf arrays are sentinel
values for an unavailable full-parameter CRB, not a claim that all identifiable
linear combinations have infinite variance. Numerical rank uses the diagonally
normalized FIM; raw and scaled condition numbers are both exposed.
"""
    s, td, ta = _inputs(s_tx, true_delays, true_amps, N_rx)
    if not np.isfinite(noise_var) or noise_var <= 0 or not 0 < rcond < 1:
        raise ValueError('positive noise_var and rcond in (0,1) required')
    L, bad = len(td), nonregular_delay_indices(td)
    def unavailable(status, fim=None, rank=0, eigenvalues=None, raw_cond=np.inf, scaled_cond=np.inf):
        inf = np.full(L, np.inf)
        return CRBResult(inf.copy(), inf.copy(), inf.copy(), fim, None, None,
                         status, rank, raw_cond, scaled_cond,
                         np.array([]) if eigenvalues is None else eigenvalues, bad)
    if bad.size:
        return unavailable('nonregular_integer_delay')
    J = mean_jacobian(s, td, ta, N_rx, method=derivative, step=step)
    F = (2/float(noise_var))*np.real(J.conj().T@J)
    F = (F+F.T)/2
    eig, raw_cond = np.linalg.eigvalsh(F), float(np.linalg.cond(F))
    diagonal = np.diag(F)
    if np.any(diagonal <= 0):
        return unavailable('singular', F, int(np.linalg.matrix_rank(F)), eig, raw_cond)
    scale = np.sqrt(diagonal)
    normalized = F/scale[:, None]/scale[None, :]
    ev = np.linalg.eigvalsh(normalized)
    rank = int(np.sum(ev > rcond*ev[-1]))
    scaled_cond = float(ev[-1]/ev[0]) if ev[0] > 0 else np.inf
    if rank != 3*L:
        return unavailable('singular_or_ill_conditioned', F, rank, eig, raw_cond, scaled_cond)
    if scaled_cond >= 1/rcond:
        return unavailable('ill_conditioned', F, rank, eig, raw_cond, scaled_cond)
    C = np.linalg.solve(normalized, np.eye(3*L))/scale[:, None]/scale[None, :]
    C = (C+C.T)/2
    T = np.zeros((3*L, 3*L))
    T[:L, :L] = np.eye(L)
    for i, amp in enumerate(ta):
        magnitude = abs(amp)
        if magnitude == 0:
            return unavailable('zero_amplitude', F, rank, eig, raw_cond, scaled_cond)
        T[L+i, L+i], T[L+i, 2*L+i] = amp.real/magnitude, amp.imag/magnitude
        T[2*L+i, L+i], T[2*L+i, 2*L+i] = -amp.imag/magnitude**2, amp.real/magnitude**2
    transformed = T@C@T.T
    variances = np.diag(transformed)
    if np.any(variances < 0) or not np.isfinite(variances).all():
        return unavailable('numerical_failure', F, rank, eig, raw_cond, scaled_cond)
    std = np.sqrt(variances)
    return CRBResult(std[:L], std[L:2*L], std[2*L:], F, C, transformed, 'ok', rank,
                     raw_cond, scaled_cond, eig, bad)


def compute_conditional_crb(s_tx, true_amps, noise_var):
    """Scalar conditional references with all other parameters known.

Not the unknown-multipath joint CRB. Phase uses waveform energy, not derivative
energy or an additional amplitude factor. Legacy signal_utils.compute_crb gives
these same scalar formulas; new MC scripts must request the joint result.
"""
    s, a = np.asarray(s_tx, complex), np.asarray(true_amps, complex)
    if s.ndim != 1 or not s.size or a.ndim != 1 or not a.size or not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError('invalid waveform/amplitudes')
    energy = float(np.vdot(s, s).real)
    derivative_energy = float(np.sum(abs(np.fft.ifft(-1j*2*np.pi*np.fft.fftfreq(len(s))*np.fft.fft(s)))**2))
    if energy <= 0 or derivative_energy <= 0 or not np.isfinite(noise_var) or noise_var <= 0:
        raise ValueError('positive energy, derivative energy and noise variance required')
    with np.errstate(divide='ignore'):
        return (np.sqrt(noise_var/(2*abs(a)**2*derivative_energy)),
                np.full(a.shape, np.sqrt(noise_var/(2*energy))),
                np.sqrt(noise_var/(2*abs(a)**2*energy)))


def rms_bound(per_path_std, *, axis=-1):
    """sqrt(mean(per-path variance)); never average per-path standard deviations."""
    values = np.asarray(per_path_std, float)
    if not values.size or np.isnan(values).any() or (values < 0).any():
        raise ValueError('nonnegative non-NaN bounds required')
    return np.sqrt(np.mean(values**2, axis=axis))
