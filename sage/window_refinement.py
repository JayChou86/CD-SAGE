"""Shared, instrumented local profile search for the Stage-1 window study.

The templates exactly retain signal_utils.fractional_delay_shift's finite-block
periodic fractional shift. Caching and batched FFTs change implementation cost,
not the model. All study configurations use this same evaluation backend.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class WindowResult:
    tau: float
    amp: complex
    center: float
    lo: float
    hi: float
    nfev: int
    window_nfev: int
    objective: float
    old_objective: float
    window_tau: float
    used_old: bool
    success: bool
    elapsed_s: float


class DelayProfile:
    def __init__(self, signal, n_rx):
        self.signal = np.asarray(signal, dtype=complex)
        self.n_rx = int(n_rx)
        self.ns = len(self.signal)
        self.max_delay = float(self.n_rx - self.ns)
        if self.max_delay < 0:
            raise ValueError('n_rx must be at least the waveform length')
        self.spectrum = np.fft.fft(self.signal)
        self.corr_spectrum = np.conj(np.fft.fft(self.signal, n=self.n_rx))
        self.omega = 2 * np.pi * np.fft.fftfreq(self.ns)
        self.energy = float(np.vdot(self.signal, self.signal).real)
        if self.energy <= 0:
            raise ValueError('zero-energy waveform')
        self.indices = np.arange(self.ns)

    def projections(self, x, taus):
        """Return <s(tau),x>, without allocating zero-filled n_rx templates."""
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        if np.any(~np.isfinite(taus)) or np.any(taus < 0) or np.any(taus > self.max_delay):
            raise ValueError('delay outside declared legal range')
        starts = np.floor(taus).astype(int)
        fractions = taus - starts
        out = np.empty(len(taus), dtype=complex)
        # Parseval gives the same inner product with one FFT per distinct
        # support start, rather than one IFFT per candidate. No cross-observation
        # caching or cross-configuration reuse is performed.
        for start in np.unique(starts):
            which = np.flatnonzero(starts == start)
            coefficient = self.spectrum.conj() * np.fft.fft(x[start:start+self.ns]) / self.ns
            for a in range(0, len(which), 256):
                inds = which[a:a+256]
                phase = np.exp(1j * fractions[inds,None] * self.omega)
                out[inds] = np.sum(phase * coefficient, axis=1)
        return out

    def objective(self, x, taus):
        return np.abs(self.projections(x, taus)) ** 2 / self.energy

    def template(self, tau):
        if not 0 <= tau <= self.max_delay:
            raise ValueError('delay outside declared legal range')
        start = int(np.floor(tau))
        local = np.fft.ifft(self.spectrum * np.exp(-1j * self.omega * (tau - start)))
        result = np.zeros(self.n_rx, dtype=complex)
        result[start:start + self.ns] = local
        return result

    def integer_correlation(self, x):
        return np.abs(np.fft.ifft(np.fft.fft(x) *
                      self.corr_spectrum))[:int(self.max_delay) + 1]


def window_grid(center, radius, max_delay, step=0.005, points=None, phase=0.0):
    if not np.isfinite(radius) or radius <= 0 or step <= 0:
        raise ValueError('positive finite radius and step required')
    lo, hi = max(0., center - radius), min(max_delay, center + radius)
    if lo > hi:
        raise ValueError('empty search interval')
    if points is not None:
        if int(points) != points or points < 3:
            raise ValueError('points must be an integer >=3')
        grid = np.linspace(center - radius, center + radius, int(points))
    else:
        intervals = int(round(2 * radius / step))
        if not np.isclose(intervals * step, 2 * radius, atol=1e-10):
            raise ValueError('2*radius must be an integer multiple of step')
        grid = center - radius + np.arange(intervals + 1) * step + phase * step
    grid = grid[(grid >= lo - 1e-10) & (grid <= hi + 1e-10)]
    grid = np.concatenate((np.clip(grid, lo, hi), [lo, hi]))
    return np.unique(np.round(grid, 12)), lo, hi


def refine_window(profile, x, center, radius, *, solver='grid', step=0.005,
                  points=None, old_tau=None, xatol=1e-7, maxiter=100,
                  grid_phase=0.0):
    """Pure window search, optionally safeguarded by the previous delay.

The bounded solver checks endpoints and splits at integers because the repo's
finite-block template can change support there. This is a local solver, not a
claim of global optimization. nfev includes the old-point safeguard.
"""
    t0 = perf_counter()
    lo, hi = max(0., center - radius), min(profile.max_delay, center + radius)
    if radius <= 0 or lo > hi:
        raise ValueError('invalid window')
    success = True
    if solver == 'grid':
        taus, lo, hi = window_grid(center, radius, profile.max_delay, step, points, grid_phase)
        values = profile.objective(x, taus)
        nfev = len(taus)
    elif solver == 'bounded':
        cuts = np.unique(np.r_[lo, np.arange(np.ceil(lo), np.floor(hi) + 1), hi])
        taus = list(cuts)
        values = list(profile.objective(x, cuts))
        nfev = len(cuts)
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 2 * xatol:
                continue
            res = minimize_scalar(lambda tau: -float(profile.objective(x, [tau])[0]),
                                  bounds=(a, b), method='bounded',
                                  options={'xatol': xatol, 'maxiter': maxiter})
            nfev += res.nfev
            success = success and bool(res.success)
            taus.append(float(res.x))
            values.append(float(-res.fun))
        taus, values = np.asarray(taus), np.asarray(values)
    else:
        raise ValueError(f'unknown solver: {solver}')
    best = int(np.argmax(values))
    tau, objective = float(taus[best]), float(values[best])
    window_tau = tau
    window_nfev = nfev
    old_objective, used_old = np.nan, False
    if old_tau is not None:
        old_objective = float(profile.objective(x, [old_tau])[0])
        nfev += 1
        if old_objective > objective:
            tau, objective, used_old = float(old_tau), old_objective, True
    amp = profile.projections(x, [tau])[0] / profile.energy
    return WindowResult(tau, amp, float(center), float(lo), float(hi), nfev,
                        window_nfev, objective, old_objective, window_tau,
                        used_old, success, perf_counter() - t0)
