"""Frozen CD-SAGE Coarse v1, selected from the revision evidence.

All delays are in samples. The finite-block atom uses a periodic fractional
shift of the pilot, inserted at floor(delay), without truncating its support.
Historical algo_SAGE and earlier experimental entry points remain unchanged.
"""
from collections import Counter
from dataclasses import dataclass, asdict
import numpy as np

VERSION = 'CD-SAGE-Coarse-v1-20261012'


@dataclass(frozen=True)
class CDConfig:
    sweeps: int = 15
    integer_halfwidth: int = 10
    radius: float = .5
    coarse_step: float = .05
    fine_step: float = .005
    stage1: bool = True
    denominator_threshold: float = 1e-15
    ls_rcond: float = 1e-12


DEFAULT_CONFIG = CDConfig()


class CDModel:
    """Waveform-dependent reusable state; no observations cached here."""
    def __init__(self, s, nr):
        self.s = np.asarray(s, complex).copy()
        if self.s.ndim != 1 or not len(self.s) or not np.isfinite(self.s).all():
            raise ValueError('pilot must be a finite nonempty one-dimensional array')
        if isinstance(nr, (bool, np.bool_)) or int(nr) != nr or nr < len(self.s):
            raise ValueError('receive length must be an integer at least the pilot length')
        self.ns = len(self.s); self.nr = int(nr)
        self.lo = 0.; self.hi = float(self.nr-self.ns)
        self.sf = np.fft.fft(self.s); self.w = 2*np.pi*np.fft.fftfreq(self.ns)
        self.es = float(np.vdot(self.s, self.s).real)
        if self.es <= 0: raise ValueError('pilot energy must be positive')
        self.cs = np.conj(np.fft.fft(self.s, n=self.nr))
        self.phase_cache = {}

    def atoms(self, ts):
        ts = np.atleast_1d(ts).astype(float)
        if not np.isfinite(ts).all() or np.any(ts < self.lo) or np.any(ts > self.hi):
            raise ValueError('delay outside full-support domain')
        out = np.zeros((self.nr, len(ts)), complex)
        for j, t in enumerate(ts):
            k = int(np.floor(t)); f = t-k
            out[k:k+self.ns, j] = np.fft.ifft(self.sf*np.exp(-1j*self.w*f))
        return out

    def coeff(self, x, k):
        return self.sf.conj()*np.fft.fft(x[k:k+self.ns])/self.ns

    def integers(self, x):
        return np.abs(np.fft.ifft(np.fft.fft(x)*self.cs))

    def ls(self, y, ts, rcond=1e-12):
        atoms = self.atoms(ts)
        amps = np.linalg.lstsq(atoms, y, rcond=rcond)[0]
        return amps, atoms, y-atoms@amps


class _Objective:
    def __init__(self, model, x, counts):
        self.m = model; self.x = x; self.counts = counts; self.coefficients = {}

    def values(self, ts, phase=None, origin=None):
        ts = np.atleast_1d(ts); self.counts['nfev'] += len(ts)
        floors = np.floor(ts).astype(int); out = np.empty(len(ts))
        for k in np.unique(floors):
            ix = np.flatnonzero(floors == k)
            if k not in self.coefficients: self.coefficients[k] = self.m.coeff(self.x, int(k))
            co = self.coefficients[k]
            if phase is None: p = np.exp(1j*(ts[ix]-k)[:, None]*self.m.w)
            else: p = phase[ix]; co = co*np.exp(1j*(origin-k)*self.m.w)
            out[ix] = abs(p@co)**2/self.m.es
        return out


def _local(m, x, center, previous, counts, cfg):
    o = _Objective(m, x, counts)
    lo = max(m.lo, center-cfg.radius); hi = min(m.hi, center+cfg.radius)
    cuts = np.unique(np.r_[lo, np.arange(np.ceil(lo), np.floor(hi)+1), hi])
    candidates = list(cuts)+[previous]
    candidates.extend(np.nextafter(v, -np.inf) for v in cuts if v > lo and v == int(v))

    def scan(origin, delta, end):
        ts = origin+np.arange(int(round((end-origin)/delta))+1)*delta
        ts = ts[np.flatnonzero((ts >= lo)&(ts <= hi))]
        key = (len(ts), float(delta)); phase = m.phase_cache.get(key)
        if phase is None:
            phase = np.exp(1j*np.arange(len(ts))[:, None]*delta*m.w)
            m.phase_cache[key] = phase
        if not len(ts): return None
        return float(ts[np.argmax(o.values(ts, phase, ts[0]))])

    best = scan(center-cfg.radius, cfg.coarse_step, center+cfg.radius)
    if best is not None:
        fine = scan(best-cfg.coarse_step, cfg.fine_step, best+cfg.coarse_step)
        if fine is not None: candidates.append(fine)
    values = o.values(candidates); chosen = int(np.argmax(values))
    old_index = len(cuts); counts['old_kept'] += int(chosen == old_index)
    counts['msteps'] += 1
    return float(candidates[chosen]), dict(center=center, lo=lo, hi=hi,
        previous=previous, selected=float(candidates[chosen]), selected_power=float(values[chosen]),
        previous_power=float(values[old_index]), old_kept=chosen == old_index,
        previous_outside=previous < lo or previous > hi, integer_cuts=len(cuts))


def _validate_config(cfg):
    if not isinstance(cfg, CDConfig): raise TypeError('config must be CDConfig')
    if isinstance(cfg.sweeps, bool) or not isinstance(cfg.sweeps, int) or cfg.sweeps < 1:
        raise ValueError('sweeps must be a positive integer')
    if isinstance(cfg.integer_halfwidth, bool) or not isinstance(cfg.integer_halfwidth, int) or cfg.integer_halfwidth < 0:
        raise ValueError('integer halfwidth must be a nonnegative integer')
    for value in (cfg.radius, cfg.coarse_step, cfg.fine_step, cfg.denominator_threshold, cfg.ls_rcond):
        if not np.isfinite(value) or value <= 0: raise ValueError('positive finite search/LS parameters required')
    if cfg.radius > .5 or cfg.coarse_step > 2*cfg.radius or cfg.fine_step > cfg.coarse_step:
        raise ValueError('invalid continuous search scales')
    for ratio in (2*cfg.radius/cfg.coarse_step, 2*cfg.coarse_step/cfg.fine_step):
        if abs(ratio-round(ratio)) > 1e-10: raise ValueError('scan interval must be divisible by step')
    if not isinstance(cfg.stage1, (bool, np.bool_)): raise ValueError('stage1 must be boolean')


def fit_cd_sage(model, y, path_count, config=DEFAULT_CONFIG, *, trace=False, capture=False):
    """Known-order fixed-sweep estimator; return delays, complex gains, diagnostics.

    No truth, collision relocation or early stopping is used. The optional
    trace records numerical update checks without changing decisions. A
    non-default config is an explicit study variant, not the frozen default.
    """
    _validate_config(config)
    if not isinstance(model, CDModel): raise TypeError('prepare a CDModel for this waveform')
    y = np.asarray(y, complex)
    if y.shape != (model.nr,) or not np.isfinite(y).all(): raise ValueError('finite receive vector of declared length required')
    if isinstance(path_count, (bool, np.bool_)) or int(path_count) != path_count or not 0 <= path_count <= model.nr:
        raise ValueError('path count must be a nonnegative integer no greater than receive length')
    L = int(path_count); ts = np.zeros(L); amps = np.zeros(L, complex)
    atoms = np.zeros((model.nr, L), complex); res = y.copy(); counts = Counter()
    if L == 0:
        return ts, amps, dict(version=VERSION,config=asdict(config),counts={},residual_energy=[float(np.vdot(y,y).real)],updates=[],snapshots=[])
    energy = []; updates = []; snapshots = []; ls_records = []
    for it in range(config.sweeps):
        for j in range(L):
            before = float(np.vdot(res, res).real) if trace else None
            x = res+atoms[:, j]*amps[j] if it else res.copy(); corr = model.integers(x)
            if it:
                lo = max(0, int(np.rint(ts[j]))-config.integer_halfwidth)
                hi = min(int(model.hi), int(np.rint(ts[j]))+config.integer_halfwidth)
            else: lo, hi = 0, int(model.hi)
            k = lo+int(np.argmax(corr[lo:hi+1])); center = float(k)
            if config.stage1 and 0 < k < model.hi:
                p = corr[k-1:k+2]**2; den = p[0]-2*p[1]+p[2]
                if den < -config.denominator_threshold:
                    center += float(np.clip(.5*(p[0]-p[2])/den, -.5, .5))
            previous = float(ts[j]) if it else float(k)
            t, record = _local(model, x, center, previous, counts, config)
            if capture and (it, j) in ((0,0), (0,L-1), (config.sweeps-1,0)):
                snapshots.append(dict(sweep=it,path_slot=j,x=x.copy(),integer_peak=k,**record))
            ts[j] = t; atoms[:, j] = model.atoms([t])[:, 0]
            amps[j] = np.vdot(atoms[:, j], x)/model.es; res = x-atoms[:, j]*amps[j]
            if trace:
                record.update(sweep=it,path_slot=j,integer_peak=k,RSS_before=before,RSS_after=float(np.vdot(res,res).real))
                updates.append(record)
        ls_before = float(np.vdot(res,res).real) if trace else None
        amps, atoms, res = model.ls(y, ts, config.ls_rcond)
        energy.append(float(np.vdot(res,res).real))
        if trace:
            ls_records.append(dict(sweep=it,RSS_before=ls_before,RSS_after=energy[-1]))
    return ts, amps, dict(version=VERSION, config=asdict(config), counts=dict(counts),
                         residual_energy=energy, updates=updates, joint_ls=ls_records, snapshots=snapshots)


def estimate_cd_sage(y, pilot, path_count, *, config=DEFAULT_CONFIG, trace=False):
    """Convenience entry including waveform preparation in the call."""
    return fit_cd_sage(CDModel(pilot, len(y)), y, path_count, config, trace=trace)
