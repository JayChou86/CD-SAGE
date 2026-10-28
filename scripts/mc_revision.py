"""Shared MC contract: paired observations, active budgets, joint CRB, raw trials.

Historic estimator choices remain explicit; repairing evaluation does not freeze
the manuscript's final algorithm. All outputs require a new run directory.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sage.signal_utils import generate_tx_signal, fractional_delay_shift, build_dictionary
from sage.algorithms import algo_SAGE, algo_MF, algo_OMP, algo_NOMP, algo_SBL, algo_SAGE_baseline
from sage.evaluation import match_paths, phase_errors, detection_metrics, reconstruct_signal, normalized_squared_error
from sage.bounds import compute_joint_crb, compute_conditional_crb, rms_bound

BASE = ['IntSAGE', 'CD-SAGE', 'MF', 'OMP', 'NOMP', 'SBL']
GLOBAL = ['GlobalGrid(0.02)', 'GlobalGrid(0.005)']
EXTRA = ['SAGE-bounded', 'Stage1-grid(.3)']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_observations(clean, noise_var, seed, snr, count):
    """Common trial prefix at a given SNR, independent of budgets/list ordering."""
    bits = int(np.array(float(snr), dtype='<f8').view('<u8')) if snr != 0 else 0
    observations = []
    for trial in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, bits & 0xffffffff, bits >> 32, trial]))
        observations.append(clean+np.sqrt(noise_var/2)*(rng.normal(size=len(clean))+1j*rng.normal(size=len(clean))))
    return observations


def collect_algorithm_trials(observations, runners, budgets, true_delays, true_amps, clean, reconstruct):
    """Execute each algorithm exactly its budget; failures remain recorded.

Public helper supports a regression with cheap estimators. Unrun entries are
NaN with attempted=False; an attempted failure has infinite delay/mag error.
"""
    n, A, L = len(observations), len(runners), len(true_delays)
    if len(budgets) != A or any(not isinstance(b, (int, np.integer)) or b <= 0 or b > n for b in budgets):
        raise ValueError('positive integer budgets up to number of observations required')
    shape = (A, n, L)
    raw = {k: np.full(shape, np.nan) for k in ['delay_error', 'magnitude_error', 'phase_error', 'tau_hat']}
    raw['amplitude_error'] = np.full(shape, np.nan+1j*np.nan)
    raw['alpha_hat'] = np.full(shape, np.nan+1j*np.nan)
    for k in ['total_s', 'prediction_nmse', 'true_positive', 'missed', 'false_positive']:
        raw[k] = np.full((A, n), np.nan)
    raw['attempted'], raw['success'] = np.zeros((A, n), bool), np.zeros((A, n), bool)
    raw['phase_valid'] = np.zeros(shape, bool)
    failures = []
    for trial, observation in enumerate(observations):
        for ai in np.roll(np.arange(A), trial % A):
            if trial >= budgets[ai]:
                continue
            raw['attempted'][ai, trial] = True
            start = perf_counter()
            try:
                de, ae = runners[ai](observation)
                md, ma = match_paths(true_delays, true_amps, de, ae)
                pe, valid = phase_errors(true_amps, ma)
                predicted = reconstruct(md, ma)
                detection = detection_metrics(true_delays, de, max_delay_error=.1)
                raw['tau_hat'][ai, trial], raw['alpha_hat'][ai, trial] = md, ma
                raw['delay_error'][ai, trial] = md-true_delays
                raw['magnitude_error'][ai, trial] = abs(ma)-abs(true_amps)
                raw['amplitude_error'][ai, trial] = ma-true_amps
                raw['phase_error'][ai, trial], raw['phase_valid'][ai, trial] = pe, valid
                raw['prediction_nmse'][ai, trial] = normalized_squared_error(clean, predicted)
                for key in ['true_positive', 'missed', 'false_positive']:
                    raw[key][ai, trial] = detection[key]
                raw['success'][ai, trial] = True
            except Exception as exc:
                # No filtering: complete-estimator failure gives infinite loss.
                raw['delay_error'][ai, trial] = np.inf
                raw['magnitude_error'][ai, trial] = np.inf
                raw['amplitude_error'][ai, trial] = complex(np.inf, 0)
                raw['prediction_nmse'][ai, trial] = np.inf
                raw['true_positive'][ai, trial], raw['missed'][ai, trial] = 0, L
                failures.append(dict(algorithm_index=int(ai), trial=int(trial), error=repr(exc)))
            raw['total_s'][ai, trial] = perf_counter()-start
    return raw, failures


def summarize_trials(raw):
    """Pool only attempted trials; never silently drop attempted failures."""
    A, _, L = raw['delay_error'].shape
    summary = {}
    for source, suffix in [('delay_error', 'delay'), ('magnitude_error', 'mag'),
                           ('phase_error', 'phase'), ('amplitude_error', 'complex_amplitude')]:
        per, overall, biases, variance = [], [], [], []
        for ai in range(A):
            e = raw[source][ai, raw['attempted'][ai]]
            mse = np.mean(abs(e)**2, axis=0)
            per.append(np.sqrt(mse))
            overall.append(np.sqrt(np.mean(mse)))
            if np.isfinite(e).all():
                bias = e.mean(0)
                biases.append(bias)
                variance.append(np.mean(abs(e-bias)**2, axis=0))
            else:
                biases.append(np.full(L, np.nan))
                variance.append(np.full(L, np.nan))
        summary['rmse_'+suffix] = np.asarray(overall)
        summary['per_path_'+suffix] = np.asarray(per)
        summary['bias_'+suffix], summary['variance_'+suffix] = np.asarray(biases), np.asarray(variance)
    summary['attempted_counts'] = raw['attempted'].sum(1)
    summary['successful_counts'] = raw['success'].sum(1)
    summary['phase_defined_counts'] = (raw['phase_valid'] & raw['attempted'][:, :, None]).sum((1, 2))
    summary['failure_rate'] = np.array([np.mean(abs(raw['delay_error'][i, raw['attempted'][i]]) > .1) for i in range(A)])
    summary['severe_rate'] = np.array([np.mean(abs(raw['delay_error'][i, raw['attempted'][i]]) > .5) for i in range(A)])
    for source, dest in [('total_s', 'mean_total_s'), ('prediction_nmse', 'mean_prediction_nmse'),
                         ('missed', 'mean_missed'), ('false_positive', 'mean_false_positive')]:
        summary[dest] = np.array([raw[source][i, raw['attempted'][i]].mean() for i in range(A)])
    summary['detection_rate'] = np.array([raw['true_positive'][i, raw['attempted'][i]].sum()/(L*summary['attempted_counts'][i]) for i in range(A)])
    return summary


def plot_results(data, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                         'mathtext.fontset': 'stix', 'font.size': 9, 'pdf.fonttype': 42}):
        for path in [None]+list(range(len(data['true_delays']))):
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout='constrained')
            for j, (ax, suffix, unit) in enumerate(zip(axes, ['delay', 'mag', 'phase'], ['samples', 'magnitude', 'rad'])):
                values = data['rmse_'+suffix] if path is None else data['per_path_'+suffix][:, :, path]
                for ai, name in enumerate(data['algo_names']):
                    ax.semilogy(data['SNR_dB_vec'], values[:, ai], marker='o', ms=3, label=str(name))
                joint = rms_bound(data['crb_'+suffix], axis=-1) if path is None else data['crb_'+suffix][:, path]
                conditional = rms_bound(data['crb_'+suffix+'_conditional'], axis=-1) if path is None else data['crb_'+suffix+'_conditional'][:, path]
                ax.semilogy(data['SNR_dB_vec'], joint, 'k-', lw=1.6, label='Joint unknown-path CRB')
                ax.semilogy(data['SNR_dB_vec'], conditional, 'k--', lw=1, label='Conditional reference')
                ax.set(xlabel='Total SNR (dB)', ylabel=f'RMSE ({unit})', title=f'({chr(97+j)})')
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=6)
            fig.suptitle('Overall' if path is None else f'Path {path+1}', fontsize=10)
            stem = 'fig_rmse_evaluation_corrected' if path is None else f'fig_rmse_path{path+1}_corrected'
            for fmt in ['png', 'pdf']:
                fig.savefig(out/f'{stem}.{fmt}', dpi=300)
            plt.close(fig)


def main(default_profile='main'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--mc', type=int)
    p.add_argument('--snr', default='-15,-10,-5,0,5,10,15')
    p.add_argument('--algorithms')
    p.add_argument('--budgets', help='comma-separated name=count overrides')
    p.add_argument('--grid', type=float, default=.1 if default_profile=='grid01' else .5)
    p.add_argument('--seed', type=int, default=40)
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()
    if args.mc is not None and args.mc <= 0:
        p.error('--mc must be a positive integer')
    if args.seed < 0:
        p.error('--seed must be nonnegative')
    names = args.algorithms.split(',') if args.algorithms else BASE+(GLOBAL if default_profile=='globalgrid' else [])
    if len(set(names)) != len(names) or any(name not in BASE+GLOBAL+EXTRA for name in names):
        p.error('unknown or duplicate algorithm name')
    counts = {name: args.mc or (200 if default_profile in ['globalgrid','grid01'] else 500) for name in names}
    if args.mc is None and default_profile=='globalgrid':
        counts.update({'CD-SAGE': 100, 'SBL': 50, 'GlobalGrid(0.02)': 15, 'GlobalGrid(0.005)': 10})
    if args.budgets:
        for item in args.budgets.split(','):
            name, value = item.rsplit('=', 1)
            if name not in names:
                p.error('budget override for inactive algorithm')
            counts[name] = int(value)
    budgets = [counts[name] for name in names]
    snrs = np.array([float(v) for v in args.snr.split(',')])
    if not names or min(budgets) <= 0 or not np.isfinite(snrs).all() or not snrs.size or not np.isfinite(args.grid) or args.grid <= 0:
        p.error('positive counts/grid and finite nonempty SNRs required')
    if len(np.unique(snrs)) != len(snrs):
        p.error('duplicate SNR values would overwrite raw trial files')
    out = (args.out or ROOT/'results/revision/evaluation_crb'/f'{default_profile}_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=2, status='running', algorithms=names, counts=budgets,
                    profile=default_profile, seed=args.seed, snr=snrs.tolist(), grid=args.grid,
                    matching='bijection minimizing summed squared delay error',
                    crb='joint local unknown-delay/unknown-complex-amplitude; conditional references separately labeled',
                    snr_definition='var(clean)/E|w[n]|^2', threads=1,
                    noise_seed='SeedSequence(seed, 2, IEEE float64 SNR low/high words, trial); independent of SNR-list order',
                    units=dict(delay='samples', phase='radians', time='seconds'),
                    historical_algorithms='IntSAGE/CD-SAGE use original algo_SAGE entry; global-grid uses baseline collision_on=True; extra research modes have collision_on=False',
                    missing_policy='unrun entries have attempted=False; attempted failures retained as infinite delay/magnitude loss; undefined phase propagates NaN',
                    hashes={f: digest(ROOT/f) for f in ['sage/evaluation.py','sage/bounds.py','sage/signal_utils.py','sage/algorithms.py','scripts/mc_revision.py']})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    s, n_rx = generate_tx_signal(511, args.seed), 1000
    td = np.array([50.43, 60.66, 200.25])
    ta = np.array([1., .5, .8])*np.exp(1j*np.array([.2, -1.5, .8]))
    reconstruct = lambda d, a: reconstruct_signal(d, a, s, n_rx, fractional_delay_shift)
    clean = reconstruct(td, ta)
    tau_grid = np.arange(30, 220+args.grid/2, args.grid)
    dictionary = build_dictionary(s, tau_grid, n_rx) if any(n in names for n in ['MF','OMP','SBL']) else None
    all_data, failures, bounds, conditional, bound_diagnostics = [], [], [], [], []
    with threadpool_limits(limits=1):
        for si, snr in enumerate(snrs):
            noise_var = float(np.var(clean)/10**(snr/10))
            b = compute_joint_crb(s, td, ta, noise_var, n_rx)
            if not b.valid:
                raise RuntimeError('joint bound unavailable: '+b.status)
            bounds.append([b.delay_std, b.magnitude_std, b.phase_std])
            conditional.append(compute_conditional_crb(s, ta, noise_var))
            bound_diagnostics.append(dict(status=b.status, rank=b.rank, condition=b.condition_number,
                                          scaled_condition=b.scaled_condition_number, noise_var=noise_var))
            observations = noise_observations(clean, noise_var, args.seed, snr, max(budgets))
            runner_map = {
                'IntSAGE': lambda y: algo_SAGE(y, s, n_rx, 3, do_frac=False),
                'CD-SAGE': lambda y: algo_SAGE(y, s, n_rx, 3, do_frac=True),
                'MF': lambda y: algo_MF(y, dictionary, tau_grid, 3),
                'OMP': lambda y: algo_OMP(y, dictionary, tau_grid, 3),
                'NOMP': lambda y: algo_NOMP(y, s, n_rx, 3, tau_grid),
                'SBL': lambda y: algo_SBL(y, dictionary, tau_grid, 3, use_woodbury=True),
                'GlobalGrid(0.02)': lambda y: algo_SAGE_baseline(y, s, n_rx, 3, mstep='global_grid', gg_step=.02),
                'GlobalGrid(0.005)': lambda y: algo_SAGE_baseline(y, s, n_rx, 3, mstep='global_grid', gg_step=.005),
                'SAGE-bounded': lambda y: algo_SAGE_baseline(y, s, n_rx, 3, mstep='window_bounded', collision_on=False),
                'Stage1-grid(.3)': lambda y: algo_SAGE_baseline(y, s, n_rx, 3, mstep='window_grid', collision_on=False, refine_center='stage1', refine_radius=.3)}
            raw, problems = collect_algorithm_trials(observations, [runner_map[name] for name in names], budgets, td, ta, clean, reconstruct)
            summary = summarize_trials(raw)
            all_data.append(summary)
            failures.extend([dict(snr_index=si, **problem) for problem in problems])
            np.savez_compressed(out/f'trials_snr{snr:g}.npz', **raw, y_rx=np.asarray(observations),
                                waveform=s, true_delays=td, true_amps=ta, noise_var=noise_var,
                                trial_id=np.arange(len(observations)), algorithms=np.asarray(names))
            print(f'SNR={snr:g}: counts={summary["attempted_counts"].tolist()}, failures={len(problems)}', flush=True)
    data = {k: np.asarray([d[k] for d in all_data]) for k in all_data[0]}
    bounds, conditional = np.asarray(bounds), np.asarray(conditional)
    for j, suffix in enumerate(['delay', 'mag', 'phase']):
        data['crb_'+suffix], data['crb_'+suffix+'_conditional'] = bounds[:, j], conditional[:, j]
    data.update(SNR_dB_vec=snrs, algo_names=np.asarray(names), true_delays=td, true_amps=ta,
                schema_version=2, mc_counts=np.asarray(budgets), grid_res=args.grid, seed=args.seed)
    np.savez_compressed(out/'mc_results.npz', **data)
    if not args.no_plot:
        plot_results(data, out)
    manifest.update(status='complete', crb_diagnostics=bound_diagnostics, failures=failures,
                    attempted_counts=data['attempted_counts'].tolist())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Saved '+str(out), flush=True)


if __name__ == '__main__':
    main()
