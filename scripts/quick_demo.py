"""Deterministic three-path CD-SAGE demonstration in sample units."""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage.cd_sage_default import CDModel, fit_cd_sage  # noqa: E402
from sage.signal_utils import generate_tx_signal  # noqa: E402


def main():
    pilot = generate_tx_signal(511, seed=40)
    model = CDModel(pilot, 1000)
    true_delays = np.array([50.43, 60.66, 200.25])
    true_gains = np.array([1.0, 0.5, 0.8]) * np.exp(1j * np.array([0.2, -1.5, 0.8]))
    clean = model.atoms(true_delays) @ true_gains

    rng = np.random.default_rng(20261028)
    snr_db = 20.0
    noise_var = np.var(clean) / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_var / 2) * (
        rng.normal(size=model.nr) + 1j * rng.normal(size=model.nr)
    )

    estimated_delays, estimated_gains, diagnostics = fit_cd_sage(
        model, clean + noise, path_count=3
    )
    rows, cols = linear_sum_assignment(
        (true_delays[:, None] - estimated_delays[None, :]) ** 2
    )
    ordered_delays = estimated_delays[cols]
    ordered_gains = estimated_gains[cols]
    delay_rmse = float(np.sqrt(np.mean((ordered_delays - true_delays[rows]) ** 2)))

    result = {
        "snr_db": snr_db,
        "true_delays_samples": true_delays.tolist(),
        "estimated_delays_samples": ordered_delays.tolist(),
        "estimated_gain_magnitudes": np.abs(ordered_gains).tolist(),
        "delay_rmse_samples": delay_rmse,
        "estimator_version": diagnostics["version"],
    }
    print(json.dumps(result, indent=2))
    if delay_rmse >= 0.1:
        raise RuntimeError("demo delay RMSE exceeded the 0.1-sample check")


if __name__ == "__main__":
    main()
