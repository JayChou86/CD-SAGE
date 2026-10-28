"""
SAGE Super-Resolution: 水声信道多径时延超分辨率估计

基于 SAGE (Space-Alternating Generalized Expectation-Maximization) 框架，
在 M-step 后使用二次插值突破整数采样网格精度限制。
"""

from sage.signal_utils import (
    fractional_delay_shift,
    generate_tx_signal,
    generate_rx_signal,
    compute_crb,
)
from sage.algorithms import (
    algo_SAGE,
    algo_MF,
    algo_OMP,
    algo_NOMP,
    algo_ROOTMUSIC,
    algo_SBL,
)
from sage.interpolation import (
    parabolic_interp,
    parabolic_delta,
)
from sage.evaluation import (
    match_paths,
    compute_rmse_delay,
    compute_rmse_magnitude,
    compute_rmse_phase,
    compute_all_rmse,
    assign_paths,
    detection_metrics,
    error_statistics,
    normalized_squared_error,
    compute_rmse_complex_amplitude,
)
from sage.bounds import compute_joint_crb, compute_conditional_crb, rms_bound
from sage.cd_sage_default import (
    CDConfig,
    CDModel,
    DEFAULT_CONFIG,
    estimate_cd_sage,
    fit_cd_sage,
)
