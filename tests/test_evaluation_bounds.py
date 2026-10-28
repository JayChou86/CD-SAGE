"""Regression counterexamples, statistical identities and independent CRB checks."""
import unittest
import numpy as np
from sage.evaluation import (assign_paths, match_paths, detection_metrics,
                             compute_rmse_phase, error_statistics, normalized_squared_error)
from sage.bounds import compute_joint_crb, compute_conditional_crb, mean_jacobian, rms_bound
from sage.signal_utils import generate_tx_signal


class EvaluationTests(unittest.TestCase):
    def test_duplicate_nearest_neighbor_is_not_reused(self):
        td, ed = np.array([0., 1., 10.]), np.array([0., 10., 20.])
        md, ma = match_paths(td, np.ones(3), ed, np.array([1, 2, 3], complex))
        self.assertEqual(len(np.unique(md)), 3)
        self.assertAlmostEqual(np.sqrt(np.mean((md-td)**2)), np.sqrt(181/3))
        np.testing.assert_array_equal(ma, [1, 2, 3])

    def test_gated_optimum_does_not_reject_ungated_tie(self):
        result = detection_metrics([0., 1.], [1., 2.], max_delay_error=.1)
        self.assertEqual((result['true_positive'], result['missed'], result['false_positive']), (1, 1, 1))
        np.testing.assert_array_equal(result['assignment'].true_indices, [1])

    def test_unequal_empty_and_duplicate_estimates_remain_explicit(self):
        a = assign_paths([1., 2.], [1., 1., 5.], max_delay_error=.1)
        self.assertEqual(len(a.true_indices), 1)
        self.assertEqual(len(a.unmatched_estimates), 2)
        self.assertEqual(detection_metrics([1., 2.], [])['missed'], 2)
        self.assertEqual(detection_metrics([], [1.])['false_positive'], 1)
        with self.assertRaises(ValueError):
            match_paths([1., 2.], [1, 1], [1.], [1])

    def test_phase_wrap_and_zero_phase_is_undefined(self):
        self.assertAlmostEqual(compute_rmse_phase([np.exp(1j*(np.pi-.01))],
                                                 [np.exp(1j*(-np.pi+.01))]), .02)
        self.assertTrue(np.isnan(compute_rmse_phase([1.], [0.])))

    def test_trial_pooled_mse_not_mean_of_trial_rmse(self):
        e = np.array([[0., 1.], [4., 3.]])
        st = error_statistics(e)
        np.testing.assert_allclose(st['mse'], st['variance']+st['bias']**2)
        self.assertAlmostEqual(st['overall_rmse'], np.sqrt(6.5))
        self.assertNotAlmostEqual(st['overall_rmse'], np.sqrt((e**2).mean(1)).mean())
        with self.assertRaises(ValueError):
            error_statistics([1., np.nan])

    def test_prediction_nmse_has_declared_reference(self):
        self.assertAlmostEqual(normalized_squared_error([1, 1j], [0, 0]), 1)
        with self.assertRaises(ValueError):
            normalized_squared_error([0], [0])


class BoundsTests(unittest.TestCase):
    def setUp(self):
        self.s = generate_tx_signal(63, 40)
        self.td = np.array([10.43, 20.66, 100.25])
        self.ta = np.array([1., .5, .8])*np.exp(1j*np.array([.2, -1.5, .8]))

    def test_analytic_derivative_vs_steps_and_both_parities(self):
        for ns in (62, 63):
            s = generate_tx_signal(ns, 40)
            exact = mean_jacobian(s, self.td, self.ta, 200)
            for h in (1e-4, 1e-5, 1e-6):
                numeric = mean_jacobian(s, self.td, self.ta, 200, method='finite_difference', step=h)
                np.testing.assert_allclose(exact, numeric, rtol=2e-6, atol=2e-8)

    def test_single_path_closed_form_with_unknown_phase(self):
        amp, noise = .7*np.exp(.3j), .02
        r = compute_joint_crb(self.s, [10.43], [amp], noise, 200)
        p = abs(np.fft.fft(self.s))**2/len(self.s)
        omega = 2*np.pi*np.fft.fftfreq(len(self.s))
        e, m1, m2 = p.sum(), (omega*p).sum(), (omega**2*p).sum()
        expected = noise/(2*abs(amp)**2*(m2-m1**2/e))
        self.assertTrue(r.valid)
        self.assertAlmostEqual(r.delay_std[0]**2, expected, places=12)
        self.assertAlmostEqual(r.magnitude_std[0]**2, noise/(2*e), places=12)
        self.assertAlmostEqual(r.phase_std[0]**2, noise/(2*abs(amp)**2*(e-m1**2/m2)), places=12)

    def test_joint_fim_inverse_and_numeric_crb(self):
        r = compute_joint_crb(self.s, self.td, self.ta, .02, 200)
        n = compute_joint_crb(self.s, self.td, self.ta, .02, 200, derivative='finite_difference')
        self.assertTrue(r.valid)
        np.testing.assert_allclose(r.fim, r.fim.T, atol=1e-12)
        self.assertGreater(r.eigenvalues.min(), 0)
        np.testing.assert_allclose(r.fim@r.covariance_reim, np.eye(9), atol=2e-13)
        np.testing.assert_allclose(r.delay_std, n.delay_std, rtol=2e-7)
        np.testing.assert_allclose(r.phase_std, n.phase_std, rtol=2e-7)

    def test_unknown_interference_cannot_improve_conditional_reference(self):
        r = compute_joint_crb(self.s, self.td, self.ta, .02, 200)
        c = compute_conditional_crb(self.s, self.ta, .02)
        for bound, conditional in zip((r.delay_std, r.magnitude_std, r.phase_std), c):
            self.assertTrue(np.all(bound >= conditional-1e-13))

    def test_noise_and_waveform_scale(self):
        r = compute_joint_crb(self.s, self.td, self.ta, .02, 200)
        more = compute_joint_crb(self.s, self.td, self.ta, .08, 200)
        stronger = compute_joint_crb(2*self.s, self.td, self.ta, .02, 200)
        np.testing.assert_allclose(more.phase_std, 2*r.phase_std, rtol=1e-12)
        np.testing.assert_allclose(stronger.delay_std, r.delay_std/2, rtol=1e-12)

    def test_nonoverlap_matches_independent_single_path_bounds(self):
        td = [10.43, 80.66, 150.25]
        r = compute_joint_crb(self.s, td, self.ta, .02, 230)
        for i in range(3):
            single = compute_joint_crb(self.s, [td[i]], [self.ta[i]], .02, 230)
            np.testing.assert_allclose(r.delay_std[i], single.delay_std[0], rtol=1e-12)

    def test_singular_and_nonregular_are_not_finite_pseudobounds(self):
        for td, ta in [([10.43, 10.43], [1, .5]), ([10.43], [0]),
                       ([10.43, 10.430000001], [1, .5])]:
            r = compute_joint_crb(self.s, td, ta, .02, 200)
            self.assertFalse(r.valid)
            self.assertIsNone(r.covariance_reim)
            self.assertTrue(np.isinf(r.delay_std).all())
        r = compute_joint_crb(self.s, [10.], [1], .02, 200)
        self.assertEqual(r.status, 'nonregular_integer_delay')

    def test_rms_reference_is_not_mean_std(self):
        self.assertAlmostEqual(rms_bound([1, 3]), np.sqrt(5))
        self.assertNotAlmostEqual(rms_bound([1, 3]), 2)


if __name__ == '__main__':
    unittest.main()
