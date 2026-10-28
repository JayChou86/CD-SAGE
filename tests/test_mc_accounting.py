"""Regression for mismatched MC budgets and attempted estimator failures."""
import unittest
import numpy as np
from scripts.mc_revision import collect_algorithm_trials, summarize_trials, noise_observations


class MonteCarloAccountingTests(unittest.TestCase):
    def test_noise_prefix_is_stable_when_budgets_or_snr_list_change(self):
        clean = np.ones(20, complex)
        short = noise_observations(clean, .02, 40, -5., 2)
        other_snr = noise_observations(clean, .02, 40, 0., 2)
        long = noise_observations(clean, .02, 40, -5., 4)
        np.testing.assert_array_equal(short, long[:2])
        self.assertFalse(np.array_equal(short, other_snr))
        np.testing.assert_array_equal(noise_observations(clean, .02, 40, -0., 2), other_snr)

    def collect(self, runners, budgets):
        return collect_algorithm_trials([1., 2., 3.], runners, budgets,
                                        np.array([0.]), np.array([1.+0j]),
                                        np.array([1.+0j]), lambda d, a: a)

    def test_inactive_algorithm_outputs_are_never_accumulated_again(self):
        calls = [0, 0]
        def runner(index):
            def run(observation):
                calls[index] += 1
                return np.array([observation]), np.array([1.+0j])
            return run
        raw, failures = self.collect([runner(0), runner(1)], [1, 3])
        summary = summarize_trials(raw)
        self.assertEqual(calls, [1, 3])
        self.assertEqual(failures, [])
        np.testing.assert_array_equal(summary['attempted_counts'], [1, 3])
        np.testing.assert_allclose(summary['rmse_delay'], [1., np.sqrt(14/3)])
        self.assertTrue(np.isnan(raw['delay_error'][0, 1:]).all())
        self.assertFalse(raw['attempted'][0, 1:].any())

    def test_failure_does_not_delete_other_algorithms_or_observation(self):
        def sometimes(observation):
            if observation == 2:
                raise RuntimeError('deliberate failure')
            return [observation], [1.+0j]
        raw, failures = self.collect([sometimes, lambda y: ([y], [1.+0j])], [3, 3])
        summary = summarize_trials(raw)
        np.testing.assert_array_equal(summary['attempted_counts'], [3, 3])
        np.testing.assert_array_equal(summary['successful_counts'], [2, 3])
        self.assertEqual(len(failures), 1)
        self.assertTrue(np.isinf(summary['rmse_delay'][0]))
        self.assertAlmostEqual(summary['rmse_delay'][1], np.sqrt(14/3))
        self.assertTrue(np.isnan(summary['rmse_phase'][0]))

    def test_zero_estimated_amplitude_has_no_invented_phase(self):
        raw, failures = self.collect([lambda y: ([0.], [0.+0j])], [3])
        summary = summarize_trials(raw)
        self.assertEqual(failures, [])
        self.assertTrue(np.isnan(summary['rmse_phase'][0]))
        self.assertEqual(summary['phase_defined_counts'][0], 0)
        self.assertEqual(summary['rmse_complex_amplitude'][0], 1.)


if __name__ == '__main__':
    unittest.main()
