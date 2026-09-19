import math
import unittest

import numpy as np

from src.metrics import compute_rank_metrics, fpr_at_threshold, logspace_auroc, threshold_for_fpr


class ProbeMetricTest(unittest.TestCase):
    def test_logspace_auroc_separable(self):
        y = np.array([0] * 1000 + [1] * 1000)
        scores = np.concatenate([np.linspace(0.0, 0.4, 1000), np.linspace(0.6, 1.0, 1000)])
        self.assertGreater(logspace_auroc(y, scores), 0.99)

    def test_logspace_auroc_inverted(self):
        y = np.array([0] * 1000 + [1] * 1000)
        scores = np.concatenate([np.ones(1000), np.zeros(1000)])
        self.assertEqual(logspace_auroc(y, scores), 0.0)

    def test_metrics_return_nan_for_single_class(self):
        metrics = compute_rank_metrics([0, 0, 0], [0.1, 0.2, 0.3]).as_dict()
        self.assertTrue(math.isnan(metrics["auroc"]))
        self.assertTrue(math.isnan(metrics["tpr@1fpr"]))
        self.assertTrue(math.isnan(metrics["logspace_auroc"]))

    def test_threshold_for_fpr(self):
        negatives = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = threshold_for_fpr(negatives, target_fpr=0.25)
        self.assertEqual(threshold, 0.4)
        self.assertEqual(fpr_at_threshold([0, 0, 1], [0.1, 0.4, 0.9], 0.4), 0.5)

    def test_high_logspace_auc_implies_nonchance_standard_auc(self):
        y = np.array([1] * 900 + [0] + [0] * 999 + [1] * 100)
        scores = np.arange(y.size, 0, -1)
        metrics = compute_rank_metrics(y, scores).as_dict()
        self.assertGreater(metrics["logspace_auroc"], 0.85)
        self.assertGreater(metrics["auroc"], 0.85)

    def test_logspace_auc_is_not_tpr_at_one_percent_alias(self):
        y = np.array([1] * 500 + [0] * 20 + [1] * 500 + [0] * 980)
        scores = np.arange(y.size, 0, -1)
        metrics = compute_rank_metrics(y, scores).as_dict()
        self.assertAlmostEqual(metrics["tpr@1fpr"], 0.5)
        self.assertGreater(metrics["logspace_auroc"], metrics["tpr@1fpr"] + 0.1)

    def test_two_percent_tpr_option_uses_distinct_truthful_key(self):
        y = np.array([1] * 500 + [0] * 20 + [1] * 500 + [0] * 980)
        scores = np.arange(y.size, 0, -1)

        metrics = compute_rank_metrics(y, scores, tpr_fpr=0.02).as_dict()

        self.assertIn("tpr@2fpr", metrics)
        self.assertNotIn("tpr@1fpr", metrics)
        self.assertAlmostEqual(metrics["tpr@2fpr"], 1.0)

    def test_rejects_unsupported_tpr_fpr(self):
        with self.assertRaisesRegex(ValueError, "must be 0.01 or 0.02"):
            compute_rank_metrics([0, 1], [0.0, 1.0], tpr_fpr=0.03)


if __name__ == "__main__":
    unittest.main()
