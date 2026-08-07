import unittest

from conformal.ecthr_multilabel import (
    calibrate_global_threshold,
    calibrate_label_thresholds,
    conformal_quantile,
    prediction_set,
    prediction_set_labelwise,
    set_covers_gold,
)


class ConformalTests(unittest.TestCase):
    def test_finite_sample_quantile(self) -> None:
        self.assertEqual(conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.25), 0.4)

    def test_prediction_set_is_thresholded_and_stable(self) -> None:
        scores = {"b": 0.9, "a": 0.9, "c": 0.4}
        self.assertEqual(prediction_set(scores, q_alpha=0.2), ["a", "b"])

    def test_global_calibration_uses_worst_gold_label(self) -> None:
        rows = [
            {"gold_labels": ["a", "b"], "scores": {"a": 0.9, "b": 0.6}},
            {"gold_labels": ["a"], "scores": {"a": 0.8, "b": 0.2}},
        ]
        self.assertAlmostEqual(calibrate_global_threshold(rows, alpha=0.5), 0.4)

    def test_labelwise_calibration_falls_back_for_unseen_label(self) -> None:
        rows = [{"gold_labels": ["a"], "scores": {"a": 0.8, "b": 0.7}}]
        thresholds = calibrate_label_thresholds(
            rows,
            label_names=["a", "b"],
            alpha=0.5,
            fallback_q=0.6,
        )
        self.assertAlmostEqual(thresholds["a"], 0.2)
        self.assertEqual(thresholds["b"], 0.6)
        self.assertEqual(
            prediction_set_labelwise(
                {"a": 0.8, "b": 0.5},
                thresholds,
                fallback_q=0.6,
            ),
            ["a", "b"],
        )

    def test_set_coverage_requires_every_gold_label(self) -> None:
        self.assertTrue(set_covers_gold(["a", "b"], ["a"]))
        self.assertFalse(set_covers_gold(["a"], ["a", "b"]))


if __name__ == "__main__":
    unittest.main()
