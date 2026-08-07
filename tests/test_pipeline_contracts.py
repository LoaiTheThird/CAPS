import tempfile
import unittest
from pathlib import Path

import numpy as np

from legal.build_ecthr_feature_table import build_rows_for_split
from legal.ecthr_features import write_jsonl
from legal.eval_ecthr_conformal import evaluate_sets
from legal.run_ecthr_base_scores import rows_from_scores
from legal.train_ecthr_meta_scorer import grouped_score_rows


class PipelineContractTests(unittest.TestCase):
    def test_feature_table_can_restrict_to_reasoner_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.jsonl"
            reasoner_path = root / "reasoner.jsonl"
            write_jsonl(
                base_path,
                [
                    {"id": 0, "gold_labels": ["3"], "scores": {"3": 0.8, "8": 0.2}},
                    {"id": 1, "gold_labels": ["8"], "scores": {"3": 0.3, "8": 0.7}},
                ],
            )
            write_jsonl(
                reasoner_path,
                [{"id": 0, "per_label_features": {"3": {"verdict_supported": 1.0}}}],
            )

            rows = build_rows_for_split(
                base_path=base_path,
                reasoner_path=reasoner_path,
                split="validation",
                allow_missing_reasoner=False,
                require_complete_reasoner=False,
                restrict_to_reasoner_cases=True,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual({row["id"] for row in rows}, {0})

    def test_empty_reasoner_file_cannot_define_a_restricted_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path = root / "base.jsonl"
            reasoner_path = root / "reasoner.jsonl"
            write_jsonl(base_path, [{"id": 0, "gold_labels": ["3"], "scores": {"3": 0.8}}])
            reasoner_path.touch()

            with self.assertRaises(ValueError):
                build_rows_for_split(
                    base_path=base_path,
                    reasoner_path=reasoner_path,
                    split="validation",
                    allow_missing_reasoner=False,
                    require_complete_reasoner=False,
                    restrict_to_reasoner_cases=True,
                )

    def test_metrics_reject_missing_prediction_sets(self) -> None:
        rows = [
            {"gold_labels": ["a"], "scores": {"a": 0.9, "b": 0.1}},
            {"gold_labels": ["b"], "scores": {"a": 0.2, "b": 0.8}},
        ]
        with self.assertRaises(ValueError):
            evaluate_sets(rows, ["a", "b"], [["a"]])

    def test_metrics_match_a_small_known_example(self) -> None:
        rows = [
            {"gold_labels": ["a"], "scores": {"a": 0.9, "b": 0.1}},
            {"gold_labels": ["b"], "scores": {"a": 0.2, "b": 0.8}},
        ]
        metrics = evaluate_sets(rows, ["a", "b"], [["a"], ["a"]])
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["avg_set_size"], 1.0)
        self.assertEqual(metrics["micro_f1"], 0.5)
        self.assertEqual(metrics["micro_recall"], 0.5)

    def test_score_builders_reject_length_mismatches(self) -> None:
        meta = [{"id": 0}, {"id": 1}]
        with self.assertRaises(ValueError):
            rows_from_scores(meta, ["3"], np.asarray([[0.9]]))

        feature_rows = [
            {"id": 0, "split": "test", "label": "3", "gold_labels": ["3"]},
            {"id": 0, "split": "test", "label": "8", "gold_labels": ["3"]},
        ]
        with self.assertRaises(ValueError):
            grouped_score_rows(feature_rows, [0.9], method="base_only")


if __name__ == "__main__":
    unittest.main()
