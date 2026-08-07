import unittest

from legal.ecthr_features import (
    default_reasoner_features,
    label_features_from_reasoner_record,
)
from legal.legalreasoner_prompts import sanitize_candidates


class ReasonerFeatureTests(unittest.TestCase):
    def test_reasoner_record_is_flattened_per_label(self) -> None:
        features = label_features_from_reasoner_record(
            label_names=["3", "8", "14"],
            candidate_articles=[
                {"label": "3", "reason": "detention conditions"},
                {"label": "8", "reason": "private life"},
            ],
            reasoning_articles=[
                {
                    "label": "3",
                    "preliminary_decision": "support",
                    "reasoning_steps": ["step one", "step two"],
                }
            ],
            verification_articles=[
                {
                    "label": "3",
                    "label_verdict": "supported",
                    "label_evidence": "recorded fact",
                    "step_checks": [
                        {"verdict": "supported"},
                        {"verdict": "partial"},
                    ],
                }
            ],
            support_only_labels=["3"],
        )

        self.assertEqual(features["3"]["candidate_rank"], 0.0)
        self.assertEqual(features["3"]["n_reasoning_steps"], 2.0)
        self.assertEqual(features["3"]["supported_fraction"], 0.5)
        self.assertEqual(features["3"]["hard_support_gate"], 1.0)
        self.assertEqual(features["8"]["candidate_rank_norm"], 1.0)
        self.assertEqual(features["14"]["candidate_present"], 0.0)

    def test_default_features_have_no_reasoner_signal(self) -> None:
        features = default_reasoner_features(["3"])["3"]
        self.assertEqual(features["candidate_present"], 0.0)
        self.assertEqual(features["candidate_rank"], -1.0)
        self.assertEqual(features["verdict_supported"], 0.0)

    def test_candidate_sanitizer_filters_invalid_and_duplicate_labels(self) -> None:
        candidates = sanitize_candidates(
            [
                {"label": "3", "reason": "first"},
                {"label": "3", "reason": "duplicate"},
                {"label": "99", "reason": "invalid"},
                {"label": "8", "reason": "second"},
            ],
            ["3", "8"],
            max_candidates=2,
        )
        self.assertEqual([item["label"] for item in candidates], ["3", "8"])


if __name__ == "__main__":
    unittest.main()
