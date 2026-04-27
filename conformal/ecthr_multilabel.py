from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def conformal_quantile(values: Sequence[float], alpha: float) -> float:
    if not values:
        raise ValueError("Cannot calibrate with no scores.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    xs = sorted(float(v) for v in values)
    n = len(xs)
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = max(1, min(k, n))
    return float(xs[k - 1])


def case_score_for_gold(gold_labels: Iterable[str], label_scores: Dict[str, float]) -> float:
    """Multi-label nonconformity: max true-label miss score, 1 - p_y."""
    gold = list(gold_labels)
    if not gold:
        return 0.0
    return max(1.0 - float(label_scores.get(label, 0.0)) for label in gold)


def prediction_set(label_scores: Dict[str, float], q_alpha: float) -> List[str]:
    return [
        label
        for label, score in sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))
        if 1.0 - float(score) <= q_alpha
    ]


def set_covers_gold(predicted: Iterable[str], gold: Iterable[str]) -> bool:
    pred_set = set(predicted)
    return set(gold).issubset(pred_set)


def calibrate_global_threshold(
    case_scores: Iterable[Dict[str, object]],
    *,
    alpha: float,
) -> float:
    scores = [
        case_score_for_gold(row.get("gold_labels", []), row.get("scores", {}))  # type: ignore[arg-type]
        for row in case_scores
    ]
    return conformal_quantile(scores, alpha)


def calibrate_label_thresholds(
    case_scores: Iterable[Dict[str, object]],
    *,
    label_names: List[str],
    alpha: float,
    fallback_q: float,
) -> Dict[str, float]:
    by_label: Dict[str, List[float]] = {label: [] for label in label_names}
    for row in case_scores:
        scores = row.get("scores", {})
        if not isinstance(scores, dict):
            continue
        for label in row.get("gold_labels", []):  # type: ignore[union-attr]
            label = str(label)
            if label in by_label:
                by_label[label].append(1.0 - float(scores.get(label, 0.0)))

    out: Dict[str, float] = {}
    for label in label_names:
        out[label] = conformal_quantile(by_label[label], alpha) if by_label[label] else fallback_q
    return out


def prediction_set_labelwise(
    label_scores: Dict[str, float],
    q_by_label: Dict[str, float],
    *,
    fallback_q: float,
) -> List[str]:
    return [
        label
        for label, score in sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))
        if 1.0 - float(score) <= float(q_by_label.get(label, fallback_q))
    ]
