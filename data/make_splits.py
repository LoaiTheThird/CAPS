import json
import random
from pathlib import Path
from typing import Any, Dict, List

RNG = random.Random(42)

RAW_DEV = Path(
    "data/entailmentbank_raw/entailment_bank-main/data/public_dataset/"
    "entailment_trees_emnlp2021_data_v2/dataset/task_1/dev.jsonl"
)

OUT_CALIB = Path("data/eb_task1_dev_calib.jsonl")
OUT_EVAL = Path("data/eb_task1_dev_eval.jsonl")

CALIB_FRACTION = 0.5  # use 50% of dev for calibration


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def save_jsonl(examples: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    dev = load_jsonl(RAW_DEV)
    print(f"Loaded {len(dev)} dev examples from {RAW_DEV}")

    if not dev:
        raise SystemExit("Dev set is empty – check the path.")

    RNG.shuffle(dev)

    calib_size = int(len(dev) * CALIB_FRACTION)
    calib = dev[:calib_size]
    eval_rest = dev[calib_size:]

    print(f"Calib size: {len(calib)}")
    print(f"Eval size:  {len(eval_rest)}")

    save_jsonl(calib, OUT_CALIB)
    save_jsonl(eval_rest, OUT_EVAL)

    print(f"Wrote calib to {OUT_CALIB}")
    print(f"Wrote eval  to {OUT_EVAL}")


if __name__ == "__main__":
    main()
