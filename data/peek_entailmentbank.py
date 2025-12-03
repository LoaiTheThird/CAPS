import json
from pathlib import Path

# Adjust this if your folder name is slightly different; use tab-completion in a terminal to confirm.
TASK1_DIR = Path(
    "data/entailmentbank_raw/entailment_bank-main/data/public_dataset/"
    "entailment_trees_emnlp2021_data_v2/dataset/task_1"
)


def peek(path: Path, n: int = 3) -> None:
    print(f"== {path} ==")
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            print(ex)
            print("---")


def main() -> None:
    for name in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        p = TASK1_DIR / name
        if p.exists():
            peek(p)
        else:
            print(f"Missing: {p}")


if __name__ == "__main__":
    main()
