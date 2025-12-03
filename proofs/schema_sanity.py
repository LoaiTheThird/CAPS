import json

from proofs.schema import ProofTree, ProofNode


def make_toy_tree() -> ProofTree:
    """
    Toy proof:

      f1: Metal conducts electricity.
      f2: Copper is a metal.
      h1: Copper conducts electricity.   (premises: f1, f2, rule: 'modus-ponens')
    """
    n1 = ProofNode(
        id="f1",
        text="Metal conducts electricity.",
    )
    n2 = ProofNode(
        id="f2",
        text="Copper is a metal.",
    )
    n3 = ProofNode(
        id="h1",
        text="Copper conducts electricity.",
        premises=["f1", "f2"],
        rule="modus-ponens",
    )

    return ProofTree(
        id="toy-001",
        hypothesis_id="h1",
        nodes=[n1, n2, n3],
    )


def main() -> None:
    tree = make_toy_tree()
    as_dict = tree.to_dict()

    print("Toy ProofTree as JSON:\n")
    print(json.dumps(as_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
