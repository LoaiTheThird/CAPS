from pathlib import Path
from typing import List

import clingo


KB_PATH = Path(__file__).with_name("toy_kb.lp")


def run_clingo() -> List[List[clingo.Symbol]]:
    """
    Run clingo on the toy KB and collect all stable models as lists of atoms.
    """
    ctl = clingo.Control()
    ctl.load(str(KB_PATH))
    ctl.ground([("base", [])])

    models: List[List[clingo.Symbol]] = []

    def on_model(model: clingo.Model) -> None:
        atoms = model.symbols(atoms=True)
        models.append(atoms)

    ctl.solve(on_model=on_model)
    return models


def hypothesis_entailed(models: List[List[clingo.Symbol]]) -> bool:
    """
    Check whether there exists a model containing holds(hypothesis).
    """
    for atoms in models:
        for atom in atoms:
            if atom.name == "holds" and len(atom.arguments) == 1:
                arg = atom.arguments[0]
                # We only care that it's holds(hypothesis), ignore other possible holds/args.
                if arg.name == "hypothesis":
                    return True
    return False


def main() -> None:
    print(f"Using KB file: {KB_PATH}")

    models = run_clingo()

    print("\nStable models found:")
    for i, atoms in enumerate(models, start=1):
        pretty = sorted(str(a) for a in atoms)
        print(f"  Model {i}: {pretty}")

    entailed = hypothesis_entailed(models)
    print("\nHypothesis entailed:", entailed)

    if not entailed:
        raise SystemExit("Smoke test FAILED ❌")

    print("Smoke test PASSED ✅")


if __name__ == "__main__":
    main()
