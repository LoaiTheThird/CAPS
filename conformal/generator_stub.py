# conformal/generator_stub.py

from typing import List


def generate_candidates(prompt: str, num_samples: int = 16) -> List[str]:
    """
    Stub generator.

    Later this will call vLLM / API. For now it just returns dummy
    proof strings so we can exercise the rest of the pipeline.
    """
    return [
        f"[DUMMY PROOF {i}] This is a placeholder proof for the given prompt."
        for i in range(num_samples)
    ]
