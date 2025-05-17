import random
from typing import List

def _final_sequence(records: List[dict], visit: int) -> List[int]:
    if not records:
        return []
    return [r["item_idx"] for r in records if r["visit"] == visit]

def sample_pairs(raw_results: dict[str, list[dict]],
                 n_pairs: int,
                 final_visit: int):
    """
    Tire au plus `n_pairs` couples distincts de baselines et renvoie
    pour chacun les séquences finales correspondantes.
    """

    baselines = list(raw_results)
    max_pairs = len(baselines) * (len(baselines) - 1) // 2
    n_pairs   = min(n_pairs, max_pairs)         

    pairs, seen = [], set()
    while len(pairs) < n_pairs:
        a, b = random.sample(baselines, 2)
        key  = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        seq_a = _final_sequence(raw_results[a], final_visit)
        seq_b = _final_sequence(raw_results[b], final_visit)
        if seq_a and seq_b:                    
            pairs.append((a, b, seq_a, seq_b))


    return pairs