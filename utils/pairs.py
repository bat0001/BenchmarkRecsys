import random
from typing import List

def _get_item_id(rec: dict) -> str | int | None:
    for k in ("item_key", "item_id", "item_idx", "productId"):
        if k in rec:
            return rec[k]
    return None 

def _final_sequence(records: List[dict], visit: int) -> List[int]:
    if not records:
        return []
    print(records)
    return [
        _get_item_id(r)
        for r in records
        if r.get("visit") == visit and _get_item_id(r) is not None
    ]

def sample_pairs(raw_results: dict[str, list[dict]],
                 n_pairs: int,
                 final_visit: int,
                 *,
                 keys
                 ):
    baselines = list(raw_results)
    max_pairs = len(baselines) * (len(baselines) - 1) // 2
    n_pairs   = min(n_pairs, max_pairs)         

    pairs, seen = [], set()
    while len(pairs) < n_pairs:
        a, b = random.sample(baselines, 2)

        pair_id = tuple(sorted((a, b)))
        if pair_id in seen:
            continue
        seen.add(pair_id)

        seq_a = _final_sequence(raw_results[a], final_visit)
        seq_b = _final_sequence(raw_results[b], final_visit)

        if seq_a and seq_b:                                
            pairs.append((a, b, seq_a, seq_b))
    return pairs