import random
from typing import List
import itertools

def _get_item_id(rec: dict) -> str | int | None:
    for k in ("item_key", "item_id", "item_idx", "productId"):
        if k in rec:
            return rec[k]
    return None 

def _final_sequence(records: List[dict], visit: int) -> List[int]:
    if not records:
        return []
    return [
        _get_item_id(r)
        for r in records
        if r.get("visit") == visit and _get_item_id(r) is not None
    ]
def sample_pairs(
        raw_results: dict[str, list[dict]],
        n_pairs:     int,
        final_visit: int,
        *,
        keys
):
    baselines = list(raw_results)
    all_pairs = list(itertools.combinations(baselines, 2))
    random.shuffle(all_pairs)

    pairs = []
    for a, b in all_pairs:
        if len(pairs) >= n_pairs:
            break
        seq_a = _final_sequence(raw_results[a], final_visit)
        seq_b = _final_sequence(raw_results[b], final_visit)
        if seq_a and seq_b:
            pairs.append((a, b, seq_a, seq_b))

    return pairs