from __future__ import annotations
from itertools import combinations
from collections import defaultdict, Counter

from utils.metrics.base        import BaseMetric, SequenceView
from utils.llm.judge_hf_pref   import LLMJudgeHFPref
from utils.metrics.registry    import register


@register("llm_pairwise")
class LLMPairwise(BaseMetric):
    """
    Compare la séquence *finale* (visit == final_visit) de chaque baseline,
    2‑à‑2, via un LLM‑as‑a‑Judge.
    """
    global_metric = True

    FINAL_VISIT = -1        
    MAX_LIST    = 50        

    def __init__(self, cfg):
        super().__init__(cfg)
        self.judge = LLMJudgeHFPref(
            cfg.llm.model,
            temperature=max(0.1, cfg.llm.temperature)
        )


    @staticmethod
    def _extract_final_list(raw_records: list[dict], visit_id: int):
        return [r for r in raw_records if r.get("visit") == visit_id]

    @staticmethod
    def _to_titles(item_records: list[dict] | list[int],
                   max_len: int):
       
        titles = []
        for it in item_records:
            if isinstance(it, dict):             
                title = it.get("title") or str(it.get("item_key"))
            else:                                
              title = str(it)
            titles.append(title)

        counter = Counter(titles)
        clean   = []
        seen    = set()
        for t in titles:
            if t in seen:
                continue
            seen.add(t)
            clean.append(f"{t} ×{counter[t]}" if counter[t] > 1 else t)
            if len(clean) == max_len:
                break
        return clean


    def requires_predictions(self): 
        return False

    def __call__(self, seq_view: SequenceView, cfg):
        raw        = seq_view.raw              
        baselines  = list(raw)
        if len(baselines) < 2:
            return {}

       
        votes = defaultdict(lambda: defaultdict(int))
        final_visit = max(r.get("visit", 0) for recs in raw.values() for r in recs)

        for a, b in combinations(baselines, 2):
            rec_a = self._extract_final_list(raw[a], final_visit)
            rec_b = self._extract_final_list(raw[b], final_visit)

            list_a = self._to_titles(rec_a, self.MAX_LIST)
            list_b = self._to_titles(rec_b, self.MAX_LIST)
            if not list_a or not list_b:
                continue

            win = self.judge.compare(list_a, list_b, meta=None, cfg=cfg)
            winner, loser = ((a, b) if win == 0 else (b, a))
            votes[winner][loser] += 1

        scores = {b: 1.0 for b in baselines}
        for _ in range(100):
            for a in baselines:
                num = sum(votes[a][b] for b in baselines if b != a)
                den = sum(
                    (votes[a][b] + votes[b][a]) / (scores[a] + scores[b] + 1e-9)
                    for b in baselines if b != a
                )
                scores[a] = num / max(1e-9, den)

        max_s = max(scores.values()) or 1.0
        return {b: scores[b] / max_s for b in baselines}