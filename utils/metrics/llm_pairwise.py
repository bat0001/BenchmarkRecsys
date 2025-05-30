from __future__ import annotations
from itertools import combinations
from collections import defaultdict, Counter

from utils.metrics.base      import BaseMetric, SequenceView
from utils.llm.judge_hf_pref import LLMJudgeHFPref
from utils.metrics.registry  import register


@register("llm_pairwise")
class LLMPairwise(BaseMetric):
    """
    Pour CHAQUE utilisateur, on compare les recommandations du dernier `visit`
    entre toutes les baselines deux‑à‑deux via un LLM‑Judge.

    Le score final est agrégé par Bradley‑Terry.
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
    def _dedup_titles(items: list[str], max_len: int) -> list[str]:
        """
        items : titres déjà convertis en str.
        • Dé‑duplique en conservant l’ordre.
        • Compacte « X apparaît n fois » →  « X ×n ».
        • Tronque à `max_len`.
        """
        counter = Counter(items)
        seen, clean = set(), []
        for t in items:
            if t in seen:
                continue
            seen.add(t)
            clean.append(f"{t} ×{counter[t]}" if counter[t] > 1 else t)
            if len(clean) == max_len:
                break
        return clean

    @classmethod
    def _final_list(cls, records: list[dict], visit_id: int) -> list[str]:
        """
        • Filtre sur le `visit` final
        • Convertit chaque record → titre (ou item_key si pas de titre)
        • Dé‑duplique & tronque
        """
        titles = [
            (r.get("title") or str(r.get("item_key")))
            for r in records
            if r.get("visit") == visit_id
        ]
        return cls._dedup_titles(titles, cls.MAX_LIST)

    def requires_predictions(self):    
        return False

    def __call__(self, seq_view: SequenceView, cfg):
        raw = seq_view.raw                      
        baselines = list(raw)
        if len(baselines) < 2:
            return {}

        final_visit = max(r.get("visit", 0)
                          for recs in raw.values() for r in recs)

        votes = defaultdict(lambda: defaultdict(int))

        user_ids = {
            r["user_id"] for recs in raw.values() for r in recs
            if "user_id" in r
        }

        for u in user_ids:
            user_lists = {}
            for bl in baselines:
                recs_u = [r for r in raw[bl] if r.get("user_id") == u]
                user_lists[bl] = self._final_list(recs_u, final_visit)

            for a, b in combinations(baselines, 2):
                list_a, list_b = user_lists[a], user_lists[b]
                if not list_a or not list_b:
                    continue

                win = self.judge.compare(list_a, list_b,
                                         meta=None, cfg=cfg)
                winner, loser = (a, b) if win == 0 else (b, a)
                votes[winner][loser] += 1

        scores = {b: 1.0 for b in baselines}
        for _ in range(100):
            for a in baselines:
                num = sum(votes[a][b] for b in baselines if b != a)
                den = sum(
                    (votes[a][b] + votes[b][a]) /
                    (scores[a] + scores[b] + 1e-9)
                    for b in baselines if b != a
                )
                scores[a] = num / max(1e-9, den)

        max_s = max(scores.values()) or 1.0
        return {b: scores[b] / max_s for b in baselines}