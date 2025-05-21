from collections import defaultdict
import numpy as np

from utils.metrics.base      import BaseMetric, SequenceView
from utils.metrics.registry  import register
from utils.llm.judge_hf_pref import LLMJudgeHFPref
from utils.pairs             import sample_pairs


@register("llm_bt")
class LLMPreferenceBT(BaseMetric):
    """Bradley–Terry based on LLM judge pol."""
    global_metric = True      

    def __init__(self, cfg):
        super().__init__(cfg)
        self.judge = LLMJudgeHFPref(cfg.llm.model,
                                    temperature=max(0.1, cfg.llm.temperature))

    def requires_predictions(self) -> bool:
        return False

    def __call__(self, seq_view: SequenceView, cfg):
        baselines = cfg.baselines
        if len(baselines) < 2:
            return {}

        votes = defaultdict(lambda: defaultdict(int))
        final_visit = cfg.num_iterations - 1
        pairs = sample_pairs(seq_view.raw,
                     cfg.llm.bt_pairs,
                     final_visit,
                     keys=seq_view.item_key)
        for bi, bj, seq_i, seq_j in pairs:
            win = self.judge.compare(seq_i, seq_j, meta=None, cfg=cfg)
            winner, loser = (bi, bj) if win == 0 else (bj, bi)
            votes[winner][loser] += 1

        s = {b: 1.0 for b in baselines}
        for _ in range(100):
            for a in baselines:
                num = sum(votes[a][b] for b in baselines if b != a)
                den = sum((votes[a][b] + votes[b][a]) / (s[a] + s[b])
                          for b in baselines if b != a)
                s[a] = num / max(1, den)

        max_s = max(s.values())
        return {b: s[b] / max_s for b in baselines}