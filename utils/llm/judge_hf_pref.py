import random, torch
from transformers import GenerationConfig
from .hf       import load_hf_model
from .cache    import get_cached, set_cached
from .prompts  import format_pair_prompt


class LLMJudgeHFPref:
    """
    Compare two séquences with HF local. model
    Return 0 if the 1st list win, 1 otherwise.
    """

    def __init__(self,
                 model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 device: str   = "cuda",
                 temperature: float = 0.0,
                 max_tokens: int = 4):
        self.tok, self.model = load_hf_model(model_id, device)
        self.max_t = max_tokens

        self.gen_cfg = GenerationConfig(
            max_new_tokens = max_tokens,
            temperature    = max(1e-4, temperature),  # strictly > 0
            do_sample      = temperature > 0.0
        )

    @torch.inference_mode()
    def compare(self, list_a, list_b, meta, cfg) -> int:
        print('in compare')
        flip = random.random() < 0.5
        print("flip")
        if flip:
            list_a, list_b = list_b, list_a

        prompt = format_pair_prompt(list_a, list_b, meta, cfg)
        if (v := get_cached(prompt)) is not None:
            return v ^ flip

        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        print("→ LLM compare", len(prompt), "tokens…", flush=True)
        out = self.model.generate(**ids, generation_config=self.gen_cfg)
        reply = self.tok.decode(out[0][ids.input_ids.shape[1]:],
                                skip_special_tokens=True).strip()

        win = 0 if reply.startswith("1") else 1
        set_cached(prompt, win)
        return win ^ flip