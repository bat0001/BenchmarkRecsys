from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, functools, logging, os

_DEFAULT_DTYPE = torch.float16       

@functools.lru_cache(maxsize=None)
def load_hf_model(model_id: str,
                  device: str | torch.device = "cuda",
                  dtype  = _DEFAULT_DTYPE):
    """Load (tokenizer, model) one time, then LRU cache."""
    logging.info(f"[HF‑LLM] loading {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id,
                                        use_fast=True,
                                        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,         
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()

    # facultative : avoid OOM on  < 24 Go cards
    if os.getenv("GFN_LLM_GRAD_CP", "0") == "1":
        model.gradient_checkpointing_enable()
    return tok, model