from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, logging, functools
from utils.device import DEVICE

_DEFAULT_DTYPE = torch.float16       

def _str_device(dev: torch.device | str) -> str:
    return dev if isinstance(dev, str) else dev.type if dev.index is None else f"cuda:{dev.index}"

@functools.lru_cache(maxsize=None)
def load_hf_model(model_id: str,
                  device: str | torch.device | None = None,
                  dtype_if_cpu=torch.float32,
                  dtype_if_gpu=torch.float16):
    """
    - `device=None`  → utils.device.DEVICE (auto)
    - GPU available  → fp16 + Flash‑Attention (by default)
    - CPU only       → fp32 + SDPA (deactivate Flash‑Attention)
    """
    dev_obj = DEVICE if device is None else device
    dev_str = _str_device(dev_obj)
    logging.info(f"[LLM] loading {model_id} on {dev_str}")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    common_kwargs = dict(
        trust_remote_code=True,
        device_map="auto" if dev_str.startswith("cuda") else {"": "cpu"},
    )

    if dev_str.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_if_gpu,
            **common_kwargs,
        )
    else:  # CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_if_cpu,
            attn_implementation="sdpa",
            **common_kwargs,
        )

    model.gradient_checkpointing_enable()
    return tok, model