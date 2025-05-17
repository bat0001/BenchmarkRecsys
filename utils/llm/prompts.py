"""
Helper to build a *deterministic* prompt for pairwise list comparison.
Feel free to tweak wording / add context (user info, category, language…).
"""
from typing import Sequence

_PROMPT_TEMPLATE = (
    "You are an impartial evaluator of recommendation quality.\n"
    "Two candidate recommendation lists are given:\n\n"
    "List 1: {list1}\n"
    "List 2: {list2}\n\n"
    "Consider overall usefulness, diversity and relevance for the user.\n"
    "Reply with **only** the single number `1` if List 1 is better,\n"
    "or `2` if List 2 is better."
)

def format_pair_prompt(
    list_a: Sequence[str],
    list_b: Sequence[str],
    meta,                     
    cfg                       
) -> str:
    """
    Build the final prompt string fed to the LLM.

    Parameters
    ----------
    list_a, list_b : sequences of item identifiers / titles
    meta, cfg      : available for future contextualisation (user ID, locale…)

    Returns
    -------
    str : ready‑to‑send prompt
    """
    return _PROMPT_TEMPLATE.format(
        list1=", ".join(map(str, list_a)),
        list2=", ".join(map(str, list_b)),
    )