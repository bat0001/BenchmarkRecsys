"""
Prompt factory for pairwise *product‑list* judging (LLM‑as‑a‑judge).

Two deterministic templates are exposed:

    •  with_ctx  – includes user purchase / rating history
    •  no_ctx    – no user context

The wording mirrors the NeurIPS‑24 appendix prompts, simply replacing
the music domain with amazon products.
"""
from __future__ import annotations
from typing import Sequence

_WITH_CTX = """\
You are an **impartial evaluator of e‑commerce recommendations**.

Below are the user’s past interactions:

--- BEGIN USER PURCHASED / LIKED PRODUCTS ---
{user_likes}
--- END USER PURCHASED / LIKED PRODUCTS ---

--- BEGIN USER DISLIKED / RETURNED PRODUCTS ---
{user_dislikes}
--- END USER DISLIKED / RETURNED PRODUCTS ---

Two candidate recommendation lists are given.

List 1 ➜ {list1}
List 2 ➜ {list2}

**Task.**
Think *step‑by‑step* about usefulness, diversity and personal relevance
*for THIS user*.  
After thinking, output **only** the single number `1` *if List 1 is better*,
or `2` *if List 2 is better*. No other text.
"""

_NO_CTX = """\
You are an **impartial evaluator of e‑commerce recommendations**.

Two candidate recommendation lists are given.

List 1 ➜ {list1}
List 2 ➜ {list2}

**Task.**
Think *step‑by‑step* about overall usefulness, diversity and relevance.  
After thinking, output **only** the single number `1` *if List 1 is better*,
or `2` *if List 2 is better*. No other text.
"""

_MOVIELENS_TEMPLATE = """
You are judging two movie‑recommendation lists for a user.

User history :
liked → {likes}
disliked → {dislikes}

List 1 : {list1}
List 2 : {list2}

Answer **1** if List 1 is better, **2** if List 2 is better.
"""

# def format_pair_prompt(
#     list_a: Sequence[str],
#     list_b: Sequence[str],
#     *,
#     user_likes: str | None = None,
#     user_dislikes: str | None = None,
# ) -> str:
#     """
#     Build the prompt string for the LLM judge.

#     Parameters
#     ----------
#     list_a / list_b : ordered sequences of product titles or IDs.
#     user_likes      : free‑text summary of items the user purchased / liked.
#     user_dislikes   : free‑text summary of items the user disliked / returned.
#                       Pass None to skip user‑context mode.

#     Returns
#     -------
#     str : ready‑to‑send prompt.
#     """
#     list1 = ", ".join(map(str, list_a))
#     list2 = ", ".join(map(str, list_b))

#     if user_likes is not None and user_dislikes is not None:
#         return _WITH_CTX.format(
#             list1=list1,
#             list2=list2,
#             user_likes=user_likes or "(no data provided)",
#             user_dislikes=user_dislikes or "(no data provided)",
#         )
#     else:
#         return _NO_CTX.format(list1=list1, list2=list2)

def format_pair_prompt(list_a, list_b, *, user_likes="", user_dislikes=""):
    context = f"""
User history:
liked    → {user_likes or 'none'}
disliked → {user_dislikes or 'none'}
""" if user_likes or user_dislikes else ""

    return f"""
You are judging two movie‑recommendation lists for a user.
{context}
List 1 : {', '.join(list_a)}
List 2 : {', '.join(list_b)}

**Task.**
Think step‑by‑step and answer **1** if List 1 is better,
or **2** if List 2 is better. No other text.
""".strip()