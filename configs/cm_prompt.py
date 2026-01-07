# src/configs/cm_prompt.py

from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class PromptSpec:
    ple: int = 3
    plp: int = 3
    pre: int = 3
    prp: int = 3
    inject_global_idx: int = 7
    inject_common_idx: int = 8

def compute_prompt_positions(input_len: int, spec: PromptSpec) -> List[int]:

    left = list(range(1, spec.ple + spec.plp + 1))
    right = list(range(input_len - spec.pre - spec.prp - 1, input_len - 1))
    return left + right

def target_mask_position_from_end(spec: PromptSpec) -> int:

    return -(spec.pre + spec.prp) - 2

def build_prompt_text(sent: str, listener: str, spec: PromptSpec, emotion_word: str) -> Tuple[str, str]:

    sent = sent.replace("x92", "'")
    target = "Target:" + sent
    fix_prompt = f"The possible emotional reaction of the {listener} in response to target is "

    x = " <mask>" * (spec.ple + spec.plp + 2) + " " + target + " " + fix_prompt + " <mask>" * (1 + spec.pre + spec.prp)
    y = " <mask>" * (spec.ple + spec.plp + 2) + " " + target + " " + fix_prompt + emotion_word + " <mask>" * (spec.pre + spec.prp)
    return x, y
