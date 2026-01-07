# src/configs/labels_meld.py

from dataclasses import dataclass
from typing import Dict, List


EMOTION_TOKEN_IDS: List[int] = [7974, 2755, 2490, 17437, 5823, 30883, 6378]


TOKEN_ID_TO_CLASS: Dict[int, int] = {tid: i for i, tid in enumerate(EMOTION_TOKEN_IDS)}

CLASS_TO_NAME: Dict[int, str] = {
    0: "neutral",
    1: "surprise",
    2: "fear",
    3: "sadness",
    4: "joy",
    5: "disgust",
    6: "anger",
}

NAME_TO_CLASS: Dict[str, int] = {v: k for k, v in CLASS_TO_NAME.items()}

@dataclass(frozen=True)
class MeldLabelSpace:
    emotion_token_ids: List[int] = None

    def __post_init__(self):
        object.__setattr__(self, "emotion_token_ids", EMOTION_TOKEN_IDS)

LABEL_SPACE = MeldLabelSpace()
