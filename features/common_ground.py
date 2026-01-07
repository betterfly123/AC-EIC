# src/features/common_ground.py

from typing import List
import pandas as pd
import torch


def cicero_get(tokenizer, csv_path: str) -> torch.Tensor:

    data = pd.read_csv(csv_path, names=["knowledge"])
    knowledge_list = [str(x) for x in data["knowledge"].tolist()]
    k = tokenizer(knowledge_list, return_tensors="pt", max_length=512, padding="max_length")["input_ids"]
    return k


def get_common_ground(idx: int, U: torch.Tensor, comet: torch.Tensor, role_list: List[str], device: torch.device) -> List[torch.Tensor]:

    cm_list: List[torch.Tensor] = []
    for i in range(idx + 1):
        utt = U[0, i].to(device)
        knowledge = comet[7, i, :].to(device).reshape(768)
        cm = torch.cat((utt, knowledge), dim=0)
        cm_list.append(cm)
    return cm_list
