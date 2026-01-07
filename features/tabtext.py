# src/features/tabtext.py

from typing import Dict, List
import numpy as np
import pandas as pd
import torch


def get_tab_text_feature(csv_path: str, train: bool = True) -> Dict[str, List[str]]:

    df = pd.read_csv(open(csv_path, "rb"))
    data = np.array(df)

    number = data.shape[0]
    train_size = int(number * 0.7)

    train_feature: Dict[str, List[str]] = {}
    test_feature: Dict[str, List[str]] = {}

    name = str(data[0][1])
    train_flist: List[str] = []
    test_flist: List[str] = []
    count = 0

    for i in range(number):
        newname = str(data[i][1])
        tab_text = str(data[i][0])

        if name != newname:
            train_feature[name] = train_flist
            test_feature[name] = test_flist
            name = newname
            train_flist = []
            test_flist = []
            count = 1
            train_flist.append(tab_text)
        else:
            count += 1
            if count < train_size:
                train_flist.append(tab_text)
            else:
                test_flist.append(tab_text)

    train_feature[name] = train_flist
    test_feature[name] = test_flist

    return train_feature if train else test_feature


def find_tab_text_feature(listener: str, feature: Dict[str, List[str]], tokenizer, k: int, device: torch.device) -> torch.Tensor:

    if listener in feature:
        tmp = feature[listener]
        count = max(int(len(tmp) / k), 1)
        num = 0

        tmp_feature = []
        for _ in range(k):
            f_add = ""
            if num < len(tmp):
                for j in range(num, min(num + count, len(tmp))):
                    f_add += tmp[j]
            num += count

            cur_ids = tokenizer(f_add, return_tensors="pt")["input_ids"].view(-1).tolist()
            if len(cur_ids) <= 1024:
                cur_ids += [0] * (1024 - len(cur_ids))
            else:
                cur_ids = cur_ids[:1024]
            tmp_feature.append(cur_ids)

        listener_feature = [tmp_feature]  # (1, k, 1024)
    else:
        listener_feature = [[[0] * 1024] * k]

    return torch.tensor(listener_feature, dtype=torch.float32, device=device)
