# src/data/meld.py

import pickle
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MELDDataset:

    def __init__(self, path: str, split: str = "train"):
        (
            self.speakers,
            self.emotion_labels,
            self.sentiment_labels,
            self.roberta1,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.sentences,
            self.trainIds,
            self.testIds,
            self.validIds,
        ) = pickle.load(open(path, "rb"), encoding="latin1")

        if split == "train":
            self.keys = [x for x in self.trainIds]
        elif split == "test":
            self.keys = [x for x in self.testIds]
        elif split == "valid":
            self.keys = [x for x in self.validIds]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.len = len(self.keys)

    def __getitem__(self, index: int):
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.roberta1[vid]),
            torch.FloatTensor(self.roberta2[vid]),
            torch.FloatTensor(self.roberta3[vid]),
            torch.FloatTensor(self.roberta4[vid]),
            torch.FloatTensor([[1, 0] if x == "0" else [0, 1] for x in self.speakers[vid]]),
            torch.FloatTensor([1] * len(self.emotion_labels[vid])),
            torch.LongTensor(self.emotion_labels[vid]),
            vid,
        )

    def __len__(self) -> int:
        return self.len

    def collate_fn(self, data):
        d = pd.DataFrame(data)
        return [list(d[i]) for i in d]


class MELDComet:
    def __init__(self, path: str):
        self.com = pickle.load(open(path, "rb"))

def role_dataset_train(csv_path: str) -> Dict[int, List[str]]:

    df = pd.read_csv(open(csv_path, "rb"))
    data = np.array(df)

    role_map: Dict[int, List[str]] = {}
    for row in data:
        vid = int(row[5])
        speaker = str(row[2])
        role_map.setdefault(vid, []).append(speaker)
    return role_map


def role_dataset_test(csv_path: str) -> Dict[int, List[str]]:
    df = pd.read_csv(open(csv_path, "rb"))
    data = np.array(df)

    role_map: Dict[int, List[str]] = {}
    for row in data:
        vid = int(row[5])
        speaker = str(row[2])
        role_map.setdefault(vid, []).append(speaker)
    return role_map


def get_proper_loaders(path: str, batch_size: int = 16, num_workers: int = 0, pin_memory: bool = False):
    trainset = MELDDataset(path, "train")
    testset = MELDDataset(path, "test")
    validset = MELDDataset(path, "valid")

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        collate_fn=validset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader
