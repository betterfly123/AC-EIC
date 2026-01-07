# src/training/engine.py

from typing import Dict, Any, List, Tuple

import torch
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from src.configs.cm_prompt import PromptSpec, build_prompt_text
from src.configs.labels_meld import CLASS_TO_NAME


def _build_utterance_features(dataset, vid: int, device: torch.device) -> torch.Tensor:
    """
    U = (roberta1+2+3+4)/4, shape: (1, T, 1024)
    """
    cur_text1 = torch.tensor(dataset.roberta1[vid], dtype=torch.float32, device=device)
    cur_text2 = torch.tensor(dataset.roberta2[vid], dtype=torch.float32, device=device)
    cur_text3 = torch.tensor(dataset.roberta3[vid], dtype=torch.float32, device=device)
    cur_text4 = torch.tensor(dataset.roberta4[vid], dtype=torch.float32, device=device)
    U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4.0
    return U.unsqueeze(0)


def train_one_epoch(
    epoch: int,
    *,
    dataset,
    role_map,
    cicero_tensor: torch.Tensor,
    comet,
    tokenizer,
    model,
    optimizer,
    dataloader,
    tabtext_feature: Dict[str, List[str]],
    build_tabtext_fn,
    build_common_ground_fn,
    spec: PromptSpec,
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    ground_truth: List[int] = []
    preds: List[int] = []
    cicero_id = 0

    for t, data in enumerate(dataloader):
        if t % 10 == 0:
            print(f"[train] epoch={epoch} batch={t}/{len(dataloader)}")

        vids = data[-1]
        optimizer.zero_grad()

        for vid in vids:
            vid = int(vid)
            conv = dataset.sentences[vid]
            cur_emotions = dataset.emotion_labels[vid]
            roles = role_map[vid]

            U = _build_utterance_features(dataset, vid, device=device)  # (1,T,1024)
            length = len(conv)

            # comet 
            cur_comet = [comet.com[i][vid] for i in range(9)]
            cur_com = torch.tensor([cur_comet[j] for j in range(9)], dtype=torch.float32, device=device)  # (9,T,768)

            for i in range(length - 1):
                sent = conv[i]
                gt_class = int(cur_emotions[i + 1])
                listener = roles[i + 1]

                emotion_word = CLASS_TO_NAME[gt_class]
                x_text, y_text = build_prompt_text(sent, listener, spec, emotion_word)

                # features
                lf = build_tabtext_fn(listener, tabtext_feature, tokenizer, U.shape[1], device=device)  # (1,T,1024)
                com_r = cicero_tensor[cicero_id].to(device=device, dtype=torch.float32)  # (512,)
                cicero_id += 1
                common_ground = build_common_ground_fn(i, U, cur_com, roles, device=device)  # list of (1792,)

                # tokenize
                x = tokenizer(x_text, return_tensors="pt")
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)

                y = tokenizer(y_text, return_tensors="pt")["input_ids"]
                y = y.to(device)

                # label masking
                y[:, 1 : (spec.ple + spec.plp + 3)] = -100
                y[:, (-(spec.pre + spec.prp + 1)) : -1] = -100

                probs, loss, _ = model(
                    U,
                    common_ground,
                    com_r,
                    roles,
                    lf,
                    i,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=y,
                )

                loss.backward()

                pred_class = int(probs.argmax(dim=-1).item())
                ground_truth.append(gt_class)
                preds.append(pred_class)

        optimizer.step()

    p, r, f1, _ = precision_recall_fscore_support(ground_truth, preds, average="weighted")
    report = classification_report(ground_truth, preds, digits=4)
    print(report)

    return {
        "epoch": epoch,
        "weighted_f1": float(f1),
        "report": report,
    }


@torch.no_grad()
def evaluate(
    epoch: int,
    *,
    dataset,
    role_map,
    cicero_tensor: torch.Tensor,
    comet,
    tokenizer,
    model,
    dataloader,
    tabtext_feature: Dict[str, List[str]],
    build_tabtext_fn,
    build_common_ground_fn,
    spec: PromptSpec,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    ground_truth: List[int] = []
    preds: List[int] = []
    cicero_id = 0

    for t, data in enumerate(dataloader):
        if t % 10 == 0:
            print(f"[test] epoch={epoch} batch={t}/{len(dataloader)}")

        vids = data[-1]
        for vid in vids:
            vid = int(vid)
            conv = dataset.sentences[vid]
            cur_emotions = dataset.emotion_labels[vid]
            roles = role_map[vid]

            U = _build_utterance_features(dataset, vid, device=device)
            length = len(conv)

            cur_comet = [comet.com[i][vid] for i in range(9)]
            cur_com = torch.tensor([cur_comet[j] for j in range(9)], dtype=torch.float32, device=device)

            for i in range(length - 1):
                sent = conv[i]
                gt_class = int(cur_emotions[i + 1])
                listener = roles[i + 1]
                emotion_word = CLASS_TO_NAME[gt_class]

                x_text, y_text = build_prompt_text(sent, listener, spec, emotion_word)

                lf = build_tabtext_fn(listener, tabtext_feature, tokenizer, U.shape[1], device=device)
                com_r = cicero_tensor[cicero_id].to(device=device, dtype=torch.float32)
                cicero_id += 1
                common_ground = build_common_ground_fn(i, U, cur_com, roles, device=device)

                x = tokenizer(x_text, return_tensors="pt")
                input_ids = x["input_ids"].to(device)
                attention_mask = x["attention_mask"].to(device)

                y = tokenizer(y_text, return_tensors="pt")["input_ids"].to(device)
                y[:, 1 : (spec.ple + spec.plp + 3)] = -100
                y[:, (-(spec.pre + spec.prp + 1)) : -1] = -100

                probs, loss, _ = model(
                    U,
                    common_ground,
                    com_r,
                    roles,
                    lf,
                    i,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=y,
                )

                pred_class = int(probs.argmax(dim=-1).item())
                ground_truth.append(gt_class)
                preds.append(pred_class)

    fscore = metrics.f1_score(ground_truth, preds, average="weighted")
    report = classification_report(ground_truth, preds, digits=4)
    print(report)

    return {
        "epoch": epoch,
        "weighted_f1": float(fscore),
        "report": report,
    }
