# src/scripts/run_meld_cm.py

import argparse
import torch
from transformers import RobertaTokenizer

from src.configs.cm_prompt import PromptSpec
from src.configs.labels_meld import LABEL_SPACE
from src.data.meld import MELDDataset, MELDComet, get_proper_loaders, role_dataset_train, role_dataset_test
from src.features.common_ground import cicero_get, get_common_ground
from src.features.tabtext import get_tab_text_feature, find_tab_text_feature
from src.models.cm_prompt_fix import CMPromptFIX
from src.training.engine import train_one_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld_pkl", type=str, required=True, help="MELD pickle path (原始meld pickle)")
    parser.add_argument("--comet_pkl", type=str, required=True, help="meld_features_comet.pkl path")
    parser.add_argument("--roberta_dir", type=str, required=True, help="roberta_large_tokenizer or roberta checkpoint dir")
    parser.add_argument("--train_role_csv", type=str, required=True)
    parser.add_argument("--test_role_csv", type=str, required=True)
    parser.add_argument("--cicero_train_csv", type=str, required=True)
    parser.add_argument("--cicero_test_csv", type=str, required=True)
    parser.add_argument("--tabtext_csv", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--ple", type=int, default=3)
    parser.add_argument("--plp", type=int, default=3)
    parser.add_argument("--pre", type=int, default=3)
    parser.add_argument("--prp", type=int, default=3)

    parser.add_argument("--save_dir", type=str, default="./checkpoints_cm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0) if device.type == "cuda" else None

    spec = PromptSpec(ple=args.ple, plp=args.plp, pre=args.pre, prp=args.prp)

    # tokenizer / model
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_dir)
    model = CMPromptFIX.from_pretrained(
        args.roberta_dir,
        ple=args.ple,
        plp=args.plp,
        pre=args.pre,
        prp=args.prp,
        emotion_token_ids=LABEL_SPACE.emotion_token_ids,
    ).to(device)

    # dataset / loaders
    dataset_train = MELDDataset(args.meld_pkl, split="train")
    dataset_test = MELDDataset(args.meld_pkl, split="test")

    train_loader, _, test_loader = get_proper_loaders(args.meld_pkl, batch_size=args.batch_size)

    # role maps
    role_train = role_dataset_train(args.train_role_csv)
    role_test = role_dataset_test(args.test_role_csv)

    # comet
    comet = MELDComet(args.comet_pkl)

    # cicero
    cicero_train = cicero_get(tokenizer, args.cicero_train_csv)
    cicero_test = cicero_get(tokenizer, args.cicero_test_csv)

    # tabtext
    tabtext_train = get_tab_text_feature(args.tabtext_csv, train=True)
    tabtext_test = get_tab_text_feature(args.tabtext_csv, train=False)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = None
    best_epoch = 0

    for e in range(args.epochs):
        train_info = train_one_epoch(
            e,
            dataset=dataset_train,
            role_map=role_train,
            cicero_tensor=cicero_train,
            comet=comet,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            tabtext_feature=tabtext_train,
            build_tabtext_fn=find_tab_text_feature,
            build_common_ground_fn=get_common_ground,
            spec=spec,
            device=device,
        )

        test_info = evaluate(
            e,
            dataset=dataset_test,
            role_map=role_test,
            cicero_tensor=cicero_test,
            comet=comet,
            tokenizer=tokenizer,
            model=model,
            dataloader=test_loader,
            tabtext_feature=tabtext_test,
            build_tabtext_fn=find_tab_text_feature,
            build_common_ground_fn=get_common_ground,
            spec=spec,
            device=device,
        )

        f1 = test_info["weighted_f1"]
        print(f"[epoch {e}] test weighted_f1 = {f1:.6f}")

        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_epoch = e
            save_path = f"{args.save_dir}/best"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved best checkpoint to: {save_path}")

    print(f"Best epoch: {best_epoch}, best weighted_f1: {best_f1}")


if __name__ == "__main__":
    main()
