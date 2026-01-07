PYTHONPATH=. python -m src.scripts.run_meld_cm \
  --meld_pkl ./Datasets/MELD/meld.pkl \
  --comet_pkl ./meld_features_comet.pkl \
  --roberta_dir ./roberta_large_tokenizer \
  --train_role_csv ./Datasets/MELD/train_sent_emo.csv \
  --test_role_csv ./Datasets/MELD/test_sent_emo.csv \
  --cicero_train_csv ./Datasets/CICERO/CICERO.csv \
  --cicero_test_csv ./Datasets/CICERO/CICERO_test.csv \
  --tabtext_csv ./Datasets/MELD/feature/tabText/AllFeature2.CSV \
  --epochs 3 --batch_size 1 --lr 1e-5 --weight_decay 0.0
