import os

import torch

from utils import *
from model import *
from sklearn import metrics
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments, Trainer, BertTokenizer, BertForMaskedLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# PCGEI
def train_or_eval_cped(epoch, feature, label2emotion, label2id, dataset, cicero, comet, tokenizer, model,
                      optimizer, dataloader,  ple=3, plp=3, pre=3, prp=3, train=True):
    if train:
        print("-------Train start-------")
        print(f"epoch: {epoch}")
        model.train()
        results = []
        losses = []
        ground_truth = []
        preds = []
        t = 0
        total_batch = len(dataloader)
        print(f"totalbatch: {total_batch}")
        for data in dataloader:
            if t % 50 == 0:
                print(f"cur_batch: {t}")
            t = t + 1
            vids = data[-1]
            for vid in vids:
                cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                cur_emotions = dataset.labels[vid]
                cur_listeners = dataset.speakers[vid]
                cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                cur_text4 = torch.FloatTensor(dataset.roberta4[vid])
                U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4
                U = U.cuda()
                U = U.unsqueeze(0)
                conv = dataset.sentences[vid]
                id = len(conv) - 2
                sent = conv[id]
                cur_emotion = cur_emotions[id + 1]
                listener = cur_listeners[id + 1]
                cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                cur_feature = find_tab_text_feature_em(listener, feature, tokenizer, U.shape[1])
                com_r = torch.as_tensor(cicero[vid][id], dtype=torch.float32).cuda()
                common_ground = get_common_ground(id, U, cur_com, cur_listeners, feature, tokenizer)
                target = "Target:" + sent
                fix_prompt = "The possible emotional reaction of the " + listener + " in response to target is "
                cur_emotion = emotion_translate(cur_emotion)
                x = ' <mask>' * (ple + plp + 2) + ' ' + target + ' ' + fix_prompt + ' <mask>' * (1 + pre + prp)
                y = ' <mask>' * (ple + plp + 2) + ' ' + target + ' ' + fix_prompt + cur_emotion + ' <mask>' * (pre + prp)
                input = tokenizer(x, return_tensors="pt")
                input_ids = input['input_ids'].cuda()
                attention_mask = input['attention_mask'].cuda()
                label = tokenizer(y, return_tensors="pt")["input_ids"]
                label[:, 1:(ple + plp + 3)] = -100
                label[:, (-pre - prp - 1):-1] = -100
                ground_truth.append(int(emotion2label_cped[cur_emotion]))
                label = label.cuda()
                _, loss, logits = model(U, common_ground, com_r, cur_listeners, cur_feature, id, input_ids=input_ids,
                                        attention_mask=attention_mask, labels=label)
                logits = logits[:, - ple - plp - 2]
                loss.backward()
                logits = logits.data.cpu()
                pred = int(logits.argmax(-1))
                # print("pred:", pred, "label", int(emotion2label_cped[cur_emotion]))
                # print("loss:", loss)
                preds.append(label2id[pred] if pred in label2id.keys() else 4)
            optimizer.step()
            optimizer.zero_grad()
        model.save_pretrained("./cped_" + str(epoch))
        results.append(precision_recall_fscore_support(ground_truth, preds))
        print(f"epoch: {epoch} loss: {loss}")
        _, _, f1_score, _ = precision_recall_fscore_support(ground_truth, preds)
    else:
        print("-------Test start-------")
        model.eval()
        results = []
        predict = {}
        with torch.no_grad():
            ground_truth = []
            preds = []
            t = 0
            total_batch = len(dataloader)
            print(f"totalbatch: {total_batch}")
            for data in dataloader:
                if t % 10 == 0:
                    print(f"cur_batch: {t}")
                t = t + 1
                vids = data[-1]
                for vid in vids:
                    cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                    cur_emotions = dataset.labels[vid]
                    cur_listeners = dataset.speakers[vid]
                    cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                    cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                    cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                    cur_text4 = torch.FloatTensor(dataset.roberta4[vid])
                    U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4
                    U = U.cuda()
                    U = U.unsqueeze(0)
                    conv = dataset.sentences[vid]
                    id = len(conv) - 2
                    sent = conv[id]
                    cur_emotion = cur_emotions[id + 1]
                    listener = cur_listeners[id + 1]
                    cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                    cur_feature = find_tab_text_feature_em(listener, feature, tokenizer, U.shape[1])
                    com_r = torch.as_tensor(cicero[vid - 9020][id], dtype=torch.float32).cuda()
                    common_ground = get_common_ground(id, U, cur_com, cur_listeners, feature, tokenizer)
                    cur_emotion = emotion_translate(cur_emotion)
                    target = "Target:" + sent
                    fix_prompt = "The possible emotional reaction of the " + listener + " in response to target is "
                    x = ' <mask>' * (ple + plp + 2) + ' ' + target + ' ' + fix_prompt + ' <mask>' * (1 + pre + prp)
                    y = ' <mask>' * (
                                ple + plp + 2) + ' ' + target + ' ' + fix_prompt + cur_emotion + ' <mask>' * (
                                    pre + prp)
                    input = tokenizer(x, return_tensors="pt")
                    input_ids = input['input_ids'].cuda()
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(ple + plp + 3)] = -100
                    label[:, (-pre - prp - 1):-1] = -100
                    ground_truth.append(int(emotion2label_cped[cur_emotion]))
                    label = label.cuda()
                    _, loss, logits = model(U, common_ground, com_r, cur_listeners, cur_feature, id, input_ids=input_ids,
                                            attention_mask=attention_mask, labels=label)
                    logits = logits[:, - ple - plp - 2]
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 4)
        fscore = metrics.f1_score(ground_truth, preds, average='weighted')
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses, None, None) if train else (results, None, fscore, predict)

def cped_cm_prompt_feature_TP(path, label2emotion, label2id, ple, plp, pre, prp, epoch, batch_size, lr, l2):

    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer/')
    model = CpedCMPromptFIX.from_pretrained('./roberta_large_tokenizer/')
    torch.cuda.set_device(0)
    trainset = CPEDRobertaDataset(path, 'train')
    testset = CPEDRobertaDataset(path, 'test')
    cicero = cicero_get_cped(tokenizer, 'train')
    cicero_test = cicero_get_cped(tokenizer, 'test')
    comet = Comet('./Datasets/CPED/cped_features_comet.pkl')
    train_loader, valid_loader, test_loader = get_proper_loaders_cped(trainset, testset, batch_size=batch_size)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    best_fscore = None
    best_epoch = 0
    feature = get_tab_text_feature_cped()
    fscore_list = []
    for e in range(epoch):
        train_results, train_losses, _, _ = train_or_eval_cped(e, feature, label2emotion, label2id,
                                                                                    trainset, cicero, comet, tokenizer,
                                                                                    model, optimizer, train_loader,
                                                                                    ple=ple, plp=plp, pre=pre, prp=prp,
                                                                                    train=True)

        test_results, test_losses, f_score, pred = train_or_eval_cped(e, feature, label2emotion,
                                                                                           label2id, testset, cicero_test,
                                                                                           comet, tokenizer, model,
                                                                                           optimizer, test_loader,
                                                                                           ple=ple, plp=plp, pre=pre,
                                                                                           prp=prp, train=False)

        print("epoch:", e, "f_score:", f_score)
        fscore_list.append(f_score)
        if not best_fscore or f_score > best_fscore:
            best_fscore = f_score
            best_epoch = e
    print("best_epoch is :", best_epoch)
    print("best_fscore is :", best_fscore)

    pass


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
emo_path = './Datasets/CPED/cped_features_roberta_all.pkl'

# PCGEI
cped_cm_prompt_feature_TP(emo_path, label2emotion_cped, id2label_cped, ple=3, plp=3, pre=3, prp=3, epoch=60, batch_size=32, lr=1e-5, l2=1e-2)



