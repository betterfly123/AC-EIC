import os
import torch
from utils import *
from model import *
from sklearn import metrics
from transformers import RobertaTokenizer, RobertaForMaskedLM, TrainingArguments, Trainer, BertTokenizer, BertForMaskedLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def train_or_eval_em_cm(epoch, feature, label2emotion, label2id, dataset, role, cicero, comet, tokenizer, model,
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
            if t % 10 == 0:
                print(f"cur_batch: {t}")
            t = t + 1
            vids = data[-1]
            for vid in vids:
                cur_comet = [comet.com[i][vid] for i in range(9)]  # x1,..x6, r1, r2, r3
                cur_emotions = dataset.videoLabels[vid]
                print(dataset.roberta4[vid])
                cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                cur_text4 = torch.FloatTensor(dataset.roberta4[vid])
                U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4
                print(U.shape)
                U = U.cuda()
                U = U.unsqueeze(0)
                conv = dataset.videoSentence[vid]
                length = len(conv)
                for i in range(length - 1):
                    sent = conv[i].replace('x92', "'")
                    cur_emotion = cur_emotions[i + 1]
                    cur_emotion = label2emotion[cur_emotion]
                    role_id = vid.split('_')
                    role_list = role[int(role_id[0])][int(role_id[1])][int(role_id[2])]
                    listener = role_list[i+1][2:-2]
                    cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                    cur_feature = find_tab_text_feature_em(listener, feature, tokenizer, U.shape[1])
                    com_r = torch.as_tensor(cicero[int(role_id[0])][int(role_id[1])][int(role_id[2])][i], dtype=torch.float32).cuda()
                    common_ground = get_common_ground(i, U, cur_com, role_list, feature, tokenizer)
                    target = "Target:" + sent
                    fix_prompt = "The possible emotional reaction of the " + listener + " in response to target is "
                    x = ' <mask>' * (ple + plp + 2) + ' ' + target + ' ' + fix_prompt + ' <mask>' * (1 + pre + prp)
                    y = ' <mask>' * (ple + plp + 2) + ' ' + target + ' ' + fix_prompt + cur_emotion + ' <mask>' * (pre + prp)
                    input = tokenizer(x, return_tensors="pt")
                    input_ids = input['input_ids'].cuda()
                    attention_mask = input['attention_mask'].cuda()
                    label = tokenizer(y, return_tensors="pt")["input_ids"]
                    label[:, 1:(ple + plp + 3)] = -100
                    label[:, (-pre - prp - 1):-1] = -100
                    ground_truth.append(cur_emotions[i + 1])
                    label = label.cuda()
                    _, loss, logits = model(U, common_ground, com_r, role_list, cur_feature, i, input_ids=input_ids,
                                            attention_mask=attention_mask, labels=label)
                    logits = logits[:, - ple - plp - 2]
                    loss.backward()
                    logits = logits.data.cpu()
                    pred = int(logits.argmax(-1))
                    preds.append(label2id[pred] if pred in label2id.keys() else 0)
            optimizer.step()
            optimizer.zero_grad()
        model.save_pretrained("./emorynlp_" + str(epoch))
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
                    cur_emotions = dataset.videoLabels[vid]
                    cur_text1 = torch.FloatTensor(dataset.roberta1[vid])
                    cur_text2 = torch.FloatTensor(dataset.roberta2[vid])
                    cur_text3 = torch.FloatTensor(dataset.roberta3[vid])
                    cur_text4 = torch.FloatTensor(dataset.roberta4[vid])

                    U = (cur_text1 + cur_text2 + cur_text3 + cur_text4) / 4
                    U = U.cuda()
                    U = U.unsqueeze(0)
                    conv = dataset.videoSentence[vid]
                    length = len(conv)
                    for i in range(length - 1):
                        sent = conv[i].replace('x92', "'")
                        cur_emotion = cur_emotions[i + 1]
                        cur_emotion = label2emotion[cur_emotion]
                        role_id = vid.split('_')
                        role_list = role[int(role_id[0])][int(role_id[1])][int(role_id[2])]
                        listener = role_list[i + 1][2:-2]
                        cur_com = torch.tensor([cur_comet[j] for j in range(9)])
                        cur_feature = find_tab_text_feature_em(listener, feature, tokenizer, U.shape[1])
                        com_r = torch.as_tensor(cicero[int(role_id[0])][int(role_id[1])][int(role_id[2])][i], dtype=torch.float32).cuda()
                        common_ground = get_common_ground(i, U, cur_com, role_list, feature, tokenizer)
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
                        ground_truth.append(cur_emotions[i + 1])
                        label = label.cuda()
                        _, loss, logits = model(U, common_ground, com_r, role_list, cur_feature, i, input_ids=input_ids,
                                                attention_mask=attention_mask, labels=label)
                        logits = logits[:, - ple - plp - 2]
                        logits = logits.data.cpu()
                        pred = int(logits.argmax(-1))
                        preds.append(label2id[pred] if pred in label2id.keys() else 0)
        fscore = metrics.f1_score(ground_truth, preds, average='weighted')
        print(classification_report(ground_truth, preds, digits=4))
    return (results, losses, None, None) if train else (results, None, fscore, predict)

def em_cm_prompt_feature_TP(path, label2emotion, label2id, ple, plp, pre, prp, epoch, batch_size, lr, l2):

    tokenizer = RobertaTokenizer.from_pretrained('./roberta_large_tokenizer/')
    model = EmCMPromptFIX.from_pretrained('./roberta_large_tokenizer/')
    torch.cuda.set_device(0)
    dataset = EmoryNLPRobertaCometDataset(path=path)
    role, cicero = role_em(tokenizer, 'train')
    role_test, cicero_test = role_em(tokenizer, 'test')
    comet = MELDComet('./Datasets/emorynlp/emorynlp_features_comet.pkl')
    train_loader, valid_loader, test_loader = get_proper_loaders_erm(path, batch_size=batch_size)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    best_fscore = None
    best_epoch = 0
    train_feature = get_tab_text_feature_emr(train=True)
    test_feature = get_tab_text_feature_emr(train=False)
    fscore_list = []
    for e in range(epoch):
        train_results, train_losses, _, _ = train_or_eval_em_cm(e, train_feature, label2emotion, label2id,
                                                                                    dataset, role, cicero, comet, tokenizer,
                                                                                    model, optimizer, train_loader,
                                                                                    ple=ple, plp=plp, pre=pre, prp=prp,
                                                                                    train=True)

        test_results, test_losses, f_score, pred = train_or_eval_em_cm(e, test_feature, label2emotion,
                                                                                           label2id, dataset, role_test,
                                                                                           cicero_test, comet, tokenizer, model,
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
emo_path = './Datasets/emorynlp/emorynlp_features_roberta.pkl'
em_cm_prompt_feature_TP(emo_path, label2emotion, id2label, ple=3, plp=3, pre=3, prp=3, epoch=60, batch_size=32, lr=1e-5, l2=1e-2)

