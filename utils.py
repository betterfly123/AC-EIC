import pickle
import random
import re
import numpy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

prompt = "listener's emotion is <mask>."
common_feature_presets = ''
emotionids = [7974, 2755, 2490, 17437, 5823, 30883, 6378]
id2label = {7974: 0, 2755: 1, 2490: 2, 17437: 3, 5823: 4, 30883: 5, 6378: 6}
emotion2label = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
label2emotion = {}

for key in emotion2label.keys():
    label2emotion[emotion2label[key]] = key
label2id = {emotionids[i]: i for i in range(len(emotionids))}



class MELDDataset:
    def __init__(self, path, split='train'):

        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.emotion_labels[vid])), \
               torch.LongTensor(self.emotion_labels[vid]), vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        d = pd.DataFrame(data)
        return [list(d[i]) for i in d]


class MELDComet:
    def __init__(self, path):
        self.com = pickle.load(open(path, 'rb'))
        """self.x1, self.x2, self.x3, self.x4, self.x5, self.x6,\
        self.o1, self.o2, self.o3 = pickle.load(open(path, 'rb'))"""
        '''
        ['xIntent', 'xAttr', 'xNeed', 'xWant', 'xEffect', 'xReact', 'oWant', 'oEffect', 'oReact']
        '''


def get_train_valid_sampler(trainset, v_size=0.1):
    # divide the dataset and randomize
    size = len(trainset)
    idx = list(range(size))
    split = int(v_size * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_proper_loaders(path, batch_size=16, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path, 'train')
    testset = MELDDataset(path, 'test')
    validset = MELDDataset(path, 'valid')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=validset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def roleDataset():
    file = open("./Datasets/MELD/train_sent_emo.csv", 'rb')
    df = pd.read_csv(file)
    data = np.array(df)
    number = len(set(data[:, 5]))
    # print(number)
    all_role = [[] for i in range(number + 1)]
    idx = 0
    for i in range(number):
        while True:
            # print(data[idx][5], data[idx][2])
            all_role[data[idx][5]].append(data[idx][2])
            if idx + 1 == int(data.shape[0]):
                break
            if data[idx + 1][5] != data[idx][5]:
                idx = idx + 1
                break
            idx = idx + 1
        # print("char:", all_role)
    return all_role
    # print("all_role:", all_role)


def role_testDataset():
    file = open("./Datasets/MELD/test_sent_emo.csv", 'rb')
    df = pd.read_csv(file)
    data = np.array(df)
    number = len(set(data[:, 5]))
    # print(number)
    all_role = [[] for i in range(number + 1)]
    idx = 0
    for i in range(number):
        while True:
            # print(data[idx][5], data[idx][2])
            all_role[data[idx][5]].append(data[idx][2])
            if idx + 1 == int(data.shape[0]):
                break
            if data[idx + 1][5] != data[idx][5]:
                idx = idx + 1
                break
            idx = idx + 1
        # print("char:", all_role)
    return all_role
    # print("all_role:", all_role)


def get_features(train=True):

    with open('./Datasets/MELD/feature/char_features.txt', 'r', encoding='utf-8') as f:
        sen_data = f.read()
    sentiment_list = sen_data.split('\n')
    sentiment_list = sentiment_list[1:]
    train_feature = {}
    test_feature = {}
    cs = ''
    name = sentiment_list[0].split('\t')[3]
    for i in sentiment_list:
        tmp = i.split('\t')
        cs = cs + " " + tmp[0]
        if name != tmp[3]:
            sp = cs.split(' ')
            train_cs = ''
            test_cs = ''
            train_size = int(len(sp) * 0.7)
            for j in range(0, len(sp)):
                if j == 0:
                    continue
                if j < train_size:
                    train_cs = train_cs + " " + sp[j]
                else:
                    test_cs = test_cs + " " + sp[j]
            train_feature[name] = train_cs
            test_feature[name] = test_cs
            name = tmp[3]
            cs = ''
    if train:
        return train_feature
    else:
        return test_feature


def find_feature(listener, feature, tokenizer):

    if listener in feature:
        listener_feature = feature[listener]
    else:
        listener_feature = common_feature_presets
    cur_feature = tokenizer(listener_feature, return_tensors="pt")['input_ids']
    cur_feature = cur_feature.view(-1)
    cur_feature = cur_feature.numpy().tolist()
    while len(cur_feature) <= 500:
        cur_feature.append(0)
    cur_feature = torch.FloatTensor(cur_feature).cuda()
    return cur_feature


def get_tab_text_feature(train=True):
    file = open("./Datasets/MELD/feature/AllFeature.CSV", 'rb')
    df = pd.read_csv(file)
    data = np.array(df)
    number = np.shape(data)[0]
    train_size = int(number * 0.7)
    train_feature = {}
    test_feature = {}
    train_flist = []
    test_flist = []
    name = data[0][1]
    count = 0
    for i in range(number):
        newname = data[i][1]
        if name != newname:
            train_feature[name] = train_flist
            test_feature[name] = test_flist
            count = 1
            train_flist = []
            test_flist = []
            TabText = data[i][0]
            name = newname
            train_flist.append(TabText)
        else:
            count = count + 1
            TabText = data[i][0]
            if count < train_size:
                train_flist.append(TabText)
            else:
                test_flist.append(TabText)
    train_feature[name] = train_flist
    test_feature[name] = test_flist
    if train:
        return train_feature
    else:
        return train_feature


def find_tab_text_feature(listener, feature, tokenizer, k):
    if listener in feature:
        listener_feature = []
        tmp_feature = []
        tmp = feature[listener]
        count = max(int(len(tmp) / k), int(1))
        num = 0
        for i in range(k):
            f_add = ''
            if num < len(tmp):
                for j in range(num, num + count):
                    f_add = f_add + tmp[j]
            num = num + count
            cur_feature = tokenizer(f_add, return_tensors="pt")['input_ids']
            cur_feature = cur_feature.view(-1)
            cur_feature = cur_feature.numpy().tolist()
            if len(cur_feature) <= 1024:
                [cur_feature.append(0) for i in range(1024-len(cur_feature))]
            else:
                cur_feature = cur_feature[:1024]
            tmp_feature.append(cur_feature)
        listener_feature.append(tmp_feature)
    else:
        listener_feature = [[[0] * 1024] * k]
    cur_feature = torch.FloatTensor(listener_feature).cuda()
    return cur_feature


def cicero_get(tokenizer, train=True):
    if train==True:
        file_path = './Datasets/CICERO/CICERO_MELD_train.csv'
    else:
        file_path = './Datasets/CICERO/CICERO_MELD_test.csv'
    data = pd.read_csv(file_path, names=['knowledge'])
    knowledge_list = []
    for i in range(data.shape[0]):
        knowledge_list.append(data['knowledge'][i])
    k = tokenizer(knowledge_list, return_tensors="pt", max_length=512, padding='max_length')['input_ids']
    return knowledge_list


def cur_info(conv, speaker):
    cur_inter_cause_effect_mask = numpy.zeros((len(conv), len(conv)))
    cur_intra_cause_effect_mask = numpy.zeros((len(conv), len(conv)))
    cur_intra_cause_effect_mask[0][0] = 1
    for i in range(len(conv)):
        cur_speaker = speaker[i]
        j = i - 1
        while j >= 0:
            if speaker[j] == cur_speaker:
                cur_intra_cause_effect_mask[i][j] = 1
            else:
                cur_inter_cause_effect_mask[i][j] = 1
            j -= 1
    return cur_intra_cause_effect_mask, cur_inter_cause_effect_mask


def get_common_ground(id, U, comet, role, feature, tokenizer):
    cm_list = []
    cur_speaker = role[id + 1]
    for i in range(id+1):
        utt = U[0][i].cpu()
        if cur_speaker == role[i]:
            knowledgew = comet[8, i, :].reshape(1 * 768)
        else:
            knowledgew = comet[2, i, :].reshape(1 * 768)

        cm = torch.cat((utt, knowledgew), 0)
        cm_list.append(cm.cuda())
    return cm_list


# emorynlp

common_feature_presets = ''
emotionids_erm = [5823, 7758, 7053, 7974, 17437, 2247, 2490]
id2label_erm = {5823: 0, 7758: 1, 7053: 2, 7974: 3, 17437: 4, 2247: 5,  2490: 6}
emotion2label_erm = {'joy': 0, 'mad': 1, 'peaceful': 2, 'neutral': 3, 'sadness': 4, 'powerful': 5, 'fear': 6}
label2emotion_erm = {}
for key in emotion2label_erm.keys():
    label2emotion_erm[emotion2label_erm[key]] = key
label2id_erm = {emotionids_erm[i]: i for i in range(len(emotionids_erm))}

def label_count(pred):
    for i in range(len(label_token)):
        tmp = label_token[i]
        for j in range(len(tmp)):
            if pred == tmp[j]:
                return i
    return 0

for key in emotion2label.keys():
    label2emotion[emotion2label[key]] = key
label2id = {emotionids[i]: i for i in range(len(emotionids))}

class EmoryNLPRobertaCometDataset:
    def __init__(self, path, split='train'):

        '''
        label index mapping =  {'Joyful': 0, 'Mad': 1, 'Peaceful': 2, 'Neutral': 3, 'Sad': 4, 'Powerful': 5, 'Scared': 6}
        '''

        self.videoSpeakers, self.videoLabels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.videoSentence, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [list(dat[i]) for i in dat]

def get_proper_loaders_erm(path, batch_size=32,  num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset(path, 'train')
    testset = EmoryNLPRobertaCometDataset(path,  'test')
    #validset = EmoryNLPRobertaCometDataset(path, 'valid')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, None, test_loader

def cicero_em_get(tokenizer, train=True):
    if train==True:
        file_path = './Datasets/CICERO/CICERO_emorynlp_train.csv'
    else:
        file_path = './Datasets/CICERO/CICERO_emorynlp_test.csv'
    data = pd.read_csv(file_path, names=['knowledge'])
    knowledge_list = []
    for i in range(data.shape[0]):
        knowledge_list.append(data['knowledge'][i])
    k = tokenizer(knowledge_list, return_tensors="pt", max_length=512, padding='max_length')['input_ids']
    return k

def get_tab_text_feature_emr(train=True):
    file = open("./Datasets/EmoryNLP/feature/AllFeature.CSV", 'rb')
    df = pd.read_csv(file)
    data = np.array(df)
    number = np.shape(data)[0]
    train_size = int(number * 0.7)
    train_feature = {}
    test_feature = {}
    train_flist = []
    test_flist = []
    name = data[0][1]
    count = 0
    for i in range(number):
        newname = data[i][1]
        if name != newname:
            train_feature[name] = train_flist
            test_feature[name] = test_flist
            count = 1
            train_flist = []
            test_flist = []
            TabText = data[i][0]
            name = newname
            train_flist.append(TabText)
        else:
            count = count + 1
            TabText = data[i][0]
            if count < train_size:
                train_flist.append(TabText)
            else:
                test_flist.append(TabText)
    train_feature[name] = train_flist
    test_feature[name] = test_flist
    if train:
        return train_feature
    else:
        return train_feature

def role_em(tokenizer, label):
    if label == 'train':
        file = open("./Datasets/emorynlp/emorynlp_train_final.csv", 'rb')
    else:
        file = open("./Datasets/emorynlp/emorynlp_test_final.csv", 'rb')
    df = pd.read_csv(file)
    data = np.array(df)
    number0 = 4
    number1 = 24
    number2 = 20
    all_role = [[[[] for k in range(number2 + 1)] for j in range(number1 + 2)] for i in range(number0 + 1)]

    if label == 'train':
        file_path = './Datasets/CICERO/CICERO_emorynlp_train.csv'
    else:
        file_path = './Datasets/CICERO/CICERO_emorynlp_test.csv'
    data1 = pd.read_csv(file_path, names=['knowledge'])
    knowledge_list = []
    for i in range(data1.shape[0]):
        knowledge_list.append(data1['knowledge'][i])
    k = tokenizer(knowledge_list, return_tensors="pt", max_length=512, padding='max_length')['input_ids']
    cicero = [[[[] for k in range(number2 + 1)] for j in range(number1 + 2)] for i in range(number0 + 1)]
    for i in range(data.shape[0]):
        all_role[data[i][5]][data[i][6]][data[i][3]].append(data[i][1])
        cicero[data[i][5]][data[i][6]][data[i][3]].append(k[i])

    return all_role, cicero

def find_tab_text_feature_em(listener, feature, tokenizer, k):
    ok = 0
    for key in feature:
        if re.search(key, listener):
            listener = key
            ok = 1
            break
    if ok == 1:
        listener_feature = []
        tmp_feature = []
        tmp = feature[listener]
        count = max(int(len(tmp) / k), int(1))
        num = 0
        for i in range(k):
            f_add = ''
            if num < len(tmp):
                for j in range(num, num + count):
                    f_add = f_add + tmp[j]
            num = num + count
            cur_feature = tokenizer(f_add, return_tensors="pt")['input_ids']
            cur_feature = cur_feature.view(-1)
            cur_feature = cur_feature.numpy().tolist()
            if len(cur_feature) <= 1024:
                [cur_feature.append(0) for i in range(1024-len(cur_feature))]
            else:
                cur_feature = cur_feature[:1024]
            tmp_feature.append(cur_feature)
        listener_feature.append(tmp_feature)
    else:
        listener_feature = [[[0] * 1024] * k]
    cur_feature = torch.FloatTensor(listener_feature).cuda()
    return cur_feature

#CPED

common_feature_presets = ''
emotionids_cped = [1372, 6161, 11956, 1313, 7974, 6378, 17437, 2490, 37018, 30883, 40788, 3915, 2430]
id2label_cped = {1372: 0, 6161: 1, 11956: 2, 1313: 3, 7974: 4, 6378: 5, 17437: 6, 2490: 7, 37018: 8, 30883: 9, 40788: 10, 3915: 11, 2430: 12}
emotion2label_cped = {'happy': 0, 'grateful': 1, 'relaxed': 2, 'positive': 3, 'neutral': 4, 'anger': 5, 'sadness': 6, 'fear': 7, 'depress': 8, 'disgust': 9, 'astonished': 10, 'worried': 11, 'negative': 12}
label2emotion_cped = {}
for key in emotion2label_cped.keys():
    label2emotion_cped[emotion2label_cped[key]] = key
label2id_cped = {emotionids_cped[i]: i for i in range(len(emotionids_cped))}

def emotion_translate(cur_emotion):
    if cur_emotion == 'negative-other':
        cur_emotion = 'negative'
    elif cur_emotion == 'positive-other':
        cur_emotion = 'positive'
    else:
        cur_emotion = cur_emotion
    return cur_emotion

class CPEDRobertaDataset:
    def __init__(self, path, split='train'):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]), \
               torch.FloatTensor(self.roberta2[vid]), \
               torch.FloatTensor(self.roberta3[vid]), \
               torch.FloatTensor(self.roberta4[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [list(dat[i]) for i in dat]


def cicero_get_cped(tokenizer, label):
    if label == 'train':
        file = open("./Datasets/CPED/train.csv", encoding='ISO-8859-1')
    else:
        file = open("./Datasets/CPED/test.csv", encoding='ISO-8859-1')
    df = pd.read_csv(file)
    dialog_id = []
    id = df.iloc[0]['Dialogue_ID']
    for i, row in df.iterrows():
        if id != row['Dialogue_ID']:
            id = row['Dialogue_ID']
        else:
            dialog_id.append(row['Dialogue_ID'])


    if label == 'train':
        file_path = './Datasets/CICERO/CICERO_CPED_train.csv'
    else:
        file_path = './Datasets/CICERO/CICERO_CPED_test.csv'
    data1 = pd.read_csv(file_path, names=['knowledge'])
    knowledge_list = []
    for i in range(data1.shape[0]):
        knowledge_list.append(data1['knowledge'][i])
    k = tokenizer(knowledge_list, return_tensors="pt", max_length=512, padding='max_length')['input_ids']
    cicero = []
    knowledge = []
    tmp_id = dialog_id[0]
    count = 0
    for i in range(1, len(dialog_id)):
        if tmp_id != dialog_id[i]:
            tmp_id = dialog_id[i]
            cicero.append(knowledge)
            knowledge = []
            knowledge.append(k[count])
        else:
            knowledge.append(k[count])
        count += 1
    cicero.append(knowledge)
    return cicero

class Comet:
    def __init__(self, path):
        self.com = pickle.load(open(path, 'rb'))
        """self.x1, self.x2, self.x3, self.x4, self.x5, self.x6,\
        self.o1, self.o2, self.o3 = pickle.load(open(path, 'rb'))"""
        '''
        ['xIntent', 'xAttr', 'xNeed', 'xWant', 'xEffect', 'xReact', 'oWant', 'oEffect', 'oReact']
        '''

def get_proper_loaders_cped(trainset, testset, batch_size=32,  num_workers=0, pin_memory=False):
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, None, test_loader

def get_tab_text_feature_cped():
    file = open("./Datasets/CPED/feature/feature.CSV", encoding='ISO-8859-1')
    df = pd.read_csv(file)
    data = np.array(df)
    number = np.shape(data)[0]
    feature = {}
    f_list = []
    name = data[0][0]
    for i in range(number):
        if name != data[i][0]:
            feature[name] = f_list
            TabText = data[i][1] + ': ' + data[i][2]
            f_list = []
            name = data[i][0]
            f_list.append(TabText)
        else:
            TabText = data[i][1] + ': ' + data[i][2]
            f_list.append(TabText)
    feature[name] = f_list
    return feature

def find_tab_text_feature_em(listener, feature, tokenizer, k):
    ok = 0
    for key in feature:
        if re.search(key, listener):
            listener = key
            ok = 1
            break
    if ok == 1:
        listener_feature = []
        tmp_feature = []
        tmp = feature[listener]
        f_add = ''
        for i in range(len(tmp)):
            f_add = f_add + tmp[i]
        cur_feature = tokenizer(f_add, max_length=1024, padding='max_length', return_tensors="pt")['input_ids']
        cur_feature = cur_feature.view(-1)
        cur_feature = cur_feature.numpy().tolist()
        for i in range(k):
            tmp_feature.append(cur_feature)
        listener_feature.append(tmp_feature)
    else:
        # listener_feature = [0] * 1024
        listener_feature = [[[0] * 1024] * k]
    cur_feature = torch.FloatTensor(listener_feature).cuda()
    return cur_feature
