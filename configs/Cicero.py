import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)

tokenizer = AutoTokenizer.from_pretrained("./cheakpoint")
model = AutoModelForSeq2SeqLM.from_pretrained("./cheakpoint")
model.cuda()

relation = 'What is the possible emotional reaction of the listener in response to target? <sep> target: '
relation_a = 'What is the possible emotional reaction of the '
relation_b = ' in response to target? <sep> target: '
sep = ' <sep> context: '
utt = ', <utt> '

file_path = './Datasets/MELD/test_sent_emo.csv'
data = pd.read_csv(file_path)
dialog = []
role = []
k = 0
for row in range(data.shape[0]):
    k += 1
    if data['Utterance_ID'][row] == 0:
        dialog.clear()
        role.clear()
    dialog.append(data['Utterance'][row])
    role.append(data['Speaker'][row])
    tmp = ''
    for i in range(len(dialog)):
        if i == 0:
            tmp = role[i] + ': ' + dialog[i]
        else:
            tmp = tmp + utt + role[i] + ': ' + dialog[i]
    # listener = ''
    # if row < data.shape[0]-1 and data['Utterance_ID'][row+1] != 0:
    #     listener = data['Speaker'][row+1]
    # else:
    #     continue
    x = relation + role[len(dialog) - 1] + ':' + dialog[len(dialog) - 1] + sep + tmp
    # x = relation_a + listener + relation_b + role[len(dialog) - 1] + ':' + dialog[len(dialog) - 1] + sep + tmp
    input = tokenizer(x, return_tensors="pt")
    input_ids = input['input_ids'].cuda()

    outputs = model.generate(input_ids)
    with open('CICERO_test.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([tokenizer.decode(outputs[0], skip_special_tokens=True)])





