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

# relation = ['What is the possible emotional reaction of the listener in response to target? <sep> target: ',
#             'What is or could be the cause of target? <sep> target: ',
#             'What subsequent event happens or could happen following the target? <sep> target: ',
#             'What is or could be the motivation of target? <sep> target: ']
relation = ['What is the possible emotional reaction of the listener in response to target? <sep> target: ']
sep = ' <sep> context: '
utt = ', <utt> '

file_path = './Datasets/CPED/data/test.csv'
file = open(file_path, encoding='ISO-8859-1')
data = pd.read_csv(file)
dialog = []
role = []
dialogue_id = data.iloc[0]['Dialogue_ID']
k = 0
for j, row in data.iterrows():
    k += 1
    if row['Dialogue_ID'] != dialogue_id:
        dialog = []
        role = []
        dialogue_id = row['Dialogue_ID']
    dialog.append(row['Utterance'])
    role.append(row['Speaker'])
    tmp = ''
    for i in range(len(dialog)):
        if i == 0:
            tmp = role[i] + ': ' + dialog[i]
        else:
            tmp = tmp + utt + role[i] + ': ' + dialog[i]
    for i in range(len(relation)):
        x = relation[i] + role[len(dialog) - 1] + ': ' + dialog[len(dialog) - 1] + sep + tmp
        input = tokenizer(x, return_tensors="pt")
        input_ids = input['input_ids'].cuda()
        outputs = model.generate(input_ids)
        with open('CICERO_CPED_test.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([tokenizer.decode(outputs[0], skip_special_tokens=True)])





