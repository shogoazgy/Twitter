import json
#from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
#from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, f1_score
#from sklearn.decomposition import PCA
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import BertJapaneseTokenizer
from transformers import BertModel
#from lightgbm import LGBMClassifier
import pickle

with open('/home/narita/data2/test_q.json') as f:
  test = json.load(f)
with open('/home/narita/data2/train_q.json') as f:
  train = json.load(f)
with open('/home/narita/data2/exam_data1.json') as f:
  data = json.load(f)

data_dict = {}

for d in data:
  data_dict[d[0]] = d[1]

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

train_sum_list = []
train_first_list = []
df_list = []
#bert_model.to(device)
def get_vector(word):
    x = tokenizer(word, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = bert_model(**x)
    return outputs.last_hidden_state[0][1].detach().nunpy()
for i, t in enumerate(tqdm(train)):
    for c in t['candidates']:
        if c == t['ans_id']:
            f = 1
        else:
            f = 0
        x = None
        for j, s in enumerate(data_dict[c].split('。')):
            if j > 4:
                break
            xt = get_vector(data_dict[t['title_id']] + ' [SEP] ' + s)
            if x is not None:
                x = x + xt
            else:
                x = xt
            if j == 0:
                train_first_list.append(x)
        train_sum_list.append(x)
        if f:
            df_list.append([t['title_id'], 1])
        else:
            df_list.append([t['title_id'], 0])
df = pd.DataFrame(df_list, columns=['id', 'y'])

with open('/content/drive/MyDrive/train_list_sum.pickle', 'wb') as f:
    pickle.dump(train_sum_list, f)
with open('/content/drive/MyDrive/train_list_first.pickle', 'wb') as f:
    pickle.dump(train_first_list, f)
with open('/content/drive/MyDrive/df.pickle', 'wb') as f:
    pickle.dump(df, f)