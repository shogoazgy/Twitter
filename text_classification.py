import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import torch
import pytorch_lightning as pl
import random
import os
import torch.nn as nn
import unicodedata
import umap
import optuna.integration.lightgbm as lgb
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, f1_score
from lightgbm import LGBMClassifier
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from fastprogress.fastprogress import  progress_bar
from sklearn.decomposition import PCA

df_train = pd.read_csv('/home/narita/Twitter/sigtrain.csv')
df_test = pd.read_csv('/home/narita/Twitter/sigtest.csv')
df_all = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

len_train = len(df_train)

def goal_min(x):
    if x == '100000+':
        return 100000
    return int(x.split('-')[0])

def goal_max(x):
    if x == '100000+':
        return 200000
    return int(x.split('-')[1])

def html_len(x):
    return len(x)

#目標最小
df_all['goal_min'] = df_all['goal'].apply(goal_min)

#目標最大
df_all['goal_max'] = df_all['goal'].apply(goal_max)

#htmlの長さ
#df_all['html_len'] = df_all['html_content'].apply(html_len)

#期間と目標最小の比
df_all['dura_goal_rate'] = [df_all['duration'][i] / df_all['goal_min'][i] for i in range(len(df_all['state']))]

#df_t = pd.get_dummies(df_all.drop(columns=['id', 'goal', 'html_content', 'category2']))
#ラベルエンコーディング
cat_cols = ['country', 'category1', 'category2']
le = LabelEncoder()
for col in cat_cols:
    df_all[col] = le.fit_transform(df_all[col].values)

#htmlをbs4で変換
from bs4 import BeautifulSoup
def html_parse(x):
    return BeautifulSoup(x, "html.parser")
df_all['html_content'] = df_all['html_content'].apply(html_parse)

#タグ全部の個数
#tags = ['html', 'head', 'body', 'title', 'isindex', 'base', 'meta', 'link', 'script', 'hn', 'hr', 'br', 'p', 'center', 'div', 'pre', 'blockquote', 'address', 'font', 'basefont', 'i', 'tt', 'b', 'u', 'strike', 'big', 'small', 'sub', 'sup', 'em', 'strong', 'code', 'samp', 'kbd', 'var', 'cite', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'table', 'tr', 'th', 'td', 'caption', 'a', 'img', 'map', 'area', 'form', 'input', 'select', 'option', 'textarea', 'applet', 'param', 'frameset', 'frame', 'noframes']
#tags = ['a', 'img', 'li', 'ul', 'br', 'p', 'div', 'i']
tags = ['a', 'img']
def tag_size(x):
    return len(x.find_all(tags))
df_all['size_tag'] = df_all['html_content'].apply(tag_size)

#それぞれのタグの個数
for tag in tags:
    q = []
    #q = [1 if len(df_all['html_content'][i].find_all(tag)) != 0 else 0 for i in range(len(df_all['state']))]
    for i in range(len(df_all['html_content'])):
        q.append(len(df_all['html_content'][i].find_all(tag)))
    df_all[tag] = q

#テキストの単語数
def words_size(x):
    return len(unicodedata.normalize("NFKD", x.text.replace('\n', ' ')).split(' '))
df_all['words_size'] = df_all['html_content'].apply(words_size)

#目標と単語数の比率
df_all['goal_words_rate'] = [df_all['goal_min'][i] / df_all['words_size'][i] for i in range(len(df_all['state']))]


# 文書分類モデル作成
MODEL = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class BertForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        #loss = output.loss
        self.log('train_loss', output.loss)
        return output.loss
   
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        #val_loss = output.loss
        self.log('val_loss', output.loss)
        
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        #num_correct = (labels_predicted == labels).sum().item()
        f = f1_score(labels_predicted, labels)
        self.log('f1', f)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

loss_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/'
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    callbacks=[loss_checkpoint]
)

def createBertFineDataSet(excerpts, targets):
    data = []    
    for excerpt, target in zip(excerpts, targets):
        encoding = tokenizer.encode_plus(
            excerpt,
            max_length = 512,
            padding='max_length',
            truncation=True
        )

        encoding['labels'] = target
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }

        data.append(encoding)

    return data

model = BertForSequenceClassification_pl(
    MODEL,
    num_labels=1,
    lr=1e-5
)

skf = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=0)
def to_text(x):
  return x.text.replace('\n', ' ')

d = df_all['html_content'].apply(to_text)[:len_train]
y = df_all['state'][:len_train]
for n_fold, (trn_idx, val_idx) in enumerate(skf.split(d, y)):
    train_dataloader = DataLoader(
        createBertFineDataSet(list(d.iloc[trn_idx]),list(y.iloc[trn_idx])),
        batch_size=32,
        shuffle=True
    )
    val_dataloader = DataLoader(
        createBertFineDataSet(list(d.iloc[val_idx]),list(y.iloc[val_idx])), 
        batch_size=256
    )
    trainer.fit(model, train_dataloader, val_dataloader)


best_model_path = loss_checkpoint.best_model_path

model = BertForSequenceClassification_pl.load_from_checkpoint(
    best_model_path
)

FINE_TUNED_MODEL_PATH = '/home/narita/Twitter/sig/text_fine_tuned_model_cased_512'

model.bert_sc.save_pretrained(FINE_TUNED_MODEL_PATH)