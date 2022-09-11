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
from fastprogress.fastprogress import progress_bar
from sklearn.decomposition import PCA


df_train = pd.read_csv('/home/narita/Twitter/sig/train.csv')
df_test = pd.read_csv('/home/narita/Twitter/sig/test.csv')
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

"""
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
    #gpus=1,
    accelerator='cpu',
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
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FINE_TUNED_MODEL_PATH = '/home/narita/Twitter/sig/text_fine_tuned_model_cased_512'
model = BertModel.from_pretrained(FINE_TUNED_MODEL_PATH, num_labels=1)
model.eval()

class BertDataset(nn.Module):
    def __init__(self, df, tokenizer, max_len=512):
        self.excerpt = df.to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __getitem__(self,idx):
        encode = self.tokenizer.encode_plus(
            self.excerpt[idx],
            return_tensors='pt',
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )
        return encode
    
    def __len__(self):
        return len(self.excerpt)

def get_embeddings(df, path, plot_losses=True, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is used")
            
    MODEL_PATH = path
    model = BertModel.from_pretrained(MODEL_PATH, num_labels=1)
    #model.to(device)
    model.eval()
    
    ds = BertDataset(df, tokenizer, config['max_len'])
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers = 4,
        pin_memory=True,
        drop_last=False
    )

    embeddings = list()
    with torch.no_grad():
        for i, inputs in progress_bar(list(enumerate(dl))):
            inputs = {key:val.reshape(val.shape[0], -1).to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            outputs = outputs[0][:, 0].detach().cpu().numpy()
            embeddings.extend(outputs)
            
    return np.array(embeddings)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

config = {
    'batch_size': 128,
    'max_len': 512,
    'seed': 42,
}
seed_everything(seed=config['seed'])

def to_text(x):
    return x.text.replace('\n', ' ')
d_tr = df_all['html_content'].apply(to_text)[:len_train]
d_te = df_all['html_content'].apply(to_text)[len_train:]

train_embeddings =  get_embeddings(d_tr, FINE_TUNED_MODEL_PATH)
test_embeddings = get_embeddings(d_te, FINE_TUNED_MODEL_PATH)

import pickle
with open('/home/narita/Twitter/sig/embeddings_cased_512.pkl', 'wb') as w:
    pickle.dump(np.concatenate([train_embeddings, test_embeddings]), w)

x = np.concatenate([train_embeddings, test_embeddings]).tolist()

for i in range(len(x[0])):
    df_all['x' + str(i)] = [q[i] for q in x]

df_all['goal_min'] = np.log(df_all['goal_min'])
df_all['goal_max'] = np.log(df_all['goal_max'])

df = df_all.drop(columns=['id', 'html_content', 'goal', 'goal_max'])
data = df[:len_train].drop(["state"], axis=1)
test = df[len_train:].drop(["state"], axis=1)
y = df[:len_train]["state"]
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
skf = StratifiedKFold(n_splits=5,
                      shuffle=True,
                      random_state=0)

def train_model(data_, test_, y_, folds_, seed=42):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    oof_preds_x = np.zeros(data_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns]
    
    #for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=3,
            colsample_bytree=.8,
            subsample=.9,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
            random_seed=seed,
        )
        """
        clf = LGBMClassifier(
            n_estimators=4000,
            #learning_rate=0.0995,
            learning_rate=0.19,
            num_leaves=170,
            colsample_bytree=0.6967,
            subsample=0.526563,
            max_depth=7,
            #reg_alpha=0.0008114799,
            reg_alpha=0.00000056,
            #reg_lambda=7.23305,
            reg_lambda=0.002977,
            min_split_gain=.01,
            min_child_weight=4,
            silent=-1,
            verbose=-1,
            random_seed=seed,
        )
        """
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='f1', verbose=100, early_stopping_rounds=100, categorical_feature=cat_cols)  #30)
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        oof_preds_x[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_)[:]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        #print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        print('Fold %2d F1 : %.6f' % (n_fold + 1, f1_score(val_y, oof_preds_x[val_idx])))
        ans = clf.predict(test_)
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    #print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
    print('Full F1 score %.6f' % f1_score(y, oof_preds_x)) 
    

    return oof_preds, sub_preds, feature_importance_df, ans

oof_preds, test_preds, importances, ans = train_model(data, test, y, skf)

print(importances[0:20])