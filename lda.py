# 必要なライブラリのインポート
import logging
import os
import re
import sys
import itertools
import json
from collections import defaultdict
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from MeCab import Tagger
from sklearn.model_selection import GridSearchCV

data_path = '/home/narita/covid-07_09'

# get all the file pathes in the direcotory
def get_file_pathes(directory):
    file_pathes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_pathes.append(os.path.join(root, file))
    return file_pathes

# ログの設定
logging.basicConfig(level=logging.INFO)

# read text from tweets
tweetid_text_dict = dict()
tweetid_userid_dict = defaultdict(list)



paths = get_file_pathes(data_path)
for path in paths:
    with open(path, 'r') as f:
        print(path)
        while True:
            tweet = f.readline().strip()
            if not tweet:
                break
            tweet = json.loads(tweet)
            if 'is_quote_status' in tweet.keys() and tweet['is_quote_status'] == True:
                try:
                    tweetid_userid_dict[tweet['quoted_status']['id_str']].append(tweet['user']['id_str'])
                    tweetid_text_dict[tweet['quoted_status']['id_str']] = tweet['quoted_status']['text']
                except:
                    pass
            elif 'retweeted_status' in set(tweet.keys()):
                tweetid_userid_dict[tweet['retweeted_status']['id_str']].append(tweet['user']['id_str'])
                tweetid_text_dict[tweet['retweeted_status']['id_str']] = tweet['retweeted_status']['text']
            elif tweet['in_reply_to_user_id_str'] is not None:
                tweetid_userid_dict[tweet['id_str']].append(tweet['user']['id_str'])
                tweetid_text_dict[tweet['id_str']] = tweet['text']

# remove urls
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Mecabで形態素解析
m = Tagger('-Ochasen -d /usr/lib64/mecab/dic/mecab-ipadic-neologd/')
tokenized_texts = []
for text in tweetid_text_dict.values():
    tokens = m.parse(text).splitlines()
    words = []
    for token in tokens:
        if token == "EOS" or token == "":
            continue
        # 行をタブで分割
        parts = token.split("\t")
        if parts == "":
            continue
        # 表層形と品詞を取得
        surface = parts[0]
        if len(parts) < 4:
            continue
        if len(parts[4].split("-")) == 0:
            print(parts)
        pos = parts[4].split("-")[0]
        # 品詞が名詞動詞形容詞ならリストに追加
        if pos in ["名詞", "動詞", "形容詞"]:
            if re.match(r'^\w+$', surface):
                words.append(surface)
    tokenized_texts.append(words)

# 単語の出現回数をカウントする辞書の作成
dictionary = corpora.Dictionary(tokenized_texts)
dictionary.filter_extremes(no_below=5, no_above=0.5) # あまり出現しない単語や頻出する単語は除く

# コーパスの作成（各文書を単語IDと出現回数のペアのリストに変換）
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

num_topics_list = range(1, 21)

coherence_vals = []
perplexity_vals = []

# 各トピック数に対してモデルを作成し、Coherenceスコアを計算
for num_topics in num_topics_list:
    # Coherenceスコアのリスト
    coherence_scores = []
    perplexity_scores = []

    model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=1)
    cm = models.coherencemodel.CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_vals.append(cm.get_coherence())
    perplexity_vals.append(np.exp(-model.log_perplexity(corpus)))
    print(str(num_topics), cm.get_coherence())

# 最適なトピック数とそのスコアを表示
print("最適なトピック数: ", coherence_vals.index(max(coherence_vals)) + 1)
best_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=(coherence_vals.index(max(coherence_vals))) + 1, passes=1)

topicid_userid_dict = defaultdict(list)
for tweetid, text in tweetid_text_dict.items():
    topics = best_model.get_document_topics(dictionary.doc2bow(text.split()))
    for topicid, score in topics:
        if score > 0.5:
            topicid_userid_dict[topicid].extend(tweetid_userid_dict[tweetid])

topic1_topic2_simpson_dict = dict()

def simpson(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1 & set2) / min([len(set1), len(set2)])

for topic1, topic2 in itertools.combinations(topicid_userid_dict.keys(), 2):
    topic1_topic2_simpson_dict[(topic1, topic2)] = simpson(topicid_userid_dict[topic1], topicid_userid_dict[topic2])

sorted_topic1_topic2_simpson_tuple = sorted(topic1_topic2_simpson_dict.items(), key=lambda x: x[1], reverse=True)

for topic1_topic2, simpson in sorted_topic1_topic2_simpson_tuple[:10]:
    print(topic1_topic2, simpson)

