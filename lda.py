# 必要なライブラリのインポート
import logging
import os
import re
import sys
from collections import defaultdict
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from MeCab import Tagger
from sklearn.model_selection import GridSearchCV

data_path = ''

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
        while True:
            tweet = f.readline().strip()
            if not tweet:
                break
            tweet = json.loads(tweet)
            if tweet['is_quote_status'] == True:
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

# Mecabで形態素解析
m = Tagger()
tokenized_texts = []
for text in tweetid_text_dict.values():
    tokens = m.parse(text).splitlines()
    words = []
    for token in tokens:
        # 品詞が名詞、動詞、形容詞の単語だけ抽出
        pos = token.split('\t')[1].split(',')[0]
        if pos in ['名詞', '動詞', '形容詞']:
            # 基本形を使う
            word = token.split('\t')[1].split(',')[6]
            # 数字や記号などは除く
            if re.match(r'^\w+$', word):
                words.append(word)
    tokenized_texts.append(words)

# 単語の出現回数をカウントする辞書の作成
dictionary = corpora.Dictionary(tokenized_texts)
dictionary.filter_extremes(no_below=5, no_above=0.5) # あまり出現しない単語や頻出する単語は除く

# コーパスの作成（各文書を単語IDと出現回数のペアのリストに変換）
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# LDAモデルのパラメータ候補（トピック数は2から10まで）
parameters = {'num_topics': list(range(2, 11))}

# LDAモデルのグリッドサーチ（評価指標はCoherence）
model = models.LdaModel(id2word=dictionary)
grid_search = GridSearchCV(model, parameters, scoring='coherence')
grid_search.fit(corpus)

# 最適なトピック数とそのスコアを表示
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_
best_params = grid_search.best_params_
print(f'最適なトピック数: {best_params["num_topics"]}')
print(f'Coherenceスコア: {best_score}')

# 最適なトピック数で学習したモデルとその単語分布を表示
for topic_id in range(best_model.num_topics):
    print(f'トピック{topic_id}:')
    print(best_model.print_topic(topic_id))

for tweetid, text in tweetid_text_dict.items():
    best_model.get_document_topics(dictionary.doc2bow(text.split()))