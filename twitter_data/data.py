import json
import collections
import re
import MeCab
import unicodedata
import datetime
import time
import pytz
import os
import decimal
import sys
import re
import unicodedata
from igraph import *
import leidenalg as la
import igraph
from sklearn.feature_extraction.text import TfidfVectorizer

mecab = MeCab.Tagger ()
def word_separate(text, word_class=None):
    words = []
    text = unicodedata.normalize('NFKC', text)
    node = mecab.parseToNode(text)
    while node:
        if word_class is not None:
            if node.feature.split(",")[0] in word_class:
                try:
                  words.append(node.surface)
                except:
                  print(text)
        else:
            words.append(node.surface)
        node = node.next
    return ' '.join(words)


def normalize(text):
    text = re.sub(r'[0-9]', "", text)
    return re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" , text)

def get_tf_idf(text_list):
    vectorizer = TfidfVectorizer(max_df=0.9)
    vectorizer.fit(text_list)
    return vectorizer


def extract_users(path_origin, size, file_name='rt_users'):
    with open(file_name + '.txt', 'wt') as w:
        flag = 0
        for i in range(9, size):
            print(i)
            path = path_origin + str(i)
            with open(path) as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    t = json.loads(t)
                    if flag:
                        rt_rted = str(t['user']['id_str']) + ',' + str(t['retweeted_status']['user']['id_str'])
                        w.write(rt_rted + '\n')
                    else:
                        if change_time(t['created_at']) == '2021-01-01':
                            flag = 1
                            rt_rted = str(t['user']['id_str']) + ',' + str(t['retweeted_status']['user']['id_str'])
                            w.write(rt_rted + '\n')


def change_time(created_at, only_date=True):
    st = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')        
    utc_time = datetime.datetime(st.tm_year, st.tm_mon,st.tm_mday, st.tm_hour,st.tm_min,st.tm_sec, tzinfo=datetime.timezone.utc)   
    jst_time = utc_time.astimezone(pytz.timezone("Asia/Tokyo"))
    if only_date:
        str_time = jst_time.strftime("%Y-%m-%d")                    
    else:
        str_time = jst_time.strftime("%Y-%m-%d_%H:%M:%S")                    
    return str_time

def test(path):
    date = '2021-07-24'
    i = 0
    with open(path) as f:
        while True:
            t = f.readline()
            if not t:
                break
            t = t.strip()
            t = json.loads(t)
            date_t = change_time(t['created_at'])
            if date != date_t:
                i += 1
                print(t['created_at'])
                #print(t)
            if not t['user']['id_str'].isdecimal() or not t['retweeted_status']['user']['id_str']:
                print(t)
    print(i)

def separate_days(path_origin, size, year_month, save_dir):
    day = 1
    w = open(os.path.join(save_dir, str(year_month) + '_' + str(day)), 'wt')
    for i in range(size):
        print(i)
        path = path_origin + str(i)
        with open(path) as f:
            while True:
                t = f.readline()
                if not t:
                    break
                t = t.strip()
                t = json.loads(t)
                day_t = change_time(t['created_at'])
                if int(day_t[8:]) == day:
                    rt_rted = str(t['user']['id_str']) + ',' + str(t['retweeted_status']['user']['id_str'])
                    w.write(rt_rted + '\n')
                elif int(day_t[8:]) < day:
                    x = open(os.path.join(save_dir, str(year_month) + '_' + str(int(day_t[8:]))), 'a')
                    rt_rted = str(t['user']['id_str']) + ',' + str(t['retweeted_status']['user']['id_str'])
                    x.write(rt_rted + '\n')
                    x.close()
                else:
                    print(day_t)
                    day += 1
                    w.close()
                    w = open(os.path.join(save_dir, str(year_month) + '_' + str(day)), 'wt')
    w.close()



def extract_date(path_origin, size, file_name='dates'):
    dates = collections.defaultdict(int)
    for i in range(size):
        print(i)
        path = path_origin + str(i)
        with open(path) as f:
            while True:
                t = f.readline()
                if not t:
                    break
                t = t.strip()
                t = json.loads(t)
                created_at = change_time(t['created_at'])
                dates[created_at] += 1
    with open(file_name + '.txt', 'wt') as f:
        for date in dates.keys():
            d = str(date) + ',' + str(dates[date])
            f.write(d + '\n')
        
def write_texts(paths, save_filename):
    rt_user_list = []
    with open(save_filename, 'wt') as w:
        for path in paths:
            i = 0
            j = 0
            print(path)
            print(len(rt_user_list))
            with open(path, 'r') as f:
                while True:
                    i += 1
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    t = json.loads(t)
                    if 'retweeted_status' in t.keys():
                        w.write(t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str'] + ',' + t['retweeted_status']['text'].replace('\n', '') + '\n')
                    else:
                        j += 1
                print(i)
                print(j)

def walk_dir(path_origin, since=None, until=None):
    paths = []
    month = '02'
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            if filename[5:7] != month:
                print(filename)
                print('--reset--')
                path = os.path.join(pathname, filename)
                if filename[5:7] == '07':
                    print(filename)
                    write_texts(paths, save_filename=filename[:5] + month + '_tweets')
                month = filename[5:7]
                paths = []
                paths.append(path)
            else:
                print(filename)
                path = os.path.join(pathname, filename)
                paths.append(path)
    return paths

                

if __name__ == "__main__":
    #extract_users('/Users/shougo/Downloads/narita-mar/keyword-search.tweet.', 8, file_name='test')
    #extract_date('/Users/shougo/Downloads/narita-newyear/keyword-search.tweet.', 30, file_name='new_year_date')
    #extract_users('/Users/shougo/Downloads/narita-newyear/keyword-search.tweet.', 30, file_name='rt_users_jan')
    #separate_days('/Users/shougo/Downloads/narita-mar/keyword-search.tweet.', 8, '2021_3', '/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_3')
    #walk_dir('/Users/shougo/Downloads/narita2021')
    #test('/Users/shougo/Downloads/data/str01_03/twitter/shohei/result/PFTqp4cu38skwf0xxkr_qp4cu10z9kwlp9c7r/2021-07-24.0.tweets')
    g = g = Graph.Read_Ncol('/Users/shougo/Downloads/graphs_clusters/2020_04_clusters')
    with open('/Users/shougo/Downloads/graphs_clusters/2020_04_membership') as f:
      g.vs['cluster'] = [float(s.strip()) for s in f.readlines()]
    user_sets = []
    for i in range(20):
        user_sets.append(set(g.vs.select(lambda vertex : vertex['cluster'] == i)['name']))
    del g
    words_all = []
    words_list = []
    for i in range(20):
        words_list.append([])
    with open('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2020-04_tweets') as f:
        i = 0
        while(True):
            t = f.readline()
            if not t:
                print(i)
                break
            t = t.strip()
            i += 1
            if i % 10 != 0:
                continue
            if i % 100000 == 0:
                print(i)
            t = t.split(',')
            text = ''.join(t[2:])
            words = word_separate(text, word_class=['名詞', '形容詞'])
            words_all.append(words)
            for j in range(20):
                if t[0] in user_sets[j]:
                    if t[1] in user_sets[j]:
                        words_list[j].append(words)
                        break
                    else:
                        break
    vectorizer =  get_tf_idf(words_all)
    with open('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2020-04_tfidf', 'wt') as f:
        for i in range(20):
            texts = [' '.join(words_list[i])]
            x = vectorizer.transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            q = x.toarray().argsort()
            q = q[0][::-1]
            fre_words = []
            for ind in q[0:40]:
                fre_words.append(feature_names[ind])
            words_str = ','.join(fre_words)
            print('クラスタ' + str(i) + ': ' + words_str)
            f.write(words_str + '\n')
    """
    for pathname, dirnames, filenames in os.walk('/Users/shougo/downloads/narita-newyear'):
        for file in filenames:
            path = os.path.join(pathname, file)
            with open(path, 'r') as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    t = json.loads(t)
    """