from importlib.resources import path
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

mecab = MeCab.Tagger()
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

def change_time(created_at, only_date=True):
    st = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')        
    utc_time = datetime.datetime(st.tm_year, st.tm_mon,st.tm_mday, st.tm_hour,st.tm_min,st.tm_sec, tzinfo=datetime.timezone.utc)   
    jst_time = utc_time.astimezone(pytz.timezone("Asia/Tokyo"))
    if only_date:
        str_time = jst_time.strftime("%Y-%m-%d")                    
    else:
        str_time = jst_time.strftime("%Y-%m-%d_%H:%M:%S")                    
    return str_time

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

def walk_dir(path_origin, since=None, until=None):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths

                

if __name__ == "__main__":
    g = Graph.Read_Ncol('/Users/shougo/Downloads/graphs_clusters/2020_04_clusters')