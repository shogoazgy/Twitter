import requests
import json
import pandas as pd
import MeCab
import math
from wordcloud import WordCloud

class TextMining():
    def __init__(self):
        self.mecab = MeCab.Tagger ('~/usr/local/lib/mecab/dic/mecab-ipadic-neologdd')
    
    def text_normalize(self, text):
        text = ' '.join(text.splitlines())
        return text
    
    def word_separate(self, text, word_class=None):
        words = []
        node = self.mecab.parseToNode(text)
        if word_class is None:
            while node:
                words.append(node.surface)
                node = node.next
        else:
            while node:
                if node.feature.split(",")[0] in word_class:
                    words.append(node.surface)
                node = node.next
        return ' '.join(words)
    def word_cloud(self, words, img_file='wordcloud', font_path='~/Library/Fonts//Arial Unicode.ttf', stop_words=[], frequence = False):
        if frequence:
            wc = WordCloud(background_color='white', font_path=font_path, stopwords=stop_words, regexp=r"\w[\w']+").generate_from_frequencies(words)
        else:
            wc = WordCloud(background_color='white', font_path=font_path,  stopwords=stop_words, regexp=r"\w[\w']+").generate(words)
        wc.to_file(img_file + '.png')
        return
if __name__=='__main__':
    headers = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
    }
    guest_url = 'https://api.twitter.com/1.1/guest/activate.json'
    guest_res = requests.post(guest_url,headers=headers)
    guest_token = guest_res.json()['guest_token']

    guestheaders = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
        'x-guest-token': guest_token
    }
    
    text = ''
    tm = TextMining()
    rt_url = 'https://api.twitter.com/1.1/statuses/retweets/1397545262860537856.json?count=100'
    res = requests.get(rt_url, headers=guestheaders)
    res = res.json()
    for tweet in res:
        text += tm.text_normalize(tweet['user']['description'])
    words = tm.word_separate(text)
    stop_words = ['こと', 'よう', 'それ', 'これ', 'それ', 'もの', 'ここ', 'さん', 'ところ', 'とこ', 'https', 'co', 't', 'ー', 'です', 'から', 'たい', 'ので', 'ます', 'ない', 'ある', 'まし']
    tm.word_cloud(words, img_file='1397545262860537856',stop_words=stop_words)




