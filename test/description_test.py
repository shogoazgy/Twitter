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

if __name__ == '__main__':
    with open('tweets.json') as f:
        tweets = json.load(f)
    description = ''
    tm = TextMining()
    stop_words = ['こと', 'よう', 'それ', 'これ', 'それ', 'もの', 'ここ', 'さん', 'ところ', 'とこ', 'https', 'co', 't', 'ー', 'です', 'から', 'たい', 'ので', 'ます', 'ない', 'ある', 'まし']
    for tweet in tweets:
        description += tm.text_normalize(tweet['user']['description'])
    words = tm.word_separate(description)
    tm.word_cloud(words, img_file='#東京五輪の中止を求めます_description',stop_words=stop_words)
