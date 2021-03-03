import requests
import json
import pandas as pd
import MeCab
import math
from wordcloud import WordCloud
import folium
# 以下に自身のTwitter api v2対応しているbearer tokenを入力。ない場合はv1.1エンドポイントのみ使用可。
bearer_token = ''

class TwiiterSearchError(Exception):
    pass

class TwitterClient:
    def __init__(self):
        self.v1headers = {
            'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
        }
        self.v2headers = {
            'Authorizatio': 'Bearer ' + bearer_token
        }
    def search_tweets(self, query, geocode=None, count=10):
        url = 'https://api.twitter.com/1.1/search/universal.json?modules=status'
        max_id = -1
        tweet_list = []
        if type(count) is not int or count < 1:
            raise TwiiterSearchError("'count' must be positive integer")
        if type(query) is not str:
            raise TwiiterSearchError("'query' must be str")
        if geocode is not None:
            if type(geocode) is not str:
                raise TwiiterSearchError("'geocode' must be str")
        if count > 100:
            repnum = ((count - 1) // 100) + 1
            rem = count % 100
        else:
            repnum = 1
            rem = count
        if rem == 0:
            rem = 100
        if geocode is not None:
            params = {
                'q': query + ' OR @i -@i' + ' max_id:' + str(max_id) + ' exclude:retweets' + ' geocode:' + geocode,
                'count': 100,
                'modules': 'status',
                'result_type': 'recent',
                'tweet_mode': 'extended'
            }
        else:
            params = {
                'q': query + ' OR @i -@i' + ' max_id:' + str(max_id) + ' exclude:retweets',
                'count': 100,
                'modules': 'status',
                'result_type': 'recent',
                'tweet_mode': 'extended'
            }
        while repnum > 0:
            if repnum == 1:
                params['count'] = rem
            res = requests.get(url,headers=self.v1headers,params=params)
            if res.status_code == 200:
                dicres = res.json()
                tweets = dicres['modules']
                for tweet in tweets:
                    tweet_list.append(tweet['status']['data'])
            else:
                print('Error: ' + res.status_code)
            repnum = repnum - 1
            print(repnum)
            if len(tweets) != 100:
                break
        if geocode is None:
            tweet_list[0]['geocode'] = None
        else:
            tweet_list[0]['geocode'] = geocode
        return tweet_list
    
    def save_tweets(self, tweets, filename='tweets'):
        with open(filename + '.json', 'w') as f:
            json.dump(tweets, f, ensure_ascii=False, indent=3, sort_keys=False, separators=(',', ': '))

class TweetMining():
    def __init__(self, tweets):
        df = pd.json_normalize(tweets)
        self.tweets = tweets
        self.text_list = df['full_text']
        
    def word_separate(self, word_class=None):
        words = []
        mecab = MeCab.Tagger ('~/usr/local/lib/mecab/dic/mecab-ipadic-neologdd')
        for text in self.text_list:
            if pd.isna(text):
                pass
            else:
                node = mecab.parseToNode(text)
                while node:
                    if word_class is not None:
                        if node.feature.split(",")[0] in word_class:
                            words.append(node.surface)
                    else:
                        words.append(node.surface)
                    node = node.next
        return ' '.join(words)
    
    def word_cloud(self, words, img_file='wordcloud', font_path='~/Library/Fonts//Arial Unicode.ttf', stop_words=[]):
        wc = WordCloud(background_color='white', font_path=font_path, stopwords=stop_words).generate(words)
        wc.to_file(img_file + '.png')
        return
    
    def map_plot(self, filename='map'):
        if tweets[0]['geocode'] is None:
            location = [35.6809591,139.7673068]
        else:
            location_str = tweets[0]['geocode'].split(',')
            latitude = float(location_str[0])
            longitude = float(location_str[1])
            location = [latitude, longitude]
        map = folium.Map(location=location, zoom_start=10)
        for tweet in self.tweets:
            lat = 0
            lon = 0
            if tweet['geo'] is None:
                if tweet['place'] is None:
                    continue
                else:
                    for t in tweet['place']['bounding_box']['coordinates'][0]:
                        lat += t[1]
                        lon += t[0]
                    lat /= len(tweet['place']['bounding_box']['coordinates'][0])
                    lon /= len(tweet['place']['bounding_box']['coordinates'][0])
            else:
                lat = tweet['geo']['coordinates'][0]
                lon = tweet['geo']['coordinates'][1]
            if lat != 0 or lon != 0:
                folium.Marker([lat, lon]).add_to(map)
            map.save(filename + '.html')




if __name__=='__main__':
    api = TwitterClient()
    tweets = api.search_tweets(query='山手線', count=1000, geocode='35.690921,139.70025799999996,50km')
    api.save_tweets(tweets)
    stop_words = ['こと', 'よう', 'それ', 'これ', 'それ', 'もの', 'ここ', 'さん', 'ところ', 'とこ', 'https', 'co', 't', 'ー']
    textmining = TweetMining(tweets)
    textmining.map_plot()
    
