import collections
import time
import calendar
import requests

class TwiiterSearchError(Exception):
    pass

class TwitterClient:
    def __init__(self):
        self.v1headers = {
            'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
        }
    def search_tweets(self, query, since, until, geocode=None, count=10):
        url = 'https://api.twitter.com/1.1/search/universal.json?modules=status'
        max_id = -1
        tweet_list = []

        #ここは例外クラス使ってみたかっただけで無くてもいいやつ
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
                'q': query + ' OR @i -@i' + ' max_id:' + str(max_id) + ' exclude:retweets' + ' since:' + since + ' until:' + until + ' geocode:' + geocode,
                'count': 100,
                'modules': 'status',
                'result_type': 'recent',
                'tweet_mode': 'extended'
            }
        else:
            params = {
                'q': query + ' OR @i -@i' + ' max_id:' + str(max_id) + ' exclude:retweets' + ' since:' + since + ' until:' + until,
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
                    tweet_list.append(tweet['status']['data']['created_at'])
            else:
                print('Error: ' + str(res.status_code))
                continue
            repnum = repnum - 1
            if len(tweets) != 100:
                print(len(tweets))
                break
            max_id = tweets[99]['status']['data']['id_str']
            params['q'] = query + ' OR @i -@i' + ' max_id:' + str(max_id) + ' exclude:retweets' + ' since:' + since + ' until:' + until + ' geocode:' + geocode
            print(repnum)
        return tweet_list



if __name__=='__main__':
    api = TwitterClient()
    tweets = api.search_tweets(query='桜', since='2021-03-05', until='2021-04-05', count=100, geocode='38.314297,140.2083761,50km')
    tweeted_time_list = []
    tweeted_time_str = '\n'.join(tweets)
    with open("sendai.txt", 'wt') as f:
        f.write(tweeted_time_str)
    for tweet in tweets:
        tweeted_time_utc = time.strptime(tweet, '%a %b %d %H:%M:%S +0000 %Y')
        tweeted_time_unix = calendar.timegm(tweeted_time_utc)
        tweeted_time_local = time.localtime(tweeted_time_unix)
        tweeted_time_japan = time.strftime("%Y-%m-%d", tweeted_time_local)
        tweeted_time_list.append(tweeted_time_japan)
    tweeted_time_list_counter = collections.Counter(tweeted_time_list)
    print(tweeted_time_list_counter)


