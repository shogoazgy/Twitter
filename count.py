import os
import collections
import json
import matplotlib.pyplot as plt
import datetime

def walk_dir(path_origin):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths

def main(paths):
    global rt_d
    global quote_d
    global time_dict
    for path in paths:
        print(path)
        with open(path) as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if 'retweeted_status' in tweet.keys():
                    rt_d[tweet['retweeted_status']['id_str']].append(tweet['created_at'])
                    time_dict[tweet['retweeted_status']['id_str']] = tweet['retweeted_status']['created_at']
                elif 'quoted_status' in tweet.keys():
                    quote_d[tweet['quoted_status']['id_str']].append(tweet['created_at'])
                    quote_d[tweet['quoted_status']['id_str']].append(str(len(tweet['text'])))
                    time_dict[tweet['quoted_status']['id_str']] = tweet['quoted_status']['created_at']

def convert(std_time, times):
    x = []
    print(std_time)
    std = datetime.datetime.strptime(std_time, '%a %b %d %H:%M:%S %z %Y')

    for t in times:
        x.append((datetime.datetime.strptime(t, '%a %b %d %H:%M:%S %z %Y') - std).total_seconds())
    return x

if __name__ == '__main__':
    """
    paths = walk_dir('/home/narita/2020-covid-media-02-07')
    rt_d = collections.defaultdict(list)
    quote_d = collections.defaultdict(list)
    time_dict = collections.defaultdict(str)
    quote_len_dict = collections.defaultdict(int)
    main(paths)
    with open('100_quote.txt', 'wt') as w:
        for k, q in quote_d.items():
            if len(q) >= 100:
                w.write(str(k) + ',' + str(time_dict[k]) + ',' + ','.join(q) + '\n')
    
    with open('10000_rt.txt') as f:
        for _ in range(10):
            t = f.readline().strip().split(',')
            x = convert(t[1], t[2:])
            plt.hist(x[2:], bins=100, range=(0, 500000))
            plt.savefig('count_image/' + str(t[0]) + '.png')
            plt.clf()
    
    with open('100_quote.txt') as f:
        for _ in range(10000):
            t = f.readline().strip().split(',')
            x_t = t[2:]
            x = convert(t[1], x_t[0::2])
            plt.hist(x, bins=100, range=(0, 500000))
            plt.savefig('count_image/' + str(t[0]) + '.png')
            plt.clf()
            y = [int(a) for a in x_t[1::2]]
            plt.scatter(x, y)
            plt.savefig('len_text/' + str(t[0]) + '.png')
            plt.clf()
    """
    paths = walk_dir('/home/narita/2020-covid-media-02-07')
    target_set = set()
    text_set_dict = collections.defaultdict(set)
    with open('10000_rt.txt') as f:
        while True:
            t = f.readline()
            if not t:
                break
            t = t.split(',')
            target_set.add(t[0])
    i = 0
    for path in paths:
        print(path)
        with open(path) as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if 'quoted_status' in tweet.keys():
                    if tweet['quoted_status']['id_str'] in target_set:
                        text_set_dict[tweet['quoted_status']['id_str']].add(tweet['text'])
        i += 1
        if i > 10:
            break
    for k, q in text_set_dict.items():
        with open('quote_texts/' + str(k) + '_quoted_texts.csv', 'wt') as w:
            for text in q:
                w.write(str(text) + ',' + '\n') 
    