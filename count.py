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
                    time_dict[tweet['quoted_status']['id_str']] = tweet['quoted_status']['created_at']

def convert(std_time, times):
    x = []
    print(std_time)
    std = datetime.datetime().strptime(std_time, '%a %b %d %H:%M:%S %z %Y')

    for t in times:
        x.append((std - datetime.datetime().strptime(t, '%a %b %d %H:%M:%S %z %Y')).total_seconds())
    return x


if __name__ == '__main__':
    """
    paths = walk_dir('/home/narita/2020-covid-media-02-07')
    rt_d = collections.defaultdict(list)
    quote_d = collections.defaultdict(list)
    time_dict = collections.defaultdict(str)
    main(paths)
    with open('10000_rt.txt', 'wt') as w:
        for k, rt in rt_d.items():
            if len(rt) >= 10000:
                w.write(str(k) + ',' + str(time_dict[k]) + ',' + ','.join(rt) + '\n')
    """
    with open('10000_rt.txt') as f:
        for _ in range(10):
            t = f.readline().strip().split(',')
            x = convert(t[1], t[2:])
            plt.hist(x[2:], bins=50)
            plt.savefig(str(t[0]) + '.png')
            plt.clf()
    