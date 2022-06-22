from asyncio import tasks
from urllib.parse import urlparse
import aiohttp
import asyncio
import json
import os
import collections 
import pickle

from PIL import Image
from io import BytesIO

d = collections.defaultdict(int)

all_img_count= 0
img_found_count = 0
i = 0
async def fetch(url, session_get):
    global img_found_count
    global i
    global d
    i += 1
    try:
        async with session_get(url) as res:
            if i % 100 == 0:
                print(i)
            if res.status == 200:
                img_found_count += 1
                id = urlparse(url).path[7:]
                if id in d.keys():
                    d[url] += 1
                else:
                    i = Image.open(BytesIO(res.content))
                    i.save(id)
            else:
                #print(str(res.status) + ' : ' + str(url))
                pass
    except:
        async with session_get(url) as res:
            print('あああああああああああああああああああああ')
            if res.status == 200:
                img_found_count += 1
            else:
                print('2 X')

def scan_tweet(paths):
    global all_img_count
    urls = []
    c_set = set()
    for path in paths:
        with open(path) as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if 'extended_entities' in tweet.keys():
                    for media in tweet['extended_entities']['media']:
                        if media['type'] == 'photo':
                            url = media['media_url_https']
                            urls.append(url)
                            all_img_count += 1
                            c_set.add(url)
                elif 'retweeted_status' in tweet.keys():
                        if 'extended_entities' in tweet['retweeted_status'].keys():
                            for media in tweet['retweeted_status']['extended_entities']['media']:
                                if media['type'] == 'photo':
                                    url = media['media_url_https']
                                    urls.append(url)
                                    all_img_count += 1
                                    c_set.add(url)
    print(len(c_set))
    return urls

def walk_dir(path_origin):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths


async def main(urls):
    global d
    tasks = []
    to = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        session_get = session.get
        print(len(urls))
        tasks = [fetch(url, session_get) for url in urls]
        print('task start')
        await asyncio.wait(tasks)
        print(all_img_count)
        print(img_found_count)
        with open('d.pickle', 'wb') as f:
            pickle.dump(d, f)


if __name__ == '__main__':
    paths = walk_dir('/home/narita/2020-covid-media-test/')
    urls = scan_tweet(paths)
    print('x')
    #asyncio.get_event_loop().run_until_complete(main(urls))