from asyncio import tasks
import time
from urllib.parse import urlparse
import aiohttp
import asyncio
import json
import os
import collections 
import pickle

from PIL import Image
from io import BytesIO


i = 0
async def fetch(record, session_get):
    global i
    global not_found_urls
    i += 1
    url, created_at = record.split(',')
    try:
        async with session_get(url) as res:
            if i % 1000 == 0:
                print(i)
            if res.status == 200:
                id = urlparse(url).path[7:]
                i = Image.open(BytesIO(res.content))
                i.save('image/' + str(id))
            else:
                not_found_urls.append(url)
    except:
        async with session_get(url) as res:
            print('あああああああああああああああああああああ')
            if res.status == 200:
                id = urlparse(url).path[7:]
                i = Image.open(BytesIO(res.content))
                i.save('image/' + str(id))
            else:
                not_found_urls.append(url)

def scan_tweet(path, url_set):
    global not_found_urls
    records = []
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
                        if url in not_found_urls:
                            pass
                        else:
                            records.append(str(url) + ',' + str(tweet['created_at']))
                            if url not in url_set:
                                url_set.add(url)
                        
            elif 'retweeted_status' in tweet.keys():
                    if 'extended_entities' in tweet['retweeted_status'].keys():
                        for media in tweet['retweeted_status']['extended_entities']['media']:
                            if media['type'] == 'photo':
                                url = media['media_url_https']
                                if url in not_found_urls:
                                    pass
                                else:
                                    records.append(str(url) + ',' + str(tweet['created_at']))
                                    if url not in url_set:
                                        url_set.add(url)
    return [records, url_set]

def walk_dir(path_origin):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths


async def main(records):
    tasks = []
    to = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        session_get = session.get
        tasks = [fetch(record, session_get) for record in records]
        print('task start')
        await asyncio.wait(tasks)


if __name__ == '__main__':
    paths = walk_dir('/home/narita/2020-covid-media-test/')
    with open('url_set.txt') as f:
        url_set = set(f.read().splitlines())
    with open('not_found_urls.txt') as f:
        not_found_urls = f.read().splitlines()
    
    for path in paths:
        print(path)
        records, url_set = scan_tweet(path, url_set)
        print(len(records))
        print(len(url_set))
        asyncio.get_event_loop().run_until_complete(main(records))
        with open('record.txt', 'a') as w:
            for record in records:
                if record.split(',')[0] not in set(not_found_urls):
                    w.write(record + '\n')
        records = []
        time.sleep(100)
    
    with open('url_set.txt', 'wt') as w:
        w.write('\n'.join(list(url_set)))
    with open('not_found_urls.txt', 'wt') as w:
        w.write('\n'.join(not_found_urls))
    