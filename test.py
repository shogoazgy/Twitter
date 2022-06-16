from asyncio import tasks
import aiohttp
import asyncio
import json
import os
from PIL import Image
from io import BytesIO

all_img_count= 0
img_found_count = 0

async def fetch(url, session_get):
    global img_found_count
    async with session_get(url) as res:
        print(url)
        if res.status == 200:
            img_found_count += 1
        else:
            print('Z')

def scan_tweet(paths, queue):
    global all_img_count
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
                            queue.put_nowait(url)
                            all_img_count += 1
                elif 'retweeted_status' in tweet.keys():
                        if 'extended_entities' in tweet['retweeted_status'].keys():
                            for media in tweet['retweeted_status']['extended_entities']['media']:
                                if media['type'] == 'photo':
                                    url = media['media_url_https']
                                    queue.put_nowait(url)
                                    all_img_count += 1

async def worker(queue, session_get):
    while True:
        url = await queue.get()
        await fetch(url, session_get)
        queue.task_done()

def walk_dir(path_origin):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths


async def main():
    paths = walk_dir('/home/narita/2020-covid-media-test/')
    queue = asyncio.Queue()
    scan_tweet(paths, queue)
    print(queue)
    tasks = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100)) as session:
        session_get = session.get
        task = asyncio.ensure_future(worker(queue, session_get))
        tasks.append(task)
    await queue.join()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print(all_img_count)
    print(img_found_count)

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())