import imp
import aiohttp
import asyncio
import json
import os

urls = []

async def fetch(url, session_get):
    async with session_get(url) as res:
        print(url)
        if res.status == 200:
            pass
        else:
            print('Z')

async def scan_tweet(paths, queue):
    for path in paths:
        with open(path) as f:
            tweet = json.loads(f.readline().strip())
            if 'extended_entities' in tweet.keys():
                for media in tweet['extended_entities']['media']:
                    if media['type'] == 'photo':
                        url = media['media_url_https']
                        queue.put_nowait(url)
            elif 'extended_entities' in tweet['retweeted_status'].keys():
                for media in tweet['extended_entities']['media']:
                    if media['type'] == 'photo':
                        url = media['media_url_https']
                        queue.put_nowait(url)

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
    tasks = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100)) as session:
        session_get = session.get
        task = asyncio.create_task(worker(queue, session_get))
        tasks.append(task)
    await queue.join()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    asyncio.run(main())