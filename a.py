from wsgiref import headers
import requests

key = 'AAAAAAAAAAAAAAAAAAAAALcwKQEAAAAAgoxyQYzkkALjVaG3Px6WD5ihbbA%3DYbBEmPZVThDqxXmu85J6pwVLSoIYmQrAQBKZyH5FU6GmrB7fix'
id_str = '1223011527415828480'
url = 'https://api.twitter.com/2/tweets/' + id_str + '/quote_tweets?max_results=100'
header = {
    'Authorization': 'Bearer ' + key,
}

#print(requests.get(url + '&pagination_token=9an1hcsuutq8', headers=header).json())

with open(str(id_str) + '_quoted_texts.csv', 'wt') as w:
    w.write('quote tweet id,quote text,label\n')
    res = requests.get(url, headers=header).json()
    for tweet in res['data']:
        w.write(tweet['id'] + ',' + repr(str(tweet['text']))[1:-1] + ',' + '\n')
    while True:
        print(len(res['data']))
        n_url = url + '&pagination_token=' + res['meta']['next_token']
        res = requests.get(n_url, headers=header).json()
        for tweet in res['data']:
            w.write(tweet['id'] + ',' + repr(str(tweet['text']))[1:-1] + ',' + '\n')
        if 'next_token' in res['meta'].keys()
            break




