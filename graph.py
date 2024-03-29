import pandas as pd
import numpy as np
from igraph import *
import collections
import leidenalg as la
import igraph
import requests
import matplotlib.pyplot as plt
import time
import os
import json
import sys
import subprocess
import pickle
import random
import datetime
#from sklearn.linear_model import LinearRegression

def count_lines(path):
    cmd = "wc -l " + path
    c = subprocess.check_output(cmd.split()).decode().split()[0]
    return c

def count_users(rt_users_path):
    with open(rt_users_path, 'r') as f:
        rt_user_list = [s.strip() for s in f.readlines()]
    c = collections.Counter(rt_user_list)
    user_list = []
    for k in c.most_common():
        to_from_freq = k[0].split(',')
        user_list.append(to_from_freq[0])
        user_list.append(to_from_freq[1])
    user_list_c = collections.Counter(user_list)
    print(len(user_list_c))
    return user_list_c

def clustering(g, resolution_parameter=1):
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, weights=g.es['weight'], n_iterations=-1, resolution_parameter=resolution_parameter)
    return partition

def optimise_resolition_parameter(g):
    optimiser = la.Optimiser()
    profile = optimiser.resolution_profile(g, la.RBConfigurationVertexPartition, weights=g.es['weight'], resolution_range=(0, 1))
    return profile

def subgraph(maingraph, vertices):
    sub = maingraph.induced_subgraph(vertices)
    return sub

def save_gml(g, filename):
    g.save(filename, format='gml')

def get_member_str_from_cluster(partition):
    member_list = []
    for member in partition:
        member_list.append(g.vs[member]['name'])
    return member_list

def plot_cluster(partition, filename):
    visual_style = {}
    visual_style["vertex_size"] = 3
    #visual_style["edge_width"] = [int(x**0.5) for x in g.es["weight"]]
    visual_style["edge_width"] = 1
    visual_style['edge_arrow_size'] = 1
    visual_style['dge_arrow_width'] = 1
    igraph.plot(partition, filename + '.png', **visual_style)

def get_guest_token():
  guesturl = "https://api.twitter.com/1.1/guest/activate.json"
  header1 = {
    'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
  }
  fres = requests.post(guesturl,headers=header1)
  guest_token = fres.json()['guest_token']
  print('guest_token: ' + guest_token)
  return guest_token

def id_to_screen_name(id, guest_token):
    headers = {
        'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
        'x-guest-token': guest_token
    }
    url = 'https://api.twitter.com/1.1/users/show.json?user_id=' + str(id)
    try:
        res = requests.get(url, headers=headers)
        res = res.json()
        screen_name = res['screen_name']
    except:
        screen_name = None
    return screen_name

def cal_strength(g):
    #strength_list = []
    #for g in g_list:
    #    strength_list.append(g.strength(g.vs, mode='in', loops=False, weights=g.es['weight']))
    #return strength_list
    return g.strength(g.vs, mode='in', weights=g.es['weight'])

def sort_max_print(subg, centrality, guest_token, len=10):
    cen_sorted = sorted(centrality, reverse=True)
    for i in range(len):
        ind = centrality.index(cen_sorted[i])
        print(id_to_screen_name(subg.vs[ind]['name'], guest_token=guest_token))

def build_network(paths, save_filename, mode='quoted', days=None, months=None):
    rt_user_list = []
    gs = []
    if days:
        start_date = datetime.datetime.strptime(paths[0].split('/')[-1].split('.')[0], '%Y-%m-%d')
    for path in paths:
        date = datetime.datetime.strptime(path.split('/')[-1].split('.')[0], '%Y-%m-%d')
        if (date - start_date).days >= days:
            c = collections.Counter(rt_user_list)
            rt_user_list = []
            to_from_freq_list = []
            for k in c.most_common():
                to_from_freq = k[0].split(',')
                to_from_freq.append(k[1])
                to_from_freq_list.append(to_from_freq)
            g = Graph.TupleList(to_from_freq_list, weights=True, directed=True)
            g.save(save_filename + start_date.strftime('%Y-%m-%d') + '_' + (date - datetime.timedelta(days=1)).strftime('%Y-%m-%d'), format='gml')
            gs.append([g, start_date.strftime('%Y-%m-%d') + '_' + (date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')])
            start_date = date
            print('next')
        print(path)
        with open(path, 'r') as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if mode == 'quoted':
                    #if tweet['is_quote_status'] == True:
                    try:
                        rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['quoted_status']['user']['id_str'])
                    except:
                        c += 1
                elif mode == 'retweet':
                    if 'retweeted_status' in set(tweet.keys()):
                        rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['retweeted_status']['user']['id_str'])
                elif mode == 'reply':
                    if tweet['in_reply_to_user_id_str'] is not None:
                        rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['in_reply_to_user_id_str'])
    c = collections.Counter(rt_user_list)
    del rt_user_list
    to_from_freq_list = []
    for k in c.most_common():
        to_from_freq = k[0].split(',')
        to_from_freq.append(k[1])
        to_from_freq_list.append(to_from_freq)
    g = Graph.TupleList(to_from_freq_list, weights=True, directed=True)
    g.save(save_filename + start_date.strftime('%Y-%m-%d') + '_' + date.strftime('%Y-%m-%d'), format='gml')
    gs.append([g, start_date.strftime('%Y-%m-%d') + '_' + date.strftime('%Y-%m-%d')])
    return gs

def walk_dir(path_origin):
    paths = []
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            path = os.path.join(pathname, filename)
            paths.append(path)
    return paths

def plot_tweet_count(path_origin):
    day = '01'
    c = 0
    y_datas = []
    x_datas = []
    path = None
    p_filename = None
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            if filename[8:10] != day:
                print('--------')
                y_datas.append(c)
                x_datas.append(p_filename[:10])
                c = 0
                day = filename[8:10]
            print(filename)
            path = os.path.join(pathname, filename)
            c += int(count_lines(path))
            p_filename = filename
    y_datas_average = []
    y_datas_average.append(y_datas[0])
    y_datas_average.append(y_datas[1])
    y_datas_average.append(y_datas[2])
    y_datas_average.append(y_datas[3])
    for i in range(4, len(x_datas)):
        y_datas_average.append((y_datas[i] + y_datas[i-1] + y_datas[i-2] + y_datas[i-3] + y_datas[i-4]) / 5)
    plt.plot(x_datas, y_datas_average)
    xt = [30 * i for i in range(15)]
    plt.xticks(xt, rotation=90)
    plt.show()

def calc_ration_over_communities(paths, g):
    twi_count = 0
    twi_count_target = 0
    for path in paths:
        #print(path)
        with open(path, 'r') as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                #if tweet['is_quote_status']:
                if tweet['in_reply_to_user_id_str']:
                    #twi_count += 1
                    try:
                        source = g.vs.find(name=tweet['user']['id_str'])
                        #target = g.vs.find(name=tweet['quoted_status']['user']['id_str'])
                        target = g.vs.find(name=tweet['in_reply_to_user_id_str'])
                        if source['cluster'] == target['cluster']:
                            twi_count_target += 1
                            twi_count += 1
                        else:
                            twi_count += 1
                    except:
                        pass
    return twi_count_target / twi_count
    
def scan_tweet(paths):
    for path in paths:
        with open(path) as f:
            tweet = json.loads(f.readline().strip())

def extract_image_urls(tweet):
    if 'extended_entities' in tweet.keys():
        for media in tweet['extended_entities']['media']:
            if media['type'] == 'photo':
                url = media['media_url_https']
    elif 'extended_entities' in tweet['retweeted_status'].keys():
        pass

def extact_random(paths, max_count=1000, prob=0.001):
    quoted = []
    quote = []
    quote_ids = []
    count = 0
    for path in paths:
        if count > max_count:
            break
        with open(path, 'r') as f:
            while True:
                t = f.readline().strip()
                if not t:
                    break
                if count > max_count:
                    break
                t = json.loads(t)
                if 'quoted_status' in t.keys():
                    if random.random() < prob:
                        if t['text'][:2] != 'RT':
                            try:
                                quoted.append(t['quoted_status']['text'])
                                quote.append(t['text'])
                                quote_ids.append(t['id_str'])
                                count += 1
                                print(count)
                            except:
                                print('e')
    return pd.DataFrame({'quote_id_str': quote_ids, 'quoted': quoted, 'quote': quote})

def exped(paths):
    count = 0
    for path in paths:
        print(path)
        with open(path, 'r') as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if 'quoted_status' in tweet.keys():
                    if tweet['text'][:2] != 'RT':
                        count += 1
    print(count)

def extact(paths):
    quoted = []
    quote = []
    quote_ids = []
    count = 0
    for path in paths:
        with open(path, 'r') as f:
            while True:
                t = f.readline().strip()
                if not t:
                    break
                t = json.loads(t)
                if 'quoted_status' in t.keys():
                    if t['text'][:2] != 'RT':
                        try:
                            quoted.append(t['quoted_status']['text'])
                            quote.append(t['text'])
                            quote_ids.append(t['id_str'])
                            count += 1
                            print(count)
                        except:
                            print('e')
    return pd.DataFrame({'quote_id_str': quote_ids, 'quoted': quoted, 'quote': quote})




if __name__ == "__main__":
    paths = walk_dir('/home/narita/covid-07_09')
    g = build_network(paths, 'covid_graphs/', mode='retweet', days=7)
    
    #g = Graph.Read_GML('kokuso_clusters_rp_05')
    #df = extact_random(paths, max_count=3000, prob=0.005)
    #df.to_csv('kokuso_anno.csv')