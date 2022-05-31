#from importlib.resources import path
from html import entities
from statistics import mode
from tkinter.messagebox import NO
from unicodedata import name
from unittest import result
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

def cal_strength(g_list):
    strength_list = []
    for g in g_list:
        strength_list.append(g.strength(g.vs, mode='in', loops=False, weights=g.es['weight']))
    return strength_list

def sort_max_print(subg, centrality, guest_token, len=10):
    cen_sorted = sorted(centrality, reverse=True)
    for i in range(len):
        ind = centrality.index(cen_sorted[i])
        print(id_to_screen_name(subg.vs[ind]['name'], guest_token=guest_token))

def build_network(paths, save_filename, mode='quoted'):
    rt_user_list = []
    c = 0
    for path in paths:
        print(path)
        with open(path, 'r') as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if mode == 'quoted':
                    if tweet['is_quote_status'] == True:
                        try:
                            rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['quoted_status']['user']['id_str'])
                        except:
                            c += 1
                elif mode == 'retweet':
                    rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['retweeted_status']['user']['id_str'])
                elif mode == 'reply':
                    if tweet['in_reply_to_user_id_str'] is not None:
                        rt_user_list.append(tweet['user']['id_str'] + ',' + tweet['in_reply_to_user_id_str'])
    print('無効なツイート: ' + str(c))
    c = collections.Counter(rt_user_list)
    del rt_user_list
    to_from_freq_list = []
    for k in c.most_common():
        to_from_freq = k[0].split(',')
        to_from_freq.append(k[1])
        to_from_freq_list.append(to_from_freq)
    g = Graph.TupleList(to_from_freq_list, weights=True, directed=True)
    g.save(save_filename, format='gml')
    return g

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
        print(path)
        with open(path, 'r') as f:
            while True:
                tweet = f.readline().strip()
                if not tweet:
                    break
                tweet = json.loads(tweet)
                if tweet['is_quote_status']:
                    twi_count += 1
                    try:
                        source = g.vs.find(name=tweet['user']['id_str'])
                        target = g.vs.find(name=tweet['quoted_status']['user']['id_str'])
                        if source['cluster'] == target['cluster']:
                            twi_count_target += 1
                    except:
                        pass
    return twi_count_target / twi_count
    

if __name__ == "__main__":
    #g = Graph.Read_GML('2020_07_quoted_clusters')
    all_paths = walk_dir('/home/narita/2020-ex-rt-jp')
    paths = []
    pre_month = ''
    rt_result = {}
    quoted_result = {}
    reply_result = {}
    info_list = ['term', 'node', 'edge', 'sum edge', 'modurality', 'community edge ratio']
    for info in info_list:
        rt_result[info] = []
        quoted_result[info] = []
        reply_result[info] = []
    #quoted_result['community edge ratio (RT network)'] = []
    #reply_result['community edge ratio (RT network)'] = []
    """
    paths = walk_dir('/home/narita/Twitter/graphs/RT')
    for path in paths:
        if path[-1] == 's':
            g = Graph.Read_GML(path)
            p = la.ModularityVertexPartition(g,weights=g.es['weight'], initial_membership=[int(i) for i in g.vs['cluster']])
            summary(g)
            rt_result['term'].append('2020_' + path[-12:-10])
            rt_result['node'].append(len(g.vs))
            rt_result['edge'].append(len(g.es))
            rt_result['RT'].append(sum(g.strength(g.vs, weights=g.es['weight'], mode='out')))
            rt_result['modurality'].append(p.quality())
            sum_in = 0
            sum_all = 0
            for edge in g.es:
                sum_all += edge['weight']
                if g.vs[edge.target]['cluster'] == g.vs[edge.source]['cluster']:
                    sum_in += edge['weight']
            rt_result['community edge ratio'].append(sum_in / sum_all)
    df_rt = pd.DataFrame(rt_result)
    df_rt = df_rt.set_index('term')
    df_rt.to_csv("rt.csv", encoding="shift_jis")
    #paths = walk_dir('/home/narita/Twitter/graphs/reply')
    """
    for path in all_paths:
        if path[-12:-10] != pre_month:
            if paths != []:
                g = build_network(paths, save_filename='/home/narita/Twitter/graphs/quoted/2020_' + pre_month + '_quoted', mode='quoted')
                print('quoted clustering')
                p = clustering(g)
                print('quoted clustering end')
                g.vs['cluster'] = p.membership
                save_gml(g, '/home/narita/Twitter/graphs/quoted/2020_' + pre_month + '_quoted_clusters')
                summary(g)
                quoted_result['term'].append('2020_' + pre_month)
                quoted_result['node'].append(len(g.vs))
                quoted_result['edge'].append(len(g.es))
                quoted_result['sum edge'].append(sum(g.strength(g.vs, weights=g.es['weight'], mode='out')))
                quoted_result['modurality'].append(p.quality())
                sum_in = 0
                sum_all = 0
                for edge in g.es:
                    sum_all += edge['weight']
                    if g.vs[edge.target]['cluster'] == g.vs[edge.source]['cluster']:
                        sum_in += edge['weight']
                quoted_result['community edge ratio'].append(sum_in / sum_all)
                g = build_network(paths, save_filename='/home/narita/Twitter/graphs/reply/2020_' + pre_month + '_reply', mode='reply')
                summary(g)
                print('reply clustering')
                p = clustering(g)
                print('reply clustering end')
                g.vs['cluster'] = p.membership
                save_gml(g, '/home/narita/Twitter/graphs/reply/2020_' + pre_month + '_reply_clusters')
                reply_result['term'].append('2020_' + pre_month)
                reply_result['node'].append(len(g.vs))
                reply_result['edge'].append(len(g.es))
                reply_result['sum edge'].append(sum(g.strength(g.vs, weights=g.es['weight'], mode='out')))
                reply_result['modurality'].append(p.quality())
                sum_in = 0
                sum_all = 0
                for edge in g.es:
                    sum_all += edge['weight']
                    if g.vs[edge.target]['cluster'] == g.vs[edge.source]['cluster']:
                        sum_in += edge['weight']
                reply_result['community edge ratio'].append(sum_in / sum_all)
                print(pre_month)
                paths = []
        pre_month = path[-12:-10]
        paths.append(path)
    df_quoted = pd.DataFrame(quoted_result)
    df_quoted = df_quoted.set_index('term')
    df_quoted.to_csv("quoted.csv", encoding="shift_jis")
    df_reply = pd.DataFrame(reply_result)
    df_reply = df_reply.set_index('term')
    df_reply.to_csv("reply.csv", encoding="shift_jis")