# coding: utf-8
#import pandas as pd
#import numpy as np
import re
from requests.api import get
from igraph import *
import collections
import leidenalg as la
import igraph
import requests
#import matplotlib.pyplot as plt
import json
import sys
#import sqlite3
import time
import faulthandler
import collections
import os
"""
def extract_users(path_origin, file_name='rt_users'):
    g = Graph(directed=True)
    g.es['weights'] = 0
    c = collections.Counter()
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in filenames:
            path = os.path.join(pathname, filename)
            print(path)
            with open(path, 'r') as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    t = json.loads(t)
                    c[t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str']] += 1
    with open(file_name, 'wt') as f:
        for k in c.keys():
            tf = k.split(',')
            to = tf[0]
            frm = tf[1]
            f.write(str(to) + ',' + str(frm) + ',' + str(c[k]))
def extract_unique_users(path_origin, size, file_name='unique_users'):
    s = set()
    for i in range(size):
        print(i)
        path = path_origin + str(i)
        with open(path, 'r') as f:
            while True:
                t = f.readline()
                if not t:
                    break
                t = t.strip()
                t = json.loads(t)
                s.add(t['user']['id_str'])
                s.add(t['retweeted_status']['user']['id_str'])
    with open(file_name, 'wt') as f:
        for x in s:
            f.write(x + '\n')
"""
def extract_users(path_origin, file_name='rt_users'):
    c = collections.defaultdict(int)
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in filenames:
            path = os.path.join(pathname, filename)
            print(path)
            with open(path, 'r') as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    try:
                        t = json.loads(t)
                    except Exception as e:
                        print(e)
                        print(t)
                    try:
                        c[t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str']] += 1
                    except Exception as e:
                        print(e)
                        print(t)
    with open(file_name, 'wt') as f:
        for k, v in c.items():
            tf = k.split(',')
            to = tf[0]
            frm = tf[1]
            f.write(str(to) + ' ' + str(frm) + ' ' + str(v) + '\n')
def extract_users_to_db(path_origin):
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE twitter(name text primary key, freq INTEGER)")
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in filenames:
            path = os.path.join(pathname, filename)
            print(path)
            with open(path, 'r') as f:
                name = ""
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    try:
                        t = json.loads(t)
                    except Exception as e:
                        print(e)
                        print(t)
                    try:
                        name = t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str']
                        ss = "INSERT INTO twitter(name, freq) values('" + str(name) + "', 1) on conflict(name) do update set freq=freq+1"
                        cursor.execute(ss)
                    except Exception as e:
                        print(e)
    conn.commit()
    cursor.close()
    conn.close()
"""
def extract_users(path_origin, file_name='rt_users'):
    c = collections.Counter()
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in filenames:
            path = os.path.join(pathname, filename)
            print(path)
            with open(path, 'r') as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    try:
                        t = json.loads(t)
                    except Exception as e:
                        print(e)
                        print(t)
                        sys.exit()
                    try:
                        c[t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str']] += 1
                    except Exception as e:
                        print(e)
                        print(t)
    with open(file_name, 'wt') as f:
        for k in c.keys():
            tf = k.split(',')
            to = tf[0]
            frm = tf[1]
            f.write(str(to) + ',' + str(frm) + ',' + str(c[k]))
"""
def build_network(rt_path, file_name='graph'):
    tff_list = []
    with open(rt_path, 'r') as f:
        while True:
            t = f.readline()
            if not t:
                break
            t = t.strip().split(',')
            tff = [t[0], t[1], t[2]]
            tff_list.append(tff)
    g = Graph.TupleList(tff_list, weights=True, directed=True)
    g.save(file_name, format='gml')
    summary(g)
    return g

"""
def build_network(unique_path, user_path, save_filename='graph'):
    g = Graph(directed=True)
    g.es['weight'] = 0
    with open(unique_path, 'r') as f:
        while True:
            t = f.readline()
            if not t:
                break
            t = t.strip()
            g.add_vertex(t)
    with open(user_path, 'r') as f:
        i = 0
        while True:
            t = f.readline()
            if i % 1000 == 0:
                print(i)
            if not t:
                break
            t = t.strip()
            t = t.split(',')
            #g.add_edges([(t[0], t[1])])
            #g.es['weight'][i] = t[2]
            i += 1
    #g = Graph.TupleList(to_from_freq_list, weights=True, directed=True)
    g.save(save_filename, format='gml')
    summary(g)
    return g
"""
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

def clustering(g, resolution_parameter=1, n_iterations=-1):
    partition = la.find_partition(g, la.RBConfigurationVertexPartition, weights=g.es['weight'], n_iterations=n_iterations, resolution_parameter=resolution_parameter)
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
        if res.status_code == requests.codes.ok:
            res = res.json()
            screen_name = res['screen_name']
        elif res.status_code == 429 or res.status_code == 420:
            time.sleep(9600)
            res = requests.get(url, headers=headers)
            if res.status_code == requests.codes.ok:
                res = res.json()
                screen_name = res['screen_name']
            else:
                screen_name = None
        else:
            screen_name = None
    except:
        screen_name = None
    return screen_name



def cal_strength(g):
    return g.strength(g.vs, mode='in', loops=False, weights=g.es['weight'])

def sort_max_print(subg, centrality, guest_token, len=10):
    cen_sorted = sorted(centrality, reverse=True)
    sns = []
    for i in range(len):
        ind = centrality.index(cen_sorted[i])
        name = subg.vs[ind]['name']
        sn = id_to_screen_name(name, guest_token=guest_token)
        print(sn)
        if sn is not None:
            sn = sn + ',' + name
            sns.append(sn)
    return sns

    


if __name__ == "__main__":
    faulthandler.enable()
    #extract_users_to_db('/mnt/0202tmp/all-tweets')
    #extract_unique_users('/Users/shougo/Downloads/narita-mar/keyword-search.tweet.', 8, file_name='unique_test')
    #build_network(unique_path='unique_test', user_path='test', save_filename='graph_test')
    #g = Graph.Read_Ncol('2021_mar_rts')
    #g = Graph.Read_Ncol('test')
    #g = Graph.Read_GML('2021_mar_all')
    #print('read: finished')
    #g = Graph.Read_GML('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/cluster_1/cluster_1_3/2021_mar_all_cluster_1_3')
    #g = Graph.Read_GML('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/cluster_22/2021_mar_all_cluster_22')
    #g = Graph.Read_GML('2020_apr_1_rp_1_clusters')
    #g = Graph.Read_GML('2021_mar_all_clusters/')
    g_apr = Graph.Read_Ncol('2020_apr_clusters')
    with open('2020_apr_membership', 'r') as f:
        g_apr.vs['cluster'] = [float(s.strip()) for s in f.readlines()]
    summary(g_apr)
    p = clustering(g_apr, resolution_parameter=0)
    print(len(p))
    if len(p) != 1:
        print('z')
        sub = subgraph(g_apr, p[0])
        p = clustering(sub)
        sub.vs['cluster'] = p.membership
        sub.save('2020_apr_gcc', format='ncol')
        with open('2020_mar_gcc_membership', 'wt') as f:
            for m in p.membership:
                f.write(str(m) + '\n')
        
    """
    p = clustering(g)
    gt = get_guest_token()
    for i in range(10):
        sv = p[i]
        sub = subgraph(g, sv)
        print(i)
        summary(sub)
        st = cal_strength(sub)
        sort_max_print(sub, st, gt, len=10)
        print('\n')
    """
    """
    #g.vs['cluster'] = p.membership
    summary(g)
    gt = get_guest_token()
    for i in range(10):
        sv = g.vs.select(lambda vertex : vertex['cluster'] == i)
        st = g.strength(sv, mode='out', loops=False, weights=g.es['weight'])
        stin = g.strength(sv, mode='in', loops=False, weights=g.es['weight'])
        sub = subgraph(g, sv)
        stsub = sub.strength(sub.vs, mode='out', loops=False, weights=sub.es['weight'])
        sort_max_print(sub, stin, gt, len=8)
        print(sum(stsub)/sum(st))
        print('\n')
    """
    """
    gt = get_guest_token()
    for i in range(10):
        sv = g.vs.select(lambda vertex : vertex['cluster'] == i)
        sub = subgraph(g, sv)
        st = cal_strength(sub)
        st_sorted = sorted(st, reverse=True)
        ind = st.index(st_sorted[0])
        name = sub.vs[ind]['name']
        sn = id_to_screen_name(name, guest_token=gt)
        if sn is None:
            ind = st.index(st_sorted[1])
            name = sub.vs[ind]['name']
            sn = id_to_screen_name(name, guest_token=gt)
        tv = g.vs.find(name=name)
        tv['label'] = sn
        print(str(i) + ': ' + sn)

    i = 0
    svs = []
    while True:
        sv = g.vs.select(lambda vertex : vertex['cluster'] == i)
        #if len(sv) < 50000:
        if i == 6:
            break
        if i == 4:
            i += 1
            continue
        svs.extend(sv)
        i += 1
    subg = subgraph(g, svs)
    summary(subg)
    pal = igraph.drawing.colors.ClusterColoringPalette(30)
    strength = g.strength(g.vs, mode='in', loops=False, weights=g.es['weight'])
    visual_style = {}
    #visual_style["vertex_size"] = 3
    visual_style["vertex_size"] = [2 if i ==0 else int(i**0.3)+2 for i in strength]
    visual_style["vertex_frame_width"] = 0
    visual_style["edge_width"] = [int((x/2)**0.3) * 0.1 for x in subg.es["weight"]]
    visual_style["vertex_color"] = pal.get_many(subg.vs['cluster'])
    #visual_style["edge_width"] = 1
    visual_style["vertex_label"] = subg.vs["label"]
    visual_style['edge_arrow_size'] = 1/150
    visual_style['edge_arrow_width'] = 1/100
    #visual_style['vertex_shape'] = 'hidden'
    visual_style['layout'] = "fr"
    visual_style["bbox"] = (1200, 1200)
    #layout = g.layout_fruchterman_reingold(grid=True)
    print('drawing')
    plot(subg, '2021_mar_1.png', **visual_style)
    #plot(partition, 'fr.png', **visual_style)
    """
    """
    #time.sleep(10)
    print('clustering: start')
    p = clustering(g)
    print('clustering: finished')
    gt = get_guest_token()
    for i in range(5):
        print('cluster:' + str(i))
        sub = subgraph(g, p[i])
        summary(sub)
        strength = cal_strength(sub)
        sort_max_print(sub, strength, gt, len=5)
        print('\n')
    #with open('2021_mar_all_membership', 'wt') as f:
    #    for x in p.membership:
     #       f.write(str(x) + '\n')
    g.vs['cluster'] = p.membership
    del p
    i = 0
    """
    """
    with open('2021_mar_all_membership', 'r') as f:
        g.vs['cluster'] = [s.strip() for s in f.readlines()]
    os.makedirs('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters', exist_ok=True)
    for i in range(60):
        sub = subgraph(g, g.vs.select(lambda vertex : vertex['cluster'] == str(i)))
        if len(sub.vs['name']) >= 30000:
            sub.save('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_' + str(i), format='gml')
            if len(sub.vs['name']) >= 500000:
                p = clustering(sub)
                os.makedirs('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_' + str(i) + '_sub', exist_ok=True)
                for j in range(len(p)):
                    if len(p[j]) >= 30000:
                        sub_sub = subgraph(sub, p[j])
                        sub_sub.save('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_' + str(i) + '_sub/2021_mar_all_cluster_' + str(i) + '_' + str(j), format='gml')
                    else:
                        break
        else:
            break
    for pathname, dirnames, filenames in os.walk('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/'):
        for filename in filenames:
            path = os.path.join(pathname, filename)
            print(path)
            g = Graph.Read_GML(path)
            strength = cal_strength(g)
            gt = get_guest_token()
            sns = sort_max_print(g, strength, gt)
            print('\n')
            with open(os.path.join(pathname, 'top_influencers.txt'), 'wt') as f:
                f.write('\n'.join(sns))
    
    """
    """
    with open('2021_mar_all_membership', 'r') as f:
        g.vs['cluster'] = [s.strip() for s in f.readlines()]
    gt = get_guest_token()
    for i in range(60):
        print('cluster:' + str(i))
        sub = subgraph(g, g.vs.select(lambda vertex : vertex['cluster'] == str(i)))
        #strength = sub.strength(sub.vs, mode='in', loops=False, weights=sub.es['weight'])
        #sns = sort_max_print(sub, strength, gt, len=6)
        #with open('2021_mar_all_cluster_' + str(i), 'wt') as f:
            #f.write('\n'.join(sns))
        sub.save('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_' +str(i), format='gml')

    #g.save('2021_mar_all_clusters', format='gml')
    """
    """
    g = Graph.Read_GML('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_0')
    print('clustering: start')
    p = clustering(g)
    g.vs['cluster'] = p.membership
    g.save('/Users/shougo/desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/2021_mar_all_cluster_0_sub/2021_mar_all_cluster_0_sub_clusters', format='gml')
    """