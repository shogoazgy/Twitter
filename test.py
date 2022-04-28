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
    