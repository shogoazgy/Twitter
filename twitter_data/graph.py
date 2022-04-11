from importlib.resources import path
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
from sklearn.linear_model import LinearRegression

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

def build_network(paths, save_filename):
    rt_user_list = []
    for path in paths:
        print(path)
        print(len(rt_user_list))
        with open(path, 'r') as f:
            while True:
                t = f.readline()
                if not t:
                    break
                t = t.strip()
                t = json.loads(t)
                try:
                    rt_user_list.append(t['user']['id_str'] + ',' + t['retweeted_status']['user']['id_str'])
                except:
                    pass
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

def walk_dir(path_origin, since=None, until=None):
    paths = []
    x = []
    y = []
    month = '02'
    for pathname, dirnames, filenames in os.walk(path_origin):
        for filename in sorted(filenames, key=str):
            x.append(filename[:10])
            path = os.path.join(pathname, filename)
            y.append(count_lines(path))
            if filename[5:7] != month:
                print(filename)
                print('--reset--')
                build_network(paths, filename[:5] + month)
                month = filename[5:7]
                paths = []
                paths.append(path)
            else:
                print(filename)
                paths.append(path)
    return [x, y]

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




if __name__ == "__main__":
    rt_counts = []
    for pathname, dirnames, filenames in os.walk('/Users/shougo/Downloads/narita2021'):
        for filename in sorted(filenames, key=str):
            rt_count = 0
            path = os.path.join(pathname, filename)
            with open(path, 'r') as f:
                while True:
                    t = f.readline()
                    if not t:
                        break
                    t = t.strip()
                    t = json.loads(t)
                    if 'retweeted_status' in t.keys():
                        rt_count += 1
                print(rt_count)
                rt_counts.append(rt_count)
    with open("rt_counts_2021.pickle", mode="wb") as f:
        pickle.dump(rt_counts, f)

    """
    with open("x.pickle", mode="rb") as f:
        x = pickle.load(f)
    with open("y.pickle", mode="rb") as f:
        y_t = pickle.load(f)
    #y_c = pd.read_csv('/Users/shougo/Downloads/newly_confirmed_cases_daily.csv')['ALL'][16:714]
    y_c = pd.read_csv('/Users/shougo/Downloads/newly_confirmed_cases_daily.csv')['ALL'][16:689]
    x_i = [i for i in range(len(x))]
    x_label = []
    t = ['2/1\n2020', '4/1', '5/31', '7/30', '9/28', '11/27', '1/24', '3/25', '5/24', '7/23', '9/21', '11/20']
    j = 0
    for i in range(len(x) - 25):
        if i % 60 == 0:
            x_label.append(t[j])
            #x_label.append(x[i])
            j += 1
        elif i == 333:
            x_label.append('\n2021')
        else:
            x_label.append('')
    plt.plot(x_i[:671], y_c[:671])
    plt.grid(b=True, which="major", color="#ababab", linestyle="-", axis="y")
    #plt.tick_params(labelbottom=False)
    plt.xticks(x_i[:671], x_label[:671])
    plt.xlabel('Date')
    plt.ylabel('The number of new cases of infection')
    

    #plt.show()
    plt.savefig('num_infection.pdf', bbox_inches="tight", pad_inches=0.05) # 'test.png'はpath+保存名です、figはplt.figureのオブジェクトです
    """
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(x_i[:671], y_t[:671], label='The number of tweets')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(x_i[:671], y_c[:671], 'r', label='The number of new cases of infection')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center')

    ax1.set_xlabel('Data')
    ax1.set_ylabel('The number of tweets')
    ax1.grid(True)
    ax2.set_ylabel('The number of new cases of infection')
    plt.xticks(x_i[:671], x_label[:671])
    #plt.show()
    plt.savefig('num_tweets_infection.pdf', bbox_inches="tight", pad_inches=0.05) # 'test.png'はpath+保存名です、figはplt.figureのオブジェクトです
    
    start_dif = 29
    start = 480
    window = 160
    y_c = y_c[start + start_dif:start + window + start_dif]
    y_t = y_t[start + start_dif:start + window + start_dif]
    x = x[start + start_dif:start + window + start_dif]

    y_c_m = [[t] for t in y_c]
    y_t_m = [[t] for t in y_t]
    x_m = [[i] for i in range(len(y_t))]

    model_t = LinearRegression()
    model_t.fit(x_m, y_t_m)
    model_c = LinearRegression()
    model_c.fit(x_m, y_c_m)

    
    y_t_n = model_t.predict(x_m)
    y_c_n = model_c.predict(x_m)
    
    y_t_avg = [y_t[i] - y_t_n[i][0] for i in range(len(x_m))]
    y_c_avg = [y_c[i + start + 16 + start_dif] - y_c_n[i][0] for i in range(len(x_m))]

    fig = plt.figure()
    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, y_t_avg, label='tweet')
    ax2.plot(x, y_c_avg, c="r", label='corona')
    xt = [i for i in range(0, len(y_t), 45)]
    plt.legend()
    #plt.plot(x, y_t_avg)
    #plt.plot(x, y_c_avg)
    #print(y_c)
    s1 = pd.Series(y_t)

    s2 = y_c
    corr = s1.corr(s2)
    print(corr)

    s1 = pd.Series(y_t_avg)
    s2 = pd.Series(y_c_avg)
    corr = s1.corr(s2)
    print(corr)

    plt.xticks(xt, rotation=90)
    #plt.show()
    
    g = Graph.Read_GML('/Users/shougo/Desktop/twitter/Twitter/twitter_data/graphs/2020-04')
    summary(g)
    g.save('2020_04', format='ncol')
    
    paths = walk_dir('/Users/shougo/Downloads/narita2021/')
    x, y = paths
    with open("x2.pickle", mode="wb") as f:
        pickle.dump(x, f)
    with open("y2.pickle", mode="wb") as f:
        pickle.dump(y, f)

    
    #plot_tweet_count('/Users/shougo/Downloads/data/str01_03/twitter/shohei/result/PFTqp4cunv2kwvy8rsp_qp4cu7c1kwwyyveo')
    g = Graph.Read_GML('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2020-03')
    #vs = g.clusters(mode='weak')
    summary(g)
    p = clustering(g, resolution_parameter=0)
    print(len(p))
    print(p[1])
    sub = subgraph(g, p[0])
    summary(sub)
    #p = la.find_partition(sub, la.ModularityVertexPartition, weights=sub.es['weight'], n_iterations=-1)
    p = clustering(sub)
    sub.vs['cluster'] = p.membership
    sub.save('2021_03_clusters', format='gml')


    #g = Graph.Read_GML('2021_mar_1')
    g = build_network('rt_users_jan.txt')
    summary(g)
    #g = build_network('rt_users_2020_apr.txt')
    p = clustering(g)
    g.vs['cluster'] = p.membership
    save_gml(g, '2021_jan_1_clusters')
    gt = get_guest_token()
    for i in range(11):
        print('\nクラスタ:' + str(i))
        #sub = Graph.Read_GML('new_year_2_rp_1_cluster_' + str(i))
        sub = subgraph(g, p[i])
        strength = sub.strength(sub.vs, mode='in', loops=False, weights=sub.es['weight'])
        sort_max_print(sub, strength, gt, len=7)
    
    fig = plt.figure(figsize=(8, 6))
    colors = ['k', 'k', 'k', 'r', 'g', 'b', 'k', 'c', 'k', 'k', 'k','k']
    c = []
    for i in range(11):
        cluster = g.vs.select(lambda vertex : vertex['cluster'] == i)
        sub = subgraph(g, cluster)
        deg = sub.degree(mode='in')
        for d in range(len(deg)):
            deg[d] += 1
        c.append(collections.Counter(deg))
    for i in range(11):
        s = len(g.vs.select(lambda vertex : vertex['cluster'] == i))
        x = []
        y = []
        for j in range(4):
            if j == 0:
                for k in range(1, 10):
                    x.append(k)
                    y.append(c[i][k]/s)
            else:
                for k in range(10**j, 10**(j+1), 10**j):
                    sum = 0
                    for l in range(k, k+10**j):
                        sum += c[i][k]
                    sum /= 10**j
                    x.append(k)
                    y.append(sum/s)
        plt.plot(x, y, lw=0.8, color=colors[i])
    plt.loglog(basex=10,basey=10)
    plt.xlabel("x",fontsize=20)
    plt.ylabel("ρ",fontsize=20)
    plt.title("in degree",fontsize=25)
    plt.show()
    #g = build_network('rt_user.txt', save_filename='new_year_2')
    #g = Graph.Read_GML('2020_apr_1_gcc')
    #partition = clustering(g)
    #g.vs['cluster'] = partition.membership
    #save_gml(g, '2021_apr_1_rp_1_clusters')
    #summary(g)
    #partition = clustering(g, resolution_parameter=1)
    partition = clustering(g, resolution_parameter=0.5)
    g.vs['cluster'] = partition.membership
    save_gml(g, filename='2020_mar_1_rp_1_clusters')

    #g_20_apr_2 = Graph.Read_GML('2020_apr_1_cluster_0')
    c_i = 3
    n_i = 9
    """

    """
    g_20_set_list = []
    for i in range(n_i):
        g_20_set_list.append(set(Graph.Read_GML('2020_apr_1_rp_1_cluster_' + str(i)).vs['name']))
    g_21_mar_t = Graph.Read_GML('2021_mar_1_rp_1_cluster_' + str(c_i))
    i_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    in_2020_user_list = []
    #print('gcc_users : ' + str(len(user_set)))
    g = Graph.Read_GML('2020_apr_1_gcc')
    user_set = set(g.vs['name'])
    del g
    in_2020_user_list = user_set & set(g_21_mar_t.vs['name'])
    print('21年3月 ' + cluster_21[c_i] + ' : ' + str(len(g_21_mar_t.vs)) + '  内 : ' + str(len(in_2020_user_list)) + ' 割合 : ' + format(len(in_2020_user_list)/len(g_21_mar_t.vs), '.3f'))
    del user_set
    del g_21_mar_t
    
    for v in in_2020_user_list:
        for i in range(n_i):
            if v in g_20_set_list[i]:
                i_list[i] += 1
                break
    

    for i in range(n_i):
        print('20年4月 ' + cluster_20[i] + ' : ' + format(100*i_list[i]/len(in_2020_user_list), '.3f') + '%')

    
    visual_style = {}
    visual_style["vertex_size"] = 4
    #visual_style["edge_width"] = [int((x/2)**0.5) for x in g.es["weight"]]
    visual_style["edge_width"] = 1
    visual_style['edge_arrow_size'] = 1/15
    visual_style['dge_arrow_width'] = 1/10
    visual_style["layout"] = "grid_fr"
    visual_style["vertex_shape"] = "circle"
    plot(partition, 'grid_fr.png', **visual_style)

    cluster_list = []
    for i in range(13):
        g = Graph.Read_GML('2020_apr_1_rp_1_cluster_' + str(i))
        #cluster = subgraph(g, partition[i]) 
        #save_gml(cluster, filename='2020_apr_rp_1_cluster_' + str(i))
        cluster_list.append(g)
    strength_list = cal_strength(cluster_list)
    for i in range(len(strength_list)):
        print('クラスタ' + str(i))
        sort_max_print(cluster_list[i], strength_list[i])
        print('\n')
    """