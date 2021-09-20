import pandas as pd
import numpy as np
from igraph import *
import collections
import leidenalg as la
import igraph
import requests
import matplotlib.pyplot as plt

def build_network(rt_users_path, save_filename='graph'):
    with open(rt_users_path, 'r') as f:
        rt_user_list = [s.strip() for s in f.readlines()]
    print(len(rt_user_list))
    c = collections.Counter(rt_user_list)
    to_from_freq_list = []
    del rt_user_list
    for k in c.most_common():
        if k[1] <= 1:
            break
        to_from_freq = k[0].split(',')
        #to_from_freq[0], to_from_freq[1] = to_from_freq[1], to_from_freq[0]
        to_from_freq.append(k[1])
        to_from_freq_list.append(to_from_freq)
    g = Graph.TupleList(to_from_freq_list, weights=True, directed=True)
    g.save(save_filename, format='gml')
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


if __name__ == "__main__":
    g = Graph.Read_GML('2021_mar_1_rp_1_clusters')
    
    gt = get_guest_token()
    for i in range(11):
        print('\nクラスタ:' + str(i))
        #sub = Graph.Read_GML('new_year_2_rp_1_cluster_' + str(i))
        sub = subgraph(g, g.vs.select(lambda vertex : vertex['cluster'] == i))
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