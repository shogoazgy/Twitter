from igraph import *
import umap
from matplotlib import pyplot as plt
#import umap.plot
import matplotlib.cm as cm
import os
import collections
from igraph import *
import igraph
import leidenalg as la

def subgraph(maingraph, vertices):
    sub = maingraph.induced_subgraph(vertices)
    return sub

def get_embedding(temp_path='/Users/shougo/Desktop/twitter/Twitter/twitter_data/temp_ncol'):
    embedding = []
    with open('/Users/shougo/Desktop/twitter/Twitter/twitter_data/embedding_file', 'r') as f:
        i = 0
        t = f.readline()
        while(True):
            t = f.readline()
            if not t:
                break
            i += 1
            #if i % 2 == 0:
            if i % 3 == 0 or i % 3 == 1:
                continue
            t = t.strip()
            t = t.split(' ')
            t = t[1:]
            t = [float(a) for a in t]
            if len(t) == 128:
                embedding.append(t)
    return embedding
def decide(g1, g2):
    d = collections.defaultdict()
    for i in range(10):
        sv1 = g1.vs.select(lambda vertex : vertex['cluster'] == i)
        sub1 = subgraph(g1, sv1)
        st1 = g1.strength(sv1, mode='in', loops=False, weights=g1.es['weight'])
        st1_sorted = sorted(st1, reverse=True)
        tp = []
        c = collections.Counter()
        for j in range(10):
            ind = st1.index(st1_sorted[j])
            tp.append(sub1.vs[ind]['name'])
        count = 0
        for t in tp:
            v = g2.vs.select(name=t)
        if len(v) == 1:
            count += 1
            c[v[0]['cluster']] += 1
        #print(c.most_common(1))
        #print(c.most_common(1)[0][1] / count)
        if c.most_common(1)[0][1] / count > 0.5 and count > 2:
            d[i] = int(float(c.most_common(1)[0][0]))
    return d

def dimension_reduction(embedding):
    mapper = umap.UMAP(random_state=42)
    X = mapper.fit_transform(embedding)
    return X

def temp_membership(g, membership_file='/Users/shougo/Desktop/twitter/Twitter/twitter_data/temp_membership', embedding_file='/Users/shougo/Desktop/twitter/Twitter/twitter_data/embedding_file'):
    d = collections.defaultdict()
    mem = []
    for v in g.vs:
        d[v['name']] = v['cluster']
    with open(embedding_file, 'r') as f:
        with open(membership_file, 'wt') as w:
            i = 0
            t = f.readline()
            while(True):
                t = f.readline()
                if not t:
                    break
                i += 1
                #if i % 2 == 0:
                if i % 3 == 0 or i % 3 == 1:
                    continue
                t = t.strip()
                t = t.split(' ')
                t = t[0]
                w.write(str(int(d[t])) + '\n')
                mem.append(int(d[t]))
    return mem

def plot_scatter(X, label, save_filename):
    plt.clf()
    plt.scatter(X[:,0],X[:,1], c=label ,cmap=cm.tab20, alpha=0.8, s=0.1)
    #plt.scatter(X[:,0],X[:,1], c=label , s=0.1)
    plt.colorbar()
    plt.savefig('pic/' + save_filename + '.png')

color_i = 9
#g_mar = Graph.Read_GML('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2021_mar_1_rp_1_clusters')
g_dec = Graph.Read_GML('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2020_dec_1_clusters')
vs = []
for i in range(10):
    vs.extend(g_dec.vs.select(lambda vertex : vertex['cluster'] == i))
sub = subgraph(g_dec, vs)
del vs
del g_dec
sub.save('/Users/shougo/Desktop/twitter/Twitter/twitter_data/2020_dec_ncol', format='ncol')
