import sys
import webbrowser

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = '/Users/shougo/Desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/cluster_' + str(sys.argv[1]) + '/top_influencers.txt'
    elif len(sys.argv) == 3:
        path = '/Users/shougo/Desktop/twitter/Twitter/twitter_data/2021_mar_all_clusters/cluster_' + str(sys.argv[1]) + '/cluster_' + str(sys.argv[1]) + '_' + str(sys.argv[2]) +  '/top_influencers.txt'
    else:
        sys.exit()
    with open(path) as f:
        sns = f.readlines()
        for sn in sns:
            sn = sn.split(',')[0]
            print(sn)
            webbrowser.open('https://twitter.com/' + str(sn))