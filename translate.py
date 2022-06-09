from googletrans import Translator
import os
import pandas as pd
import random
import requests

def main():
    translated_file = 'en-jp_deepl.csv'
    jp_file = '/Users/shougo/Downloads/jp.csv'
    df_translated = pd.read_csv(translated_file)
    df_jp = pd.read_csv(jp_file)
    info = {}
    info['文書ID'] = []
    info['ラベル'] = []
    info['文書'] = []
    info['正解'] = []
    i_set = set()
    for i in range(1, 601):
        while(True):
            rand_num = random.randrange(600)
            if rand_num not in i_set:
                break
        info['文書ID'].append(i)
        info['ラベル'].append('')
        i_set.add(rand_num)
        print(i)
        if rand_num > 308:
            info['文書'].append(df_translated['文書'][rand_num - 308])
            info['正解'].append(1)
        else:
            info['文書'].append(df_jp['文書'][rand_num])
            info['正解'].append(0)
    df = pd.DataFrame(info)
    df = df.set_index('文書ID')
    df.to_csv("anno_deepl.csv")
    


def translation(file_path):
    df = pd.read_csv(file_path)
    url = 'https://api-free.deepl.com/v2/translate'
    for i in range(len(df['文書'])):
        print(i)
        text = df['文書'][i]
        params = {
            'auth_key' : '6d89c86a-7f7f-2d2e-2d15-a805fe36b205:fx',
            'target_lang' : 'JA',
            'source_lang' : 'EN',
            'text' : text,
        }
        r = requests.get(url, params=params).json()
        df['文書'][i] = r['translations'][0]['text']
    df.to_csv("en-jp_deepl.csv")



if __name__ == "__main__":
    main()
    #translation('/Users/shougo/Downloads/en.csv')