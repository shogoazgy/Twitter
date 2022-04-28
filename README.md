# Twitter
Twitter全量データの一部を扱う上で気をつけること
## 解凍
***.tar.lz4という形になってるはず。tar.gzとかならよく見るがtar.lz4は見慣れない形式(自分はこの時に初めて見た)。まあでもLinuxやmacOSでならtarコマンドで解凍可能(windowsもかな?)。

``` bash
$  tar -xf ***.tar.lz4
```

以下は見慣れない拡張子に惑わされ、一度lz4を解凍してtarにして展開などという遠回りをした最初の自分のやり方。
まずlz4コマンドで解凍。
``` bash
$ lz4 -d [input] [output] 
```

lz4が入ってないならaptなりbrewなりでインストール。
macなら

``` bash
$ brew install lz4
```

Linuxなら多分

``` bash
$ sudo apt install liblz4-tool
```

windowsは知らないが、恐らくlz4解凍できるアプリケーション必要かな。  
その後で

``` bash
$  tar -xf ***.tar
```


## データ形式  
実際にデータを見てみるとわかるが一行ずつjson形式で書き込まれている。1ファイルあたり100万行で区切られている。  
読み込む方法をしては一行ずつパースしていくか、pythonならpandasでも適切にやると一気にdataframeで読み込める。  
ツイートidやユーザidなんかは数値にすると型によっちゃ丸められたりするので基本文字列で読み込む。  
Twitterデータの時間の形式は少し独特なので注意。以下は twitter_data/data.py の時間の表示形式の変更の関数。

``` python
def change_time(created_at, only_date=True):
    st = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')        
    utc_time = datetime.datetime(st.tm_year, st.tm_mon,st.tm_mday, st.tm_hour,st.tm_min,st.tm_sec, tzinfo=datetime.timezone.utc)   
    jst_time = utc_time.astimezone(pytz.timezone("Asia/Tokyo"))
    if only_date:
        str_time = jst_time.strftime("%Y-%m-%d")                    
    else:
        str_time = jst_time.strftime("%Y-%m-%d_%H:%M:%S")                    
    return str_time
```

一度UTCにし、さらに日本標準時(JST)に変換したものを返している。
## メモリ
Twitterの全量データなだけあって、どんなクエリでデータを絞ったかにもよるが恐らくかなり大きなデータファイルとなっているはず。1ファイルあたり100万行なのでまともに全部読み込もうとするとスペックにもよるがメモリに乗らない可能性あり。そうなると処理が重たくなるのでストリーム処理で少しずつ読み込んで処理したり必要な属性のデータだけ取り出していくことになる。
以下は twitter_data/data.py の一部でpythonでやる場合の例。

``` python data.py {.line-number}
with open(path) as f:
    while True:
        t = f.readline()
        if not t:
            break
        t = t.strip()
        t = json.loads(t)
```

一行目でファイルオブジェクトを作成し、`readline()`で一行ずつ読み込んでそれを終わりまで回し続けている。`readlines()`だとファイルの中身全てを一気に行ごとの配列として読み込んじゃうので注意。
ただこの書き方だと多分遅いから他の適切なやり方あるなら教えていただきたいくらい。まるまるメモリに乗るなら恐らく一気に読み込んだ方がこれより速いはず。  
pandasだと`chunke_size`で一度に読み込む大きさ指定できるからそれでもいいかな?
他の言語でもメモリに気をつけて。  

# グラフを扱うライブラリ(python)
もし可視化など何らかの形でTwitterデータからグラフを作成することになった場合のおすすめのpythonのライブラリは`python-igraph`  
pythonでのグラフを扱う代表的なライブラリは`networkx`なんてのもあるがこちらは100万以上ノード規模になると遅すぎるので注意。他にもグラフDBの`neo4j`なんかも良さげ(使ってみてはない)。

