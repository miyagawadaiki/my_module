# リファレンス

## `SParameter`クラス
継承してパラメータセットクラスを自作してもらう前提のクラス．基本的に`__init__()`をオーバーライドしてメンバの`pdict`という辞書に新たな変数を追加するだけで良い．

### フィールド
* `pdict` : 変数をまとめる辞書．`SParameter`の段階ではほぼ入ってないので，継承した際にここに追加すること．
* `sd` : 乱数発生のシード値．`__init__()`ではnumpyで発生させる乱数のシード値をこれに設定する．

### メソッド
* `__init__` : **オーバーライド必要アリ**．これの引数に変数の初期値を渡し，それを用いて`pdict`に登録処理をする．例はSample/Sample.ipynbを参照．
* `set_seed` : シード値を設定する．現状numpyで乱数発生させる想定しかしていない．
* `__str__`  : パラメータ名と設定された値からパラメータセットの文字列を生成する．ファイル名などで使う．float型変数は100倍して整数化された値が使われるので注意．
* `get_filename` : `__str__`を用いてファイル名を生成する．引数の`suf`は拡張子を指定する．
* `copy` : オブジェクトをコピーして返す．
* `update` : 変数名`key`, 値`val`を指定して辞書の合致するものを更新する．
* `update_from_dict` : 更新する内容が入った辞書`dict`を渡してパラメータの更新を行う．
* `update_from_argv` : コマンドライン引数の更新用部分を`sys.argv[1:]`などで渡すと更新してくれる．形式は`a=1 b=0.1`のように変数あたりスペース区切り，変数名と値は`=`区切り．

<br>

## `ParamIterator`
とある変数のみ動かす際，それに従ったパラメータセットクラスのインスタンスを次々と生成してくれる．

### メソッド
* `__init__` : `param`に基本となるパラメータセットのインスタンス（変化させないパラメータはここから参照される），`p`に動かすパラメータ名，`array`にパラメータ`p`がとる値のリスト（配列）を入れる．あとは普通のイテレータと同様に使う．

<br>

## `SimuFileHandler`クラス
データ格納や読み出しを行うクラスである．

### フィールド
* folderpath : データを格納するフォルダのパス
* tmp_param : 基本となるパラメータセット

### メソッド
引数が多くあるものが多いので，一つずつ説明する．

#### `__init__(self, foldername, tmp_param=SParameter())`
初期化を行う．
* foldername : データを格納するフォルダのパス
* tmp_param : 基本となるパラメータセット

#### `__str__(self)`
データ格納フォルダのパスとパラメータセットの変数名リストを文字列化して返す

#### `get_filepath(self, param)`
パラメータを受け取ってそれが示すファイルのパスを返す．

#### `summary(self)`
データ格納フォルダを見てどの変数で調査が進んでいるかなどを表示する．まだ開発中．

#### `add_one_result(self, param, one_result, compress=False, rewrite=False)`
パラメータセットに対して1回分のシミュレーション結果を格納する．
* `param` : パラメータ
* `one_result` : リスト形式の結果．結果として記録したいものが1つでもリスト化すること．
* `compress` : 未実装部分．ファイルを圧縮するかどうか．
* `rewrite` : ファイルを描き直しするかどうか．

#### `add_results(self, param, results, compress=False, rewrite=False)`
複数回のシミュレーション結果を一気に格納する．
* `param` : パラメータ
* `results` : 2次元リスト（配列）で渡される結果．
* `compress` と rewiteは上と同じ

#### `get_num_data(self, param)`
`param`のファイルに何行データがあるか調べて返す（ファイルが存在しなかったら0）

#### `get_num_data_matrix(self, temp_param, xlabel, ylabel, xarray, yarray)`
`get_num_data`の処理を2つのパラメータを同時に変えた領域内で行い，配列で返す．
* `temp_param` : 基本のパラメータ
* `xlabel` : 1つめの変数名
* `ylabel` : 2つめの変数名
* `xarray` : `xlabel`のとる値についてのリスト（配列）
* `yarray` : `ylabel`のとる値についてのリスト（配列）

#### `read_and_get_ave(self, param, mx=-1, show=False)`
`param`のファイルを読み込んで n 行あれば n 試行平均の値を返す．
* `param` : パラメータ
* `mx` : 最大の平均回数．負数にしていれば最大行数の平均が返される．
* `show` : 何回試行平均かを表示する．

#### `read_and_get_ave_matrix(self, temp_param, xlabel, ylabel, xarray, yarray, mx=-1, show=True)`
2つのパラメータを同時に変えた領域内でデータを読み込み，平均化した値を配列で返す．
* `temp_param` : 基本のパラメータ
* `xlabel` : 1つめの変数名
* `ylabel` : 2つめの変数名
* `xarray` : `xlabel`のとる値についてのリスト（配列）
* `yarray` : `ylabel`のとる値についてのリスト（配列）
* `mx` : 最大の平均回数．負数にしていれば最大行数の平均が返される．
* `show` : 何回試行平均かを表示する．

#### `get_ave_1D(self, temp_param, xlabel, xarray, mx, show=True)`
1つのパラメータを変えたときのデータを読み込み，平均化した値を返す．
* `temp_param` : 基本のパラメータ
* `xlabel` : 動かす変数名
* `xarray` : `xlabel`のとる値についてのリスト（配列）
* `mx` : 最大の平均回数．負数にしていれば最大行数の平均が返される．
* `show` : 何回試行平均かを表示する．

#### `get_ave_2D(self, temp_param, xlabel, ylabel, xarray, yarray, mx=-1, show=True)`
実質`read_and_get_ave_matrix()`と同じ．上のメソッドと名前を合わせただけ．
* `temp_param` : 基本のパラメータ
* `xlabel` : 1つめの変数名
* `ylabel` : 2つめの変数名
* `xarray` : `xlabel`のとる値についてのリスト（配列）
* `yarray` : `ylabel`のとる値についてのリスト（配列）
* `mx` : 最大の平均回数．負数にしていれば最大行数の平均が返される．
* `show` : 何回試行平均かを表示する．

#### `delete_rows(self, param, start, stop)`
特定の行だけ消す（シミュレーションが失敗していたとき用）
* `param` : 消したいパラメータ
* `start` : 開始行数（消すのに含まれる）
* `stop` : 終了行数（消すのに含まれない＝消えない）

#### `rename_file(self, sparam, dparam)`
名前を変更する（パラメータセットクラスに急遽新しい変数を追加しなくてはならない際などに使う）
* `sparam` : 変更前のパラメータ
* `dparam` : 変更後のパラメータ
