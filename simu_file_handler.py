import csv
import numpy as np
import os
import os.path
import copy
import pathlib
import re
import glob
import matplotlib.pyplot as plt


# Parameter クラスに継承させるスーパークラス
# __init__()は少なくともオーバーライドする
class SParameter:
    
    # 必要なパラメータに合わせてオーバーライドする
    def __init__(self, sd=None):
        self.pdict = {}
        self.sd = sd
        
        self.set_seed()
    
    
    def set_seed(self):
        if self.sd != None:
            np.random.seed(seed=self.sd)
    
    
    # ファイル名の書式を変えたければオーバーライドする
    def __str__(self):
        s = ""
        for k,v in self.pdict.items():
            if isinstance(v, int):
                s += f"{k:s}{v:d}_"
            elif isinstance(v, float):
                s += f"{k:s}{int(round(v*100)):d}_"
                #s += f"{k:s}{int(v*100):d}_"
            else:
                s += f"{v:s}_"
        return s[:-1]
    
    
    # ファイル名を取得
    def get_filename(self, suf='.csv'):
        return self.__str__() + suf
    
    
    # オブジェクトのコピー
    def copy(self):
        #cp = SParameter()
        cp = copy.deepcopy(self)
        #cp.pdict = self.pdict.copy()
        return cp


    # オブジェクトの内容を変更する
    def update(self, key, val):
        try:
            v = self.pdict[key]
            if isinstance(v, int):
                v = int(val)
            elif isinstance(v, float):
                v = float(val)
            else:
                v = val
            self.pdict[key] = v
        except KeyError as e:
            print(f"'{key}' does not exist.", e)


    # パラメータの辞書を使ってアップデートする
    def update_from_dict(self, dic):
        for k,v in dic.items():
            self.update(k,v)


    # コマンドライン引数からパラメータをアップデートする
    def update_from_argv(self, args):
        for s in args:
            sp = s.split('=')
            key = sp[0]
            val = sp[1]
            self.update(key, val)





# あるパラメータだけ動かす際に用いる Parameter のイテレータ
class ParamIterator(object):
    def __init__(self, param, p, array):
        self._param = param
        self._array = np.array(array)
        self._i = 0
        self._p = p
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i == self._array.shape[0]:
            raise StopIteration
        _tparam = self._param.copy()
        _tparam.pdict[self._p] = self._array[self._i]
        self._i += 1
        return _tparam







class SimuFileHandler():
    
    def __init__(self, foldername, tmp_param=SParameter()):
        self.folderpath = pathlib.Path(foldername).resolve()
        self.tmp_param = tmp_param
        #self.foldername = foldername


    def __str__(self):
        c = pathlib.Path.cwd()
        p = self.folderpath.relative_to(c)
        s = f'Folder: {p}' + '\n'
        s += f'Parameter: {str(list(self.tmp_param.pdict.keys()))}'
        return s
        
    
    def get_filepath(self, param):
        fname = param.get_filename()
        return self.folderpath / fname
    
    
    
    # 特定の1つのパラメータについてデータになっている値の集合を得る（その後，その種類数取得などに使う）
    def _get_one_value_set(self, pkey, param, suf='csv'):

        # フォルダ内のファイル名を全て取得
        file_list = glob.glob(str(self.folderpath) + '/*.csv')
        file_list = [os.path.split(f)[1] for f in file_list]

        # パラメータキーの直後にある数値（あるいは文字列）を抽出
        value_list = [re.findall(f'{pkey}.*?[_.]', fn)[0] for fn in file_list]
        value_list = [s[len(pkey):-1] for s in value_list]
        # print(value_list)   ## for debug

        # 数値の種類数を取得して返す
        value_set = set(value_list)
        return value_set


    def _get_multi_value_set(self, pkeys, param, suf='csv'):

        # フォルダ内のファイル名を全て取得
        file_list = glob.glob(str(self.folderpath) + '/*.csv')
        file_list = [os.path.split(f)[1] for f in file_list]

        value_list = []
        for fn in file_list:
            s = ''
            for pk in pkeys:
                tmp = re.findall(f'{pk}.*?[_.]', fn)[0]
                s += tmp[len(pk):-1] + '_'
            value_list.append(s)

        # 数値の種類数を取得して返す
        value_set = set(value_list)
        return value_set
    
    
    
    
    # 現在フォルダー内に保管されているデータについて概要を表示する
    def summary(self):
        print(self)

        print()
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        param = self.tmp_param
        pd = param.pdict
        n = len(pd.keys())
        keys = list(pd.keys())


        # パラメータがとる値の種類数をそれぞれ取得して辞書にする
        div_dict = {k:len(self._get_one_value_set(k, param)) for k in pd.keys()}

        # パラメータがとる値の種類数を棒グラフでプロット
        print('# of values that was set to each variable. ')
        x = list(div_dict.keys())
        y = list(div_dict.values())

        ax.set_ylim(1,np.max(y)*1.01)
        ax.bar(x, y, align="center")
        ax.set_xticklabels(keys, fontsize=14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        plt.show()



        

        # 2つのパラメータの組み合わせの種類数をヒートマップにする
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)

        data = np.zeros((n, n))

        for y in range(n):
            for x in range(n):
                if y == x:
                    continue
                kx = keys[x]
                ky = keys[y]

                data[y][x] = len(self._get_multi_value_set((kx, ky), param))


        ax.imshow(data, vmin=1)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_yticks(np.arange(len(keys)))

        ax.set_xticklabels(keys, fontsize=14)
        ax.set_yticklabels(keys, fontsize=14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(keys)):
            for j in range(len(keys)):
                text = ax.text(j, i, data[i, j],
                               ha="center", va="center", color="w")

        fig.tight_layout()
        plt.show()






    # 新しい結果を追加する（必ず1回分とする）
    # 1行目には収められている試行回数のみを記述する
    # そのため，データの追加と言えど一度すべて読み込み，試行回数をインクリメント
    # してから書きこみ直し，末尾に新しい行を追加する処理となる
    def add_one_result(self, param, one_result, compress=False, rewrite=False):
        #filename = param.get_filename(".csv")
        #path = os.getcwd() + "/" + foldername + filename
        #path = os.path.join(os.getcwd(), foldername, filename)
        path = str(self.get_filepath(param))
        
        if rewrite:
            os.remove(path)
        
        r = list(one_result)
        num = 0
        n_ele = len(r)
        old = []
        #return
        
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, lineterminator='\n')
                first = reader.__next__()
                num = int(first[0])
                n_ele = int(first[1])
                if n_ele != len(r):
                    raise ValueError("length of one_result do not match with the file")
    
                for row in reader:
                    old.append(row)
            
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([num+1,n_ele])
            
            if len(old) != 0:
                for row in old:
                    writer.writerow(row)
    
            r.append(1)
            writer.writerow(r)
        
        if compress:
            self.compress_file(path)
        
    
    
    # 新しい結果を複数個追加する（渡すのは2次元配列的なやつ）
    # 1行目には収められている試行回数のみを記述する
    # そのため，データの追加と言えど一度すべて読み込み，試行回数をインクリメント
    # してから書きこみ直し，末尾に新しい行を追加する処理となる
    def add_results(self, param, results, compress=False, rewrite=False):
        #filename = param.get_filename(".csv")
        #path = os.getcwd() + "/" + foldername + filename
        #path = os.path.join(os.getcwd(), foldername, filename)
        path = str(self.get_filepath(param))
        
        if rewrite:
            os.remove(path)
        
        rlist = [list(l) for l in list(results)]
        num = 0
        n_ele = len(rlist[0])
        n_add = len(rlist)
        old = []
        #return
        
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, lineterminator='\n')
                first = reader.__next__()
                num = int(first[0])
                n_ele = int(first[1])
                if n_ele != len(rlist[0]):
                    raise ValueError("length of one_result do not match with the file")
    
                for row in reader:
                    old.append(row)
            
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([num+n_add, n_ele])
            
            if len(old) != 0:
                for row in old:
                    writer.writerow(row)
    
            for r in rlist:
                r.append(1)
                writer.writerow(r)
        
        if compress:
            self.compress_file(path)
    
    
    
    # ファイルの持つデータ数を返す
    def get_num_data(self, param): #, foldername='./'):
        #filename = param.get_filename(".csv")
        #path = os.path.join(os.getcwd(), foldername, filename)
        path = str(self.get_filepath(param))
    
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, lineterminator='\n')
                first = reader.__next__()
                return int(first[0]), int(first[1])
        else:
            #print(f"{filename} does not exist.")
            return 0, 0
    
    
    
    # 2パラメータを変化させたときのデータについて、データ数の行列を返す
    def get_num_data_matrix(self, temp_param, xlabel, ylabel, xarray, yarray): #, foldername='./'):

        xarray = np.array(xarray); yarray = np.array(yarray)
        ret = np.zeros((2, yarray.shape[0], xarray.shape[0]), dtype=int)
    
        i = 0
        for yparam in ParamIterator(temp_param, ylabel, yarray):
            j = 0
            for xparam in ParamIterator(yparam, xlabel, xarray):
                t = self.get_num_data(xparam) #, foldername)
                ret[0][i][j] = t[0]; ret[1][i][j] = t[1]
                j += 1
            i += 1
    
        return ret
    
    
    
    # 指定個数以下の施行を平均したものを読み込んで返す    
    def read_and_get_ave(self, param, mx=-1, show=False): #, foldername='./'):
        #filename = param.get_filename(".csv")
        #path = foldername + filename
        #filename = param.get_filename(".csv")
        #path = os.getcwd() + "/" + foldername + filename
        #path = os.path.join(os.getcwd(), foldername, filename)
        path = str(self.get_filepath(param))
        
        num = 0
        n_ele = 0
        
        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            first = reader.__next__()
            num = int(first[0])
            n_ele = int(first[1])
            if num <= mx or mx < 0:
                mx = num
            
            data = np.zeros(n_ele)
            count = 0
            
            for row in reader:
                tmp = list(map(float, row))
                if count + tmp[-1] > mx:
                    break
                tmpa = np.array(tmp[:-1]) * tmp[-1]
                data = data + tmpa
                count += tmp[-1]
            
            print('attempts:', int(count))
            
            return data / count
    
    
    
    
    # 指定個数以下の施行を平均したものをパラメータセットごと読み込んでmatrixで返す    
    def read_and_get_ave_matrix(self, temp_param, xlabel, ylabel, 
                                xarray, yarray, mx=-1, show=True):
        
        nums = self.get_num_data_matrix(temp_param, xlabel, ylabel, 
                                        xarray, yarray)#, foldername)
        min_nd = np.amin(nums[0])
        min_ele = np.amin(nums[1])
    
        if min_nd <= 0:
            print("It's short for some data files. ")
            print(nums[0])
            max_ele = max(np.amax(nums[1]), 1)
            return np.zeros((max_ele, yarray.shape[0], xarray.shape[0]))
    
        if mx < 0:
            mx = min_nd
        else:
            mx = min(mx, min_nd)
        
        if show:
            print('attemps:', mx)
    
        ret = np.zeros((min_ele, yarray.shape[0], xarray.shape[0]))
    
        #i = yarray.shape[0]-1
        i = 0
        for yparam in ParamIterator(temp_param, ylabel, yarray):
            j = 0
            for xparam in ParamIterator(yparam, xlabel, xarray):
                data = self.read_and_get_ave(xparam, mx)#, foldername)
                for k in range(min_ele):
                    ret[k][i][j] = data[k]
                j += 1
            #i -= 1
            i += 1
    
        return ret
    
    
    
    # ベクトルで返す
    def get_ave_1D(self, temp_param, xlabel, xarray, mx=-1, show=True):
        dammy_key, dammy_val = list(temp_param.pdict.items())[-1]
        return self.read_and_get_ave_matrix(temp_param, xlabel, dammy_key, 
                                            xarray, np.array([dammy_val]), 
                                            mx, show)[:,0]
    
    
    # 行列で返す
    def get_ave_2D(self, temp_param, xlabel, ylabel, 
                   xarray, yarray, mx=-1, show=True):
        
        return self.read_and_get_ave_matrix(temp_param, xlabel, ylabel, 
                                            xarray, yarray, mx, show)
    
    
    
    # 特定の行だけ消す（シミュレーションを失敗したとき用）
    def delete_rows(self, param, start, stop, foldername='./'):
        #filename = param.get_filename(".csv")
        #path = os.path.join(os.getcwd(), foldername, filename)
        path = str(self.get_filepath(param))
        
        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            num = int(reader.__next__()[0])
            old = []
            
            for row in reader:
                old.append(row)
        
        if start >= stop or start >= num or stop < 0:
            raise ValueError("value of start or stop is not correct")
            
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([num-(stop-start)])
            i = 0
            
            for row in old:
                if not(i >= start and i < stop):
                    writer.writerow(row)
    
    
    
    # 名前を変更する(間違えた時用)
    def rename_file(self, sparam, dparam): #, foldername='./'):
        #sfilename = sparam.get_filename(".csv")
        #spath = os.path.join(os.getcwd(), foldername, sfilename)
        spath = str(self.get_filepath(sparam))
        #dfilename = dparam.get_filename(".csv")
        #dpath = os.path.join(os.getcwd(), foldername, dfilename)
        dpath = str(self.get_filepath(dparam))
    
        os.rename(spath, dpath)
    
    
    
    # 結果のファイルを圧縮する
    # 例えば，試行回数が10< となったら，10行分足し合わせて
    # 10試行のデータとする．行の先頭には10と書いておく
    def compress_file(path):
        pass
