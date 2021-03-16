import csv
import numpy as np
import os
import os.path
import copy
import pathlib
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import sys


# Parameter クラスに継承させるスーパークラス
# __init__()は少なくともオーバーライドする
class SParameter:
    
    # 必要なパラメータに合わせてオーバーライドする
    def __init__(self, sd=None):
        self.pdict = {}
        self.types = {}
        self.labels = {}
        self.sd = sd
        
        self.set_seed()


    # パラメータをリセットする．updateのたびに呼ばれる
    def reset(self):
        self.reset_types()
        self.reset_labels()
    

    # 型をリセットする
    def reset_types(self):
        #self.types = {k:type(v) for k,v in self.pdict.items()}
        for k,v in self.pdict.items():
            if k not in self.types:
                self.types[k] = type(v)
    

    # 図用のラベルをリセットする
    def reset_labels(self):
        #self.labels = {k:k for k,v in self.pdict.items()}
        for k,v in self.pdict.items():
            if k not in self.labels:
                self.labels[k] = k

    
    # シードを設定
    def set_seed(self):
        if self.sd != None:
            np.random.seed(seed=self.sd)


    # ファイル名の書式を変えたければオーバーライドする
    def __str__(self):
        s = ""
        for k,v in self.pdict.items():
            if len(self.types) > 0:
                if self.types[k] is int:
                    s += f"{k:s}={v:d}_"
                elif self.types[k] is float:
                    s += f"{k:s}={v:E}_"
                else:
                    s += f"{k:s}={str(v):s}_"
                    
            else:
                if isinstance(v, int):
                    s += f"{k:s}={v:d}_"
                elif isinstance(v, float):
                    s += f"{k:s}={v:E}_"
                    #s += f"{k:s}={int(round(v*100)):d}_"
                    #s += t.replace('.', '')
                else:
                    s += f"{k:s}={str(v):s}_"
        return s[:-1]


    # ファイル名を取得
    def get_filename(self, suf='.csv'):
        if suf == '':
            print("You must set suffix. '.csv' is set automatically.")
            suf = '.csv'

        return self.__str__() + suf
    
    
    # ファイル名からパラメータに変換
    def conv_fname_to_param(self, fname):
        cp = self.copy()
        new_dict = {}

        # .~ の形式の拡張子はファイル名に必ずついているとする
        fname = fname.rsplit('.', 1)[0]
        fname = fname + '_'

        for key, val in self.pdict.items():
            l = re.findall(f'{key}=.*?[_]', fname)
            if len(l) == 0:
                raise Exception('The file name is strange or does not have suffix.' +\
                                ' : '+fname)
            
            tmp = l[0]
            #tmp = re.findall(f'{key}=.*?[_]', fname)[0]
            #print(tmp)
            value_str = tmp[len(key)+1:-1]

            if len(self.types) > 0:
                if self.types[key] is int:
                    v = int(value_str)
                elif self.types[key] is float:
                    v = float(value_str)
                else:
                    v = value_str
                    
            else:
                if isinstance(val, int):
                    v = int(value_str)
                elif isinstance(val, float):
                    v = float(value_str)
                else:
                    v = value_str

            new_dict[key] = v
        
        cp.update_from_dict(new_dict)
        return cp

    
    
    # オブジェクトのコピー
    def copy(self):
        #cp = SParameter()
        cp = copy.deepcopy(self)
        #cp.pdict = self.pdict.copy()
        cp.reset()
        return cp



    # 値を取得する
    def get(self, key):
        return self.pdict.get(key)



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

            self.reset()
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


    # 指定された値を含むかどうか
    def include(self, pd):
        for k,v in pd.items():
            if self.pdict[k] != v:
                return False

        return True





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
        #_tparam.reset(); print(_tparam)
        #_tparam.pdict[self._p] = self._array[self._i]
        _tparam.update(self._p, self._array[self._i])
        self._i += 1
        return _tparam






class SimuFileHandler():
    
    def __init__(self, foldername, tmp_param=SParameter()):
        self.folderpath = pathlib.Path(foldername).resolve()
        self.tmp_param = tmp_param
        #self.foldername = foldername
        self.exe_file = sys.argv[0]#; print(self.exe_file)

        if not os.path.exists(str(self.folderpath)):
            print(f'Make directory: {self.folderpath}')
            os.mkdir(str(self.folderpath))

            """
            # ログファイルを作成
            log_path = self.folderpath / 'log'
            os.mkdir(str(log_path))

            # 実行ファイル名の記録用
            with open(str(log_path / 'execusion.log'), "w") as f:
                f.write('Initialized ('+str(datetime.datetime.now())+')')

            # シミュレーションしたパラメータと実行番号を紐づけて記録
            with open(str(log_path / 'data.log'), 'w') as f:
                f.write('Initialized ('+str(datetime.datetime.now())+')')
            """

        self.path_list  = None  # ファイルパスのリスト
        self.fname_list = None  # ファイル名のリスト
        self.param_list = None  # ファイルをSParameterのサブクラス型に変換したもののリスト
        
        self._makeup_lists()
            



    def __str__(self):
        c = pathlib.Path.cwd()
        p = self.folderpath.relative_to(c)
        s = f'Folder: {p}' + '\n'
        s += f'Parameter: {str(list(self.tmp_param.pdict.keys()))}'
        return s
        
    
    def get_filepath(self, param, suf=None):
        if suf != None:
            fname = param.get_filename(suf)
        else:
            fname = param.get_filename()
        return self.folderpath / fname



    def _get_all_paths(self, suf='.csv'):
        # フォルダ内のファイル名を全て取得
        file_list = glob.glob(str(self.folderpath) + '/*' + suf)

        return file_list
    
    
    
    def _get_all_fnames(self, suf='.csv'):
        # フォルダ内のファイル名を全て取得
        file_list = glob.glob(str(self.folderpath) + '/*' + suf)
        file_list = [os.path.split(f)[1] for f in file_list]

        return file_list



    def _makeup_lists(self, suf='.csv'):
        print('Reading files...')

        # フォルダ内のファイルパスを全て取得
        self.path_list = self._get_all_paths(suf)

        # フォルダ内のファイル名を全て取得
        self.fname_list = self._get_all_fnames(suf)

        self.param_list = [self.tmp_param.conv_fname_to_param(fn) for fn in self.fname_list]
        print('Completed')

    
    
    
    # 特定の1つのパラメータについてデータになっている値の集合を得る（その後，その種類数取得などに使う）
    def _get_one_value_set(self, pkey, suf='.csv'):  # param

        # param_listから値を抽出
        value_list = [p.pdict[pkey] for p in self.param_list]

        # 数値の種類数を取得して返す
        value_set = set(value_list)
        return value_set



    # 特定の複数のパラメータについて値の集合を得る
    def _get_multi_value_set(self, pkeys, suf='.csv'):  # param

        value_list = []
        for p in self.param_list:
            tmp = []
            for pk in pkeys:
                tmp.append(p.pdict[pk])
            value_list.append(tuple(tmp))

        # 数値の種類数を取得して返す
        value_set = set(value_list)
        return value_set



    def _get_num_of_sets_and_attemps(self, pdict, suf='.csv'):

        # 指定されたパラメータセットを持っている物を抽出
        p_list = [p for p in self.param_list if p.include(pdict)]

        # 試行数を計算
        s = 0
        for p in p_list:
            s += self.get_num_data(p)[0]

        return len(p_list), s

        
    
    
    
    
    # 現在フォルダー内に保管されているデータについて概要を表示する
    def summary(self, update=False):
        print(self)

        if update:
            self._makeup_lists()

        print()
        psnum = len(self.path_list)
        dnums = [self._get_num_data_from_path(p)[0] for p in self.path_list]
        total = sum(dnums)
        print(f'# of files: {psnum:d}')
        print(f'Total attemps: {total:d}')


        print()
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121)
        param = self.tmp_param
        pd = param.pdict
        n = len(pd.keys())
        keys = list(pd.keys())
        labels = [self.tmp_param.labels[k] for k in keys]


        # パラメータがとる値の種類数をそれぞれ取得して辞書にする
        div_dict = {k:len(self._get_one_value_set(k)) for k in pd.keys()}

        # パラメータがとる値の種類数を棒グラフでプロット
        print('# of values that was set to each variable. ')
        x = list(div_dict.keys())
        y = list(div_dict.values())

        ax1.set_ylim(1,np.max(y)*1.01)
        ax1.bar(x, y, align="center")
        ax1.set_xticklabels(labels, fontsize=14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        #plt.show()



        

        # 2つのパラメータの組み合わせの種類数をヒートマップにする
        #fig = plt.figure(figsize=(5,5))
        ax2 = fig.add_subplot(122)

        data = np.zeros((n, n))

        for y in range(n):
            for x in range(n):
                if y == x:
                    continue
                kx = keys[x]
                ky = keys[y]

                data[y][x] = len(self._get_multi_value_set((kx, ky)))


        ax2.imshow(data, vmin=1)
        ax2.set_xticks(np.arange(len(keys)))
        ax2.set_yticks(np.arange(len(keys)))

        ax2.set_xticklabels(labels, fontsize=14)
        ax2.set_yticklabels(labels, fontsize=14)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(keys)):
            for j in range(len(keys)):
                text = ax2.text(j, i, data[i, j],
                               ha="center", va="center", color="w")

        fig.tight_layout()
        plt.show()



        """
        # 上位2つのパラメータについて散布図を表示
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)

        div_tlist = [(k,v) for k,v in div_dict.items()]
        tmp = sorted(div_tlist, key=lambda x:x[1], reverse=True)
        x = tmp[0][0]
        y = tmp[1][0]
        self.scatter_two_param(x, y, ax=ax)

        plt.show()
        """





    # 2つのパラメータについて，シミュレーションが行われた
    # パラメータセットの数と試行数を散布図で表示する
    def scatter_two_param(self, xkey, ykey, xlim=None, ylim=None, update=False):
        fig = plt.figure(figsize=(6,4.8))
        ax = fig.add_subplot(111)

        if update:
            self._makeup_lists()
        value_list = list(self._get_multi_value_set((xkey, ykey)))

        set_list = []
        att_list = []
        for v in value_list:
            pdict = {xkey:v[0], ykey:v[1]}
            sets, attemps = self._get_num_of_sets_and_attemps(pdict)
            set_list.append(sets)
            att_list.append(attemps)

        xlist = [v[0] for v in value_list]
        ylist = [v[1] for v in value_list]

        s_arr = np.array(set_list) * 10.
        c_arr = np.array(att_list)

        ax.scatter(xlist, ylist, s=s_arr, c=c_arr, alpha=0.5)
        ax.set_xlabel(self.tmp_param.labels[xkey], fontsize=12)
        ax.set_ylabel(self.tmp_param.labels[ykey], fontsize=12)

        if xlim != None:
            ax.set_xlim(*xlim)
        if ylim != None:
            ax.set_ylim(*ylim)

        sm = plt.cm.ScalarMappable(norm=plt.Normalize(
                                        vmin=np.min(c_arr), 
                                        vmax=np.max(c_arr)))
        sm._A = []
        cbar = fig.colorbar(sm)
        cbar.set_label('attempts', fontsize=12)

        plt.show()





    # 新しい結果を追加する（必ず1回分とする）
    # 1行目には収められている試行回数のみを記述する
    # そのため，データの追加と言えど一度すべて読み込み，試行回数をインクリメント
    # してから書きこみ直し，末尾に新しい行を追加する処理となる
    def add_one_result(self, param, one_result, compress=False, rewrite=False):
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
    


    # ファイルパスからファイルを読み込んでデータ数返す
    def _get_num_data_from_path(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, lineterminator='\n')
                first = reader.__next__()
                return int(first[0]), int(first[1])
        else:
            #print(f"{filename} does not exist.")
            return 0, 0

    
    
    # ファイルの持つデータ数を返す
    def get_num_data(self, param): #, foldername='./'):
        path = str(self.get_filepath(param))
    
        return self._get_num_data_from_path(path)
    
    
    
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



    # 1パラメータセットのデータを全てリストの形で得る
    def read_and_get_one_file(self, param, sort=False):
        path = str(self.get_filepath(param))
        
        num, n_ele = self.get_num_data(param)
        if num == 0 and n_ele == 0:
            return np.zeros(1)

        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            first = reader.__next__()

            data = np.zeros((num, n_ele))
            count = 0
            
            for i, row in enumerate(reader):
                tmp = list(map(float, row))
                data[i,:] = np.array(tmp[:-1])

            if sort:
                for i in range(n_ele):
                    data = np.sort(data, axis=0)

            return data
        



    """
    # 1パラメータセットのデータを可視化する
    def show_one_file(self, param, sort=False):
        data = read_and_get_one_file(param, sort)
    #"""





    # データを1つのみ指定して（or ランダムに）得る
    def read_and_get_one(self, param, index=0):
        path = str(self.get_filepath(param))
        
        num, n_ele = self.get_num_data(param)
        if num == 0 and n_ele == 0:
            return np.zeros(1)

        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            first = reader.__next__()
            #num = int(first[0])
            #n_ele = int(first[1])

            if num <= index:
                index = num-1
            elif index < 0:
                index = np.randint(num)
                
            
            data = np.zeros(n_ele)
            count = 0
            
            for row in reader:
                tmp = list(map(float, row))
                data = np.array(tmp[:-1]) * tmp[-1]
                if count + tmp[-1] > index:
                    break
                count += tmp[-1]
            
            return data
    
    
    
    # 指定個数以下の施行を平均したものを読み込んで返す    
    def read_and_get_ave(self, param, mx=-1, show=False): #, foldername='./'):
        path = str(self.get_filepath(param))

        num, n_ele = self.get_num_data(param)
        if num == 0 and n_ele == 0:
            raise ValueError('Empty file:', path)
            return np.zeros(1)
        
        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            first = reader.__next__()

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
            
            #"""
            if show:
                print('attempts:', int(count))
            #"""
            
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
            print('attempts:', mx)
    
        ret = np.zeros((min_ele, yarray.shape[0], xarray.shape[0]))
    
        #i = yarray.shape[0]-1
        i = 0
        for yparam in ParamIterator(temp_param, ylabel, yarray):
            j = 0
            for xparam in ParamIterator(yparam, xlabel, xarray):
                data = self.read_and_get_ave(xparam, mx)#, show)#, foldername)
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




    # 指定個数以下の試行のsample standard deviationを返す    
    def read_and_get_ssd(self, param, mx=-1):

        # 平均を計算しておく
        ave = self.read_and_get_ave(param, mx)


        path = str(self.get_filepath(param))

        num, n_ele = self.get_num_data(param)
        if num == 0 and n_ele == 0:
            raise ValueError('Empty file:', path)
            return np.zeros(1)
        
        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')
            first = reader.__next__()

            if num <= mx or mx < 0:
                mx = num
            
            data = np.zeros(n_ele)
            count = 0
            
            for row in reader:
                tmp = list(map(float, row))
                if count + tmp[-1] > mx:
                    break
                tmpa = np.array(tmp[:-1]) * tmp[-1]
                #data = data + tmpa
                data = data + np.power(tmpa-ave, 2)
                count += tmp[-1]
            
            #"""
            if show:
                print('attempts:', int(count))
            #"""
            
            return np.sqrt(data / (count - 1))




    # 指定個数以下の試行の標本標準偏差をパラメータセットごと読み込んでmatrixで返す    
    def read_and_get_ssd_matrix(self, temp_param, xlabel, ylabel, 
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
            print('attempts:', mx)
    
        ret = np.zeros((min_ele, yarray.shape[0], xarray.shape[0]))
    
        #i = yarray.shape[0]-1
        i = 0
        for yparam in ParamIterator(temp_param, ylabel, yarray):
            j = 0
            for xparam in ParamIterator(yparam, xlabel, xarray):
                data = self.read_and_get_ssd(xparam, mx)#, show)#, foldername)
                for k in range(min_ele):
                    ret[k][i][j] = data[k]
                j += 1
            #i -= 1
            i += 1
    
        return ret
    
    
    
    # ベクトルで返す
    def get_ssd_1D(self, temp_param, xlabel, xarray, mx=-1, show=True):
        dammy_key, dammy_val = list(temp_param.pdict.items())[-1]
        return self.read_and_get_ssd_matrix(temp_param, xlabel, dammy_key, 
                                            xarray, np.array([dammy_val]), 
                                            mx, show)[:,0]
    
    
    # 行列で返す
    def get_ssd_2D(self, temp_param, xlabel, ylabel, 
                   xarray, yarray, mx=-1, show=True):
        
        return self.read_and_get_ssd_matrix(temp_param, xlabel, ylabel, 
                                            xarray, yarray, mx, show)




    # データ生成用の関数を使ってアニメーション用のファイルを作る
    def write_anim_data(self, param, lx, ly, MAX_ITER, 
                        iter_func, gen_func, 
                        iter_args=None, gen_args=None, 
                        ini_func=None, ini_args=None,
                        entitle_func=None, entitle_args=None):

        path = str(self.get_filepath(param, suf='.ani'))

        if iter_args != None and iter_args is tuple:
            iter_func_ = lambda :iter_func(*iter_args)
        else:
            iter_func_ = iter_func

        if gen_args != None and gen_args is tuple:
            gen_func_ = lambda :gen_func(*gen_args)
        else:
            gen_func_ = gen_func

        if entitle_func != None:
            if entitle_args != None and entitle_args is tuple:
                entitle_func_ = lambda i:entitle_func(i, *entitle_args)
            else:
                entitle_func_ = entitle_func
        else:
            entitle_func_ = lambda i: ' '


        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([lx,ly,MAX_ITER])   # [データの横幅, 縦幅, ステップ数]

            if ini_func != None:
                if ini_args != None:
                    ini_func_ = lambda :ini_func(*ini_args)
                else:
                    ini_func_ = ini_func

                ini_func_()

            data = gen_func_()
            writer.writerow([entitle_func_(0)])
            writer.writerows(data)
            

            for i in range(MAX_ITER):
                iter_func_()
                data = gen_func_()

                writer.writerow([entitle_func_(i+1)])
                writer.writerows(data)




    # animation用のimsを作成する（描画用関数を用いて）
    def read_anim_data(self, fig, param, draw_func, draw_args=None, dtype=float, interval=200, frames=None):

        path = str(self.get_filepath(param, suf='.ani'))

        if draw_args != None:
            draw_func_ = lambda data, i, text: draw_func(data, i, text, *draw_args)
        else:
            draw_func_ = draw_func
        
        #fig = plt.figure()
        ims = []

        with open(path, "r") as f:
            reader = csv.reader(f, lineterminator='\n')

            first = reader.__next__()
            lx = int(first[0])
            ly = int(first[1])
            MAX_ITER = int(first[2])

            if frames != None and MAX_ITER > frames:
                MAX_ITER = frames
            
            for i in range(MAX_ITER+1):

                data = np.zeros((ly,lx))
                title_text = reader.__next__()

                for j in range(ly):
                    row = reader.__next__()
                    tmp = np.array(list(map(dtype, row)))
                    
                    data[j,:] = tmp

                ims.append(draw_func_(data, i, title_text[0]))

                show_progress(i, MAX_ITER, 20)

            ani = animation.ArtistAnimation(fig, ims, interval)
            return ani

    


    
    
    # 特定の行だけ消す（シミュレーションを失敗したとき用）
    def delete_rows(self, param, start, end):

        path = str(self.get_filepath(param))
        num = 0
        n_ele = 0
        old = []

        if os.path.exists(path):
            with open(path, "r") as f:
                reader = csv.reader(f, lineterminator='\n')
                first = reader.__next__()
                num = int(first[0])
                n_ele = int(first[1])
                
                for row in reader:
                    old.append(row)
        else:
            raise ValueError(f"{path} does not exist.")
        

        if start >= end or start >= num or end < 0:
            raise ValueError("value of start or stop is not correct")

        if end > num:
            end = num
        if start < 0:
            start = 0

        del old[start:end]
            

        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([num-(end-start), n_ele])

            for row in old:
                writer.writerow(row)
            
            """
            for row in old:
                if not(i >= start and i < end):
                    writer.writerow(row)
            """




    # 古い(新しい)データをnum個残してあとは消す
    def trim_rows(self, param, num, basis='older'):
        n = self.get_num_data(param)[0]
        if basis == 'older':
            st = num
            ed = n
        elif basis == 'newer':
            st = 0
            ed = n - num

        self.delete_rows(param, st, ed)
        


    
    
    
    # 名前を変更する(間違えた時用)
    def rename_file(self, sparam, dparam): #, foldername='./'):
        spath = str(self.get_filepath(sparam))
        dpath = str(self.get_filepath(dparam))
        if os.path.exists(dpath):
            print('Destination file already exists.')
            return
    
        os.rename(spath, dpath)
    
    
    
    # 結果のファイルを圧縮する
    # 例えば，試行回数が10< となったら，10行分足し合わせて
    # 10試行のデータとする．行の末尾には10と書いておく
    # 現時点では必要なさそうなので後回し（21/03/16）
    def compress_file(path):
        pass






def show_progress(i, MAX_ITER, length, fs='', bs=''):
    if i%(MAX_ITER/length) == 0:
        print(fs, "*"*int(i/(MAX_ITER/length)) + "_"*(length-int(i/(MAX_ITER/length))), \
                f"{int(i/MAX_ITER*100):3d}%", bs, end='\r')

