import numpy as np

import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt


# Dataframe形式でcsvデータを読み込む関数
def pd_load_data(data_file, header=0, index_col=0):
    data = pd.read_csv(data_file, header=header, index_col=index_col)
    return data


# 訓練用・推定用データの読み込み
def load_data(data_file, data_len, haba):
    data = pd.read_csv(data_file, header=0, index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    data.loc[data["x_max"] >= haba, "max_class"] = 1
    data.loc[~(data["x_max"] >= haba), "max_class"] = 0
    data.loc[data["x_min"] >= haba, "min_class"] = 1
    data.loc[~(data["x_min"] >= haba), "min_class"] = 0
    data = np.array(data)
    tra_data = data[:, 3:-4].astype(np.float)
    tra_class_a = data[:, -2].astype(np.int)
    tra_class_b = data[:, -1].astype(np.int)
    data_in = tra_data.shape[1]
    data_out = 1
    len_seq = tra_data.shape[0] - data_len + 1
    trai_data = []
    trai_class_a = []
    trai_class_b = []
    for i in range(0, len_seq):
        i_data = tra_data[i:i+data_len, :].copy()
        for n in range(data_in):
            ave = i_data[:, n].mean()
            std = i_data[:, n].std()
            i_data[:, n] = (i_data[:, n] - ave) / std
        trai_data.append(i_data)
        trai_class_a.append(tra_class_a[i+data_len-1])
        trai_class_b.append(tra_class_b[i+data_len-1])
    train_data = np.array(trai_data).reshape(len(trai_data), data_len, data_in)
    train_class_a = np.array(trai_class_a).reshape(len(trai_data), data_out)
    train_class_b = np.array(trai_class_b).reshape(len(trai_data), data_out)

    return train_data, train_class_a, train_class_b


# 新規ディレクトリの作成
def mkdir(d, rm=False):
    if rm:
        # 既存の同名ディレクトリがあれば削除
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    else:
        # 既存の同名ディレクトリがある場合、何もしない
        try:
            os.makedirs(d)
        except FileExistsError:
            pass


# データを可視化して保存
def plot(history, filename):

    def add_subplot(nrows, ncols, index, xdata, train_ydata, valid_ydata, ylim, ylabel):
        plt.subplot(nrows, ncols, index)
        plt.plot(xdata, train_ydata, label='training', linestyle='--')
        plt.plot(xdata, valid_ydata, label='validation')
        plt.xlim(1, len(xdata))
        plt.ylim(*ylim)
        plt.xlabel('epoch')
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    # 描画領域のサイズを指定
    plt.figure(figsize=(10, 10))
    # x軸のデータを取得
    x_data = range(1, 1 + len(history['loss']))
    # 検証用データにおける損失を可視化
    add_subplot(2, 1, 1, x_data, history['loss'], history['val_loss'], (0, 5), 'loss')
    # 検証用データにおける正解率を可視化
    add_subplot(2, 1, 2, x_data, history['accuracy'], history['val_accuracy'], (0, 1), 'accuracy')
    # 可視化結果をファイルとして保存
    plt.savefig(filename)
    plt.close('all')


# データを可視化して保存
def plot2(data, pl, b_e_sig, s_e_sig, buy_c_sig, sell_c_sig, filename):
    plt.figure(figsize=(12.8, 8.6))
    # 価格を可視化
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['a_close'], 'red', linewidth=0.2)
    plt.ylabel('Price(JPY)')
    ymin = min(data[['a_high', 'a_low']].min()) - 200
    ymax = max(data[['a_high', 'a_low']].max()) - 200
    plt.vlines(b_e_sig, ymin, ymax, 'blue', linestyles='dashed', linewidth=0.1)
    plt.vlines(s_e_sig, ymin, ymax, 'red', linestyles='dashed', linewidth=0.1)
    plt.vlines(buy_c_sig, ymin, ymax, 'green', linestyles='dashed', linewidth=0.1)
    plt.vlines(sell_c_sig, ymin, ymax, 'black', linestyles='dashed', linewidth=0.1)
    # 損益を可視化
    plt.subplot(2, 1, 2)
    plt.plot(data.index, pl, linewidth=0.2)
    plt.hlines(y=0, xmin=data.index[0], xmax=data.index[-1], colors='k', linestyles='dashed', linewidth=0.1)
    plt.ylabel('Plofit(JPY)')

    # 可視化結果をファイルとして保存
    plt.savefig(filename)
    plt.close('all')


# 損益データの可視化
def plot_profit(x_data, y_data, plt_file):
    y_data = y_data.cumsum()
    plt.figure(figsize=(12.8, 8.6))
    plt.plot(x_data, y_data, linewidth=0.5)
    plt.hlines(y=0, xmin=x_data[0], xmax=x_data[-1], colors='k', linestyles='dashed', linewidth=0.1)
    plt.ylabel('Profit(JPY)')
    plt.savefig(plt_file)
    plt.close('all')
