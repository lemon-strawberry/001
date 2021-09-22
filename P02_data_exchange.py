# 指標の計算と必要なものは対数表示に変えてスケーリングまで行う
import pandas as pd
import numpy as np

import P10_util as util


class DataHenkan:

    # コンストラクタ
    def __init__(self, data_file, out_file, outlier_th,
                 sma_short_term, sma_long_term, ema_short_term, ema_long_term,
                 macd_term, bol_term, sto_k_term, sto_s_term, sto_d_term,
                 x_term
                 ):
        self.data_file = data_file
        self.out_file = out_file
        self.outlier_th = outlier_th
        self.sma_short_term = sma_short_term
        self.sma_long_term = sma_long_term
        self.ema_short_term = ema_short_term
        self.ema_long_term = ema_long_term
        self.macd_term = macd_term
        self.bol_term = bol_term
        self.sto_k_term = sto_k_term
        self.sto_s_term = sto_s_term
        self.sto_d_term = sto_d_term
        self.x_term = x_term

    # データの正規化または標準化とスケーリングと外れ値補正
    def scale(self, data, skip_h=True, standardization=True):
        sds = data.std()
        ave = data.mean()
        ths = sds * self.outlier_th
        # 外れ値補正
        if skip_h:
            data = data.where(data < ave + ths, ave + ths)
            data = data.where(data > ave - ths, ave - ths)
        # 標準化
        if standardization:
            data = (data - ave) / sds

        # 正規化
        else:
            maxs = data.max()
            mins = data.min()
            data = (data - mins) / (maxs - mins)
        return data

    # HLCをopenに対して対数で表現
    def log_henkan(self, data):
        # 新たなデータフレームの作成
        new_data = pd.DataFrame()
        new_data['a_close'] = data['close']
        new_data['a_high'] = data['high']
        new_data['a_low'] = data['low']
        new_data['close'] = np.log(data['close'] / data['open'])
        new_data['high'] = np.log(data['high'] / data['open'])
        new_data['low'] = np.log(data['low'] / data['open'])
        return new_data

    # SMAの計算を行うメソッド
    def sma_mesod(self, data):
        data['short_SMA'] = data['close'].rolling(self.sma_short_term).mean()
        data['long_SMA'] = data['close'].rolling(self.sma_long_term).mean()
        return data

    # EMAの計算を行うメソッド
    def ema_mesod(self, data, new_data):
        ema_name = 'short_EMA'

        ema = np.zeros(len(data))
        ema[:] = np.nan
        ema[self.ema_short_term - 1] = data['close'][:self.ema_short_term].mean()
        for n in range(self.ema_short_term, len(data)):
            ema[n] = ema[n - 1] + (data['close'][n] - ema[n - 1]) / (self.ema_short_term + 1) * 2
        data[ema_name] = ema

        ema_name = 'long_EMA'
        ema = np.zeros(len(data))
        ema[:] = np.nan
        ema[self.ema_long_term - 1] = data['close'][:self.ema_long_term].mean()
        for n in range(self.ema_long_term, len(data)):
            ema[n] = ema[n - 1] + (data['close'][n] - ema[n - 1]) / (self.ema_long_term + 1) * 2
        data[ema_name] = ema

        new_data['short_EMA'] = np.log(data['short_EMA'] / data['close'])
        new_data['long_EMA'] = np.log(data['long_EMA'] / data['close'])

        return data, new_data

    # MACDを計算するメソッド
    def macd_mesod(self, data, new_data):
        new_data['MACD'] = np.log(data['short_EMA'] / data['long_EMA'])
        new_data['MACD_signal'] = new_data['MACD'].rolling(self.macd_term).mean()
        return new_data

    # ボリンジャーバンドを計算するメソッド
    def bollingerband(self, data, new_data):
        data['BollingerBand+1'] = data['long_SMA'] + data['close'].rolling(self.bol_term).std()
        data['BollingerBand+2'] = data['long_SMA'] + data['close'].rolling(self.bol_term).std() * 2
        data['BollingerBand+3'] = data['long_SMA'] + data['close'].rolling(self.bol_term).std() * 3
        data['BollingerBand-1'] = data['long_SMA'] + data['close'].rolling(self.bol_term).std()
        data['BollingerBand-2'] = data['long_SMA'] - data['close'].rolling(self.bol_term).std() * 2
        data['BollingerBand-3'] = data['long_SMA'] - data['close'].rolling(self.bol_term).std() * 3
        new_data['BollingerBand+1'] = np.log(data['BollingerBand+1'] / data['close'])
        new_data['BollingerBand-1'] = np.log(data['BollingerBand-1'] / data['close'])
        # New_data['BollingerBand+2'] = np.log(data['BollingerBand+2'] / data['close'])
        # New_data['BollingerBand-2'] = np.log(data['BollingerBand-2'] / data['close'])
        # New_data['BollingerBand+3'] = np.log(data['BollingerBand+3'] / data['close'])
        # New_data['BollingerBand-3'] = np.log(data['BollingerBand-3'] / data['close'])
        return data, new_data

    # スローストキャスティクスを計算するメソッド
    def stochs(self, data):
        k_stoch = pd.DataFrame((data['close'] - data['low'].rolling(self.sto_k_term).min()) /
                               (data['high'].rolling(self.sto_k_term).max() -
                                data['low'].rolling(self.sto_k_term).min()))
        data['StochS_K'] = k_stoch.rolling(self.sto_s_term).mean()
        data['StochS_D'] = data['StochS_K'].rolling(self.sto_d_term).mean()

        return data['StochS_K'], data['StochS_D']

    # プログラム全体を制御するメソッド
    def execute(self):
        # Dataframe形式でデータを読み込む
        data = util.pd_load_data(self.data_file)
        data = data.dropna()
        print("Conplete load data.")

        # OHLCを対数表記
        new_data = self.log_henkan(data)

        # 指標の計算を行う
        data = self.sma_mesod(data)  # sma
        data, new_data = self.ema_mesod(data, new_data)     # ema
        new_data = self.macd_mesod(data, new_data)      # macd
        data, new_data = self.bollingerband(data, new_data)  # ボリンジャーバンド
        new_data['StochS_K'], new_data['StochS_D'] = self.stochs(data)     # ストキャスティクス
        print("Conplete calculation.")

        # ラベルの作成
        new_data['x_max'] = (data['high'].shift(-self.x_term).rolling(self.x_term).max() / data['close'] - 1) * 100
        new_data['x_min'] = (-data['low'].shift(-self.x_term).rolling(self.x_term).min() / data['close'] + 1) * 100
        print("Conplete maked label.")

        # 桁数の変更
        new_data[new_data.columns[:-1]] = new_data[new_data.columns[:-1]].round(5)
        # Noneデータを削除
        new_data = new_data.dropna()

        new_data.to_csv(self.out_file)
        print("Conplete saved data.")
