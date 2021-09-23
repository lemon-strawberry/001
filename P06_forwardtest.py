import sys
import pandas as pd
import datetime
import time
import numpy as np
from tensorflow.keras.models import load_model

import P10_util as util
import P12_API_util as api
import P01_get_data as get_data
import P02_data_exchange as calculation
import P13_line_util as lutil

timedelta = datetime.timedelta(hours=9)


class RunningBot:
    def __init__(self, periods, est_fileA, est_fileB, lstm_len, width, x_width, x_term, lot,
                 plt_dir, plt_file,
                 OUTLIER_TH, SMA_SHORT_TERM, SMA_LONG_TERM, EMA_SHORT_TERM, EMA_LONG_TERM,
                 MACD_TERM, BOL_TERM, STO_K_TERM, STO_S_TERM, STO_D_TERM, X_TERM):
        self.est_fileA = est_fileA
        self.est_fileB = est_fileB
        self.periods = periods
        self.lstm_len = lstm_len
        self.width = width
        self.x_width = x_width
        self.x_term = x_term
        self.lot = lot
        self.plt_dir = plt_dir
        self.plt_file = plt_file

        self.datahenkan = calculation.DataHenkan(
            data_file=None,
            out_file=None,
            outlier_th=OUTLIER_TH,
            sma_short_term=SMA_SHORT_TERM,
            sma_long_term=SMA_LONG_TERM,
            ema_short_term=EMA_SHORT_TERM,
            ema_long_term=EMA_LONG_TERM,
            macd_term=MACD_TERM,
            bol_term=BOL_TERM,
            sto_k_term=STO_K_TERM,
            sto_s_term=STO_S_TERM,
            sto_d_term=STO_D_TERM,
            x_term=X_TERM
        )

    # 現在の時間をdatetime形式->now_timeと(str"%H:%M:%S")->ntと現在の秒数->now_secondを取得
    def now_date(self):
        now_time = datetime.datetime.now()
        now_second = now_time.second
        now_minutes = now_time.minute
        now_hour = now_time.hour
        nt = now_time.strftime('%H:%M:%S')
        now_time = now_time.strftime('%Y-%m-%d %H:%M')
        now_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M')

        return nt, now_time, now_second, now_minutes, now_hour

    def data_calculation(self, data):
        New_data = self.datahenkan.log_henkan(data)
        data = self.datahenkan.sma_mesod(data)
        data, New_data = self.datahenkan.ema_mesod(data, New_data)
        New_data = self.datahenkan.macd_mesod(data, New_data)
        data, New_data = self.datahenkan.bollingerband(data, New_data)
        New_data['StochS_K'], New_data['StochS_D'], New_data['StochS_'] = self.datahenkan.stochs(data)

        New_data[New_data.columns[:-1]] = New_data[New_data.columns[:-1]].round(5)
        New_data = New_data.dropna()

        return New_data

    # フォワードテスト
    def test(self, data, predictA, predictB, now_time,
             buy_entry, sell_entry, buy_close, sell_close, passage_time, profit, pos=[0]):
        # クローズをさきに行い後でエントリーを行うことで
        # ドテンのときにも対応

        # ロングクローズ
        if pos[0] > 0:
            if data["a_high"] >= buy_entry[-1]*(1+0.01*self.width):
                now_price = buy_entry[-1] * (1+0.01 * self.width)
                buy_close.append(now_price)
                pos[0] = 0
                pure_profit = buy_close[-1] - buy_entry[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ロングクローズ {}'.format(profit[-1]))

            elif data["a_low"] <= buy_entry[-1]*(1-0.01*self.width):
                now_price = buy_entry[-1] * (1 - 0.01 * self.width)
                buy_close.append(now_price)
                pos[0] = 0
                pure_profit = buy_close[-1] - buy_entry[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ロングクローズ {}'.format(profit[-1]))

            elif pos[0] >= self.x_term:
                now_price = api.get_board("bids")
                buy_close.append(now_price)
                pos[0] = 0
                pure_profit = buy_close[-1] - buy_entry[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ロングクローズ {}'.format(profit[-1]))

            else:
                pos[0] += 1

        # ショートクローズ
        elif pos[0] < 0:
            if data["a_low"] <= sell_entry[-1]*(1-0.01*self.width):
                now_price = sell_entry[-1]*(1-0.01*self.width)
                sell_close.append(now_price)
                pos[0] = 0
                pure_profit = sell_entry[-1] - sell_close[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ショートクローズ {}'.format(profit[-1]))

            elif data["a_high"] >= sell_entry[-1]*(1+0.01*self.width):
                now_price = sell_entry[-1] * (1 + 0.01 * self.width)
                sell_close.append(now_price)
                pos[0] = 0
                pure_profit = sell_entry[-1] - sell_close[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ショートクローズ {}'.format(profit[-1]))

            elif pos[0] <= -self.x_term:
                now_price = api.get_board("asks")
                sell_close.append(now_price)
                pos[0] = 0
                pure_profit = sell_entry[-1] - sell_close[-1]
                profit.append(int(pure_profit * self.lot))
                passage_time.append(now_time)
                print('ショートクローズ {}'.format(profit[-1]))

            else:
                pos[0] -= 1

        # エントリー
        if pos[0] == 0:
            passage_time.append(now_time)
            profit.append(0)
            # ロングエントリー
            if predictA >= self.x_width > predictB:
                now_price = api.get_board("asks")
                buy_entry.append(now_price)
                pos[0] += 1
                print('ロングエントリー')

            elif predictB >= self.x_width > predictA:
                now_price = api.get_board("bids")
                sell_entry.append(now_price)
                pos[0] -= 1
                print('ショートエントリー')

            else:
                print("not_signal")

        print("position count : ", pos[0])
        print(predictA, " : ", predictB)
        print(api.get_board("asks"), " : ", api.get_board("bids"))
        print("-----------------------")

        return buy_entry, sell_entry, buy_close, sell_close, passage_time, profit

    # 損益グラフをプロット(テスト用)
    def plot_profit_test(self, passage_time, profit, now_date):
        util.mkdir(self.plt_dir, rm=False)
        plot_data = pd.DataFrame(profit, columns=['profit'], index=passage_time)
        plot_data.loc[now_date+datetime.timedelta(minutes=1)] = 0
        util.plot_profit(plot_data.index, plot_data['profit'], self.plt_file)
        message = "定期報告 {}".format(now_date)
        lutil.line_notify(message, self.plt_file)
        print('plot')

    def execute(self):
        # 学習済みモデルの読み込み
        estimatorA = load_model(self.est_fileA)
        estimatorB = load_model(self.est_fileB)

        # 証拠金変動の入手をする準備
        start_time = datetime.datetime.now()
        start_time = start_time.strftime('%Y-%m-%d %H:%M')
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M')

        # クラスの読み込み
        t_get = get_data.GetCandleStick(out_file=None, periods=self.periods)

        # メンテナンス時間の設定
        maintenance_start = "03:59:10"
        maintenance_end = "04:13:00"

        # テスト時のパラメータ
        buy_entry = []
        sell_entry = []
        buy_close = []
        sell_close = []
        passage_time = [start_time]
        profit = [0]

        print("Starting roop")
        # ここからループ
        while True:
            try:
                while True:
                    nt, now_time, now_second, now_minutes, now_hour = self.now_date()
                    # メンテナンス時間中ボットを停止
                    if maintenance_start < nt < maintenance_end:
                        time.sleep(1)
                        continue

                    if now_second < 2 and now_minutes % 5 == 0:     # 5分ごとに実行

                        # 新規データの取得
                        t_data = t_get.getcandlestick()
                        t_data.index = t_data["date"]
                        data = self.data_calculation(t_data)

                        num = data.index.get_loc(now_time - datetime.timedelta(minutes=5))

                        # 学習モデルに用いたデータに合わせて書き換える必要あり
                        est_data = data.iloc[num-self.lstm_len+1:num+1, 3:]
                        est_data = np.array(est_data)

                        for n in range(est_data.shape[1]):
                            ave = est_data[:, n].mean()
                            std = est_data[:, n].std()
                            est_data[:, n] = (est_data[:, n] - ave) / std
                        est_data_ = est_data.reshape(1, est_data.shape[0], est_data.shape[1])

                        # 推定の実行
                        predA = estimatorA.predict(est_data_)[0][0]
                        predB = estimatorB.predict(est_data_)[0][0]

                        # 売買を行う
                        buy_entry, sell_entry, buy_close, sell_close, passage_time, profit = \
                            self.test(data.iloc[num, :4], predA, predB, now_time, buy_entry, sell_entry,
                                      buy_close, sell_close, passage_time, profit)

                        # 一時間に一回データをプロットしてlineに送信
                        if now_time.minute == 0 and now_hour % 3 == 0:
                            self.plot_profit_test(passage_time, profit, now_time)

                    time.sleep(1)

            except KeyError:
                message = "KeyError"
                print(sys.exc_info())
                lutil.line_notify(message)
                time.sleep(1)

            except:
                message = "another error"
                print(sys.exc_info())
                lutil.line_notify(message)
                break
