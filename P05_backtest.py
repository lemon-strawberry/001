from tensorflow.keras.models import load_model
import pandas as pd

import P10_util as util


class BackTest:
    def __init__(self, est_fileA, est_fileB, est_data_file, save_dir, save_file, lstm_len, width, x_width, lot, x_term,
                 sfd_long, sfd_short, cost):
        self.est_fileA = est_fileA
        self.est_fileB = est_fileB
        self.est_data_file = est_data_file
        self.save_dir = save_dir
        self.save_file = save_file
        self.lstm_len = lstm_len
        self.width = width
        self.x_width = x_width
        self.lot = lot
        self.x_term = x_term
        self.sfd_long = sfd_long
        self.sfd_short = sfd_short
        self.cost = cost

    def backtest(self, data, pred_data_a, pred_data_b):
        buy_entry_signals = []
        sell_entry_signals = []
        buy_close_signals = []
        sell_close_signals = []
        count_trade = 0
        pos = 0
        pl = [0]

        pl_per_trade = []

        buy_entry = []
        buy_close = []
        sell_entry = []
        sell_close = []

        for i in range(len(data)):
            if i > 0:
                last_pl = pl[-1]
                pl.append(last_pl)

            # close
            # long_close
            if pos > 0:
                if data["a_high"][i] >= int(buy_entry[-1] * (1+0.01*self.width)):
                    count_trade += 1
                    pos = 0
                    buy_close.append(int(buy_entry[-1] * (1+0.01*self.width)))

                    pl_range = buy_close[-1] - buy_entry[-1] - buy_close[-1] * self.sfd_long * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    buy_close_signals.append(data.index[i])

                elif data["a_low"][i] <= int(buy_entry[-1] * (1-0.01*self.width)):
                    count_trade += 1
                    pos = 0
                    buy_close.append(int(buy_entry[-1] * (1-0.01*self.width)))

                    pl_range = buy_close[-1] - buy_entry[-1] - buy_close[-1] * self.sfd_long * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    buy_close_signals.append(data.index[i])

                elif pos >= self.x_term:
                    count_trade += 1
                    pos = 0
                    buy_close.append(data["a_close"][i])

                    pl_range = buy_close[-1] - buy_entry[-1] - buy_close[-1] * self.sfd_long * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    buy_close_signals.append(data.index[i])

                else:
                    pos += 1

            elif pos < 0:
                if data["a_low"][i] <= int(sell_entry[-1] * (1-0.01*self.width)):
                    count_trade += 1
                    pos = 0
                    sell_close.append(int(sell_entry[-1] * (1-0.01*self.width)))

                    pl_range = sell_entry[-1] - sell_close[-1] - sell_close[-1] * self.sfd_short * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    sell_close_signals.append(data.index[i])

                elif data["a_high"][i] >= int(sell_entry[-1] * (1+0.01*self.width)):
                    count_trade += 1
                    pos = 0
                    sell_close.append(int(sell_entry[-1] * (1 + 0.01 * self.width)))

                    pl_range = sell_entry[-1] - sell_close[-1] - sell_close[-1] * self.sfd_short * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    sell_close_signals.append(data.index[i])

                elif pos <= -self.x_term:
                    count_trade += 1
                    pos = 0
                    sell_close.append(int(data["a_close"][i]))

                    pl_range = sell_entry[-1] - sell_close[-1] - sell_close[-1] * self.sfd_short * 0.01
                    pl_per_trade.append((pl_range - self.cost) * self.lot)
                    pl[-1] = pl[-2] + pl_per_trade[-1]
                    sell_close_signals.append(data.index[i])

                else:
                    pos -= 1

            # entry
            if pos == 0:
                if pred_data_a[i] >= self.x_width and pred_data_b[i] >= self.x_width:
                    pass
                # long_entry
                elif pred_data_a[i] >= self.x_width and pred_data_b[i] <= self.x_width * 0.9:
                    pos = 1
                    buy_entry.append(data['a_close'][i])
                    buy_entry_signals.append(data.index[i])

                # short_entry
                elif pred_data_b[i] >= self.x_width and pred_data_a[i] <= self.x_width * 0.9:
                    pos = -1
                    sell_entry.append(data['a_close'][i])
                    sell_entry_signals.append(data.index[i])

        return pl, buy_entry_signals, sell_entry_signals, buy_close_signals, sell_close_signals, count_trade, pl_per_trade

    # データの集計を行うメソッド
    def totalization(self, plofit_per_trade, count_trade, pl):
        win_trade = sum([1 for i in plofit_per_trade if i > 0])
        lose_trade = sum([1 for i in plofit_per_trade if i < 0])
        if count_trade == 0:
            win_per = 0
        else:
            win_per = round(win_trade / count_trade * 100, 2)

        win_total = sum([i for i in plofit_per_trade if i > 0])
        lose_total = sum([i for i in plofit_per_trade if i < 0])
        if lose_total == 0:
            profit_factor = 0
        else:
            profit_factor = round(win_total / -lose_total, 3)

        if plofit_per_trade == []:
            max_profit = 0
            max_loss = 0
        else:
            max_profit = max(plofit_per_trade)
            max_loss = min(plofit_per_trade)

        if win_trade != 0:
            win_average = round(win_total / win_trade, 2)
        else:
            win_average = 0
        if lose_trade != 0:
            lose_average = round(lose_total / lose_trade, 2)
        else:
            lose_average = 0

        expected_value = win_per * 0.01 * win_average + (1 - win_per * 0.01) * lose_average

        print("Total pl: {}JPY".format(int(pl[-1])))
        print("The number of Trades: {}".format(count_trade))
        print("The Winning percentage: {}%".format(win_per))
        print("The profitFactor: {}".format(profit_factor))
        print("The maximum Profit and Loss: {}JPY, {}JPY".format(max_profit, max_loss))
        print("Profit Average Win:{}JPY Lose:{}JPY".format(win_average, lose_average))
        print("expected_value: {} JPY/trade".format(expected_value))

    def execute(self):
        # 学習済みモデルの読み込み
        estimator_a = load_model(self.est_fileA)
        estimator_b = load_model(self.est_fileB)

        # 推定用データの読み込み
        test_data, test_class_a, test_class_b = util.load_data(self.est_data_file, self.lstm_len, self.width)

        # 推定の実行
        pred_data_a = estimator_a.predict(test_data)
        pred_data_b = estimator_b.predict(test_data)

        del test_data
        del estimator_a
        del estimator_b

        # backtest用データの読み込み
        data = pd.read_csv(self.est_data_file, header=0, index_col=0)
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
        data = data.iloc[len(data) - len(pred_data_a):, :]

        # 売買を記録
        pl, buy_entry_signal, sell_entry_signal, buy_close_signal, sell_close_signal, count_trade, plofit_per_trade = \
            self.backtest(data, pred_data_a, pred_data_b)
        del pred_data_a, pred_data_b

        self.totalization(plofit_per_trade, count_trade, pl)

        # データの可視化と保存
        util.mkdir(self.save_dir, rm=False)
        util.plot2(data, pl, buy_entry_signal, sell_entry_signal, buy_close_signal, sell_close_signal, self.save_file)
        print("\nfinish...")
