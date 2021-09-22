from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

import P10_util as util


class Estimator:

    # コンストラクタ
    def __init__(self, dst_dir, test_file,
                 est_fileA, est_fileB, plot_file, plot_dir, lstm_len, width):
        self.dst_dir = dst_dir
        self.test_file = test_file
        self.est_fileA = est_fileA
        self.est_fileB = est_fileB
        self.plot_dir = plot_dir
        self.plot_file = plot_file
        self.lstm_len = lstm_len
        self.width = width

        pd.set_option('display.max_columns', 20)

    # プログラム全体を制御するメソッド
    def execute(self):
        # モデルの読み込み
        estimator_a = load_model(self.est_fileA)
        estimator_b = load_model(self.est_fileB)

        # 推定の準備
        test_data, test_class_a, test_class_b = util.load_data(self.test_file, self.lstm_len, self.width)

        print("Start estimate.")
        # 推定の実行
        pred_data_a = estimator_a.predict(test_data)
        pred_data_b = estimator_b.predict(test_data)

        # 比較用データの読み込み
        data = pd.read_csv(self.test_file, header=0, index_col=0)
        data = data.iloc[len(data) - len(pred_data_a):, :]

        print("Start plot.")
        plt.figure(figsize=(12.8, 8.6))
        plt.subplot(2, 2, 1)
        plt.scatter(pred_data_a, data["x_max"])
        plt.subplot(2, 2, 2)
        plt.scatter(pred_data_b, data["x_min"])
        plt.subplot(2, 2, 3)
        plt.hist(pred_data_a, bins=10)
        plt.subplot(2, 2, 4)
        plt.hist(pred_data_b, bins=10)
        util.mkdir(self.plot_dir)
        plt.savefig(self.plot_file)
        plt.close("all")
        print("all finish")
