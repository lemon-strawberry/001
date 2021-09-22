from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import P10_util as util
import P11_model_util as mutil


class ModelMaker:

    # コンストラクタ
    def __init__(self, est_dir, est_file, est_file_b, cls_file, info_file,
                 graph_file, hist_file, hist_file_b, train_data_file,
                 lstm_dims, dims, lr, min_lr, lstm_len, batch_size, epochs,
                 valid_rate, es_patience, lr_patience, width):
        self.est_dir = est_dir
        self.est_file = est_file
        self.est_fileB = est_file_b
        self.cls_file = cls_file
        self.info_file = info_file
        self.graph_file = graph_file
        self.hist_file = hist_file
        self.hist_fileB = hist_file_b
        self.train_data_file = train_data_file
        self.lstm_dims = lstm_dims
        self.dims = dims
        self.lr = lr
        self.min_lr = min_lr
        self.lstm_len = lstm_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_rate = valid_rate
        self.es_patience = es_patience
        self.lr_patience = lr_patience
        self.width = width

    # モデルを定義するメソッド
    def define_model(self, width):
        # 入力層の定義
        input_x = Input(shape=(self.lstm_len, width))
        x = input_x

        # LSTM層の定義
        x = mutil.add_lstm_layer(x, self.lstm_dims)

        # 結合層の定義
        if len(self.dims) > 1:
            for dim in self.dims[:-1]:
                x = mutil.add_dense_layer(x, dim)

        # 出力層の定義
        x = mutil.add_dense_layer(x, self.dims[-1], use_bn=False, activation='sigmoid')

        # モデル全体の入出力を定義
        model = Model(input_x, x)

        model.compile(
            optimizer=Adam(lr=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    # モデルを訓練するメソッド
    def fit_model(self):
        # データの読み込み
        train_data, train_classesA, train_classesB = util.load_data(self.train_data_file, self.lstm_len, self.width)

        # モデルを定義
        model = self.define_model(len(train_data[0][0]))
        model2 = self.define_model(len(train_data[0][0]))

        # コールバックの定義
        early_stopping = EarlyStopping(
            patience=self.es_patience,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr_op = ReduceLROnPlateau(
            patience=self.lr_patience,
            min_lr=self.min_lr,
            verbose=1
        )
        callbacks = [early_stopping, reduce_lr_op]

        # 訓練の実行
        history = model.fit(
            train_data,
            train_classesA,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.valid_rate,
            callbacks=callbacks
        )

        history2 = model2.fit(
            train_data,
            train_classesB,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.valid_rate,
            callbacks=callbacks
        )

        return model, model2, history.history, history2.history

    # プログラム全体を制御するメソッド
    def execute(self):
        # モデルを訓練
        model, model2, history_a, history_b = self.fit_model()

        # 訓練したモデルを保存
        util.mkdir(self.est_dir, rm=True)
        model.save(self.est_file)
        model2.save(self.est_fileB)

        # ネットワーク構造を保存
        mutil.save_model_info(self.info_file, self.graph_file, model)

        # 訓練状況を保存
        util.plot(history_a, self.hist_file)
        util.plot(history_b, self.hist_fileB)

        # リスト内の最小値とそのインデックスを返す処理
        def get_min(loss):
            min_val = min(loss)
            min_ind = loss.index(min_val)
            return min_val, min_ind

        # 検証用データにおける最小の損失を標準出力(1回目訓練終了後)
        print('Before fine-tuning')
        min_val, min_ind = get_min(history_a['val_loss'])
        print('val_loss: %f (Epoch: %d)' % (min_val, min_ind + 1))
        min_val, min_ind = get_min(history_b['val_loss'])
        print('val_loss: %f (Epoch: %d)' % (min_val, min_ind + 1))