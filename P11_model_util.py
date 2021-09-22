from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LSTM
from tensorflow.keras.utils import plot_model


# LSTM層を追加する関数
def add_lstm_layer(x, dim, dropout=0.2):
    x = LSTM(dim, dropout=dropout)(x)
    return x


# 全結合層を追加する関数
def add_dense_layer(x, dim, use_bn=True, activation='relu'):
    x = Dense(dim, use_bias=not use_bn)(x)
    x = Activation(activation)(x)
    if use_bn:
        x = BatchNormalization()(x)
    return x


# ネットワーク構造を可視化して保存する関数
def save_model_info(info_file, graph_file, model):
    with open(info_file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_model(model, to_file=graph_file, show_shapes=True)
