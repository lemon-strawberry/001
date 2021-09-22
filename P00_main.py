"""
n期間の間にx％上昇・下落する確立を予測するモデルを上昇と下落に分け学習を行う(二値分類を二つ)
事前に入手しているデータは全て学習に用いり、テストにはcryptowatchから入手した直近のデータを用いる。
"""
import os
import sys

#データ処理の設定
CON_DATA_FILE = 'D00_ohlcvData/15T_ohlcv_data.csv'
COT_DATA_FILE = 'D00_ohlcvData/crypto_data.csv'
CON_OUT_FILE = 'D00_ohlcv_data/OHLCV_15T_data.csv'
COT_OUT_FILE = 'D00_ohlcv_data/OHLCV_cry_data.csv'
OUTLIER_TH = 3
SMA_SHORT_TERM = 7
SMA_LONG_TERM = 21
EMA_SHORT_TERM = 7
EMA_LONG_TERM = 21
MACD_TERM = 9
BOL_TERM = SMA_LONG_TERM
STO_K_TERM = 14
STO_S_TERM = 3
STO_D_TERM = 3
X_TERM = 12

#CryptoWatchからデータを保存
CRY_OUT_FILE = 'D00_ohlcv_data/crypto_data.csv'
CRY_PERIOD = "900" #秒指定（60で1分）

#推定器構築ステップ用の設定
EST_DIR = 'D01_estimator'
EST_FILE = os.path.join(EST_DIR, 'estimator.h5')
EST_FILEB = os.path.join(EST_DIR, 'estimatorB.h5')
CLS_FILE = os.path.join(EST_DIR, 'class.pkl')
INFO_FILE = os.path.join(EST_DIR, 'model_info.txt')
GRAPH_FILE = os.path.join(EST_DIR, 'model_graph.pdf')
HIST_FILE = os.path.join(EST_DIR, 'history.pdf')
HIST_FILEB = os.path.join(EST_DIR, 'history2.pdf')
LSTM_DIMS = 8
MES_DIMS = [1]
MES_LR = 1e-3
MES_MIN_LR = 1e-15
LSTM_LEN = 16
BATCH_SIZE = 256
EPOCHS = 4000
VARID_RATE = 0.2
ES_PATIENCE = 20
LR_PATIENCE = 10
WIDTH = 1

#推定スッテプ用の設定
EST_DST_DIR = 'D02_result'
EST_DRS_FILE = os.path.join(EST_DST_DIR, 'detailed_result.txt')
EST_SRS_FILE = os.path.join(EST_DST_DIR, 'summary_result.txt')

PLOT_DIR = "D03_plot"
PLOT_FILE = os.path.join(PLOT_DIR, "est.pdf")

#データを保存するファイルの設定
SAVE_DIR = 'D02_result'
SAVE_FILE = os.path.join(SAVE_DIR, 'past01.pdf')
#その他の設定
COST = 0
LOT = 0.02
SFD_LONG = 0
SFD_SHORT = 0
X_WIDTH = 0.25

PLT_DIR = "D04_plot"
PLT_FILE = os.path.join(PLT_DIR, "profit.png")

# 使用方法を表示
if len(sys.argv) == 1:
    print('Usage: python3 %s STEP...' % sys.argv[0])
    print('''
------------
Step options
------------
con, cot: Converts data format for analysis.
    (con: labeled training data, cot: test data)

cry: Get data.

mes: Makes an estimator.

est: Conducts estimation.
    ''')
    sys.exit(0)

#Cryptowatchからデータを取得
if 'cry' in sys.argv:
    print('Get cryptowatch data...')
    from P01_get_data import GetCandleStick
    getcandlestick = GetCandleStick(
        out_file = CRY_OUT_FILE,
        periods = CRY_PERIOD
    )
    getcandlestick.execute()

#データ変換ステップを実行
if any(step in ['con', 'cot'] for step in sys.argv):
    from P02_data_exchange import DataHenkan

    if 'con' in sys.argv:
        print('[Labeled] Converting data...')
        datahenkan = DataHenkan(
            data_file = CON_DATA_FILE,
            out_file = CON_OUT_FILE,
            outlier_th = OUTLIER_TH,
            sma_short_term = SMA_SHORT_TERM,
            sma_long_term = SMA_LONG_TERM,
            ema_short_term = EMA_SHORT_TERM,
            ema_long_term = EMA_LONG_TERM,
            macd_term = MACD_TERM,
            bol_term = BOL_TERM,
            sto_k_term = STO_K_TERM,
            sto_s_term = STO_S_TERM,
            sto_d_term = STO_D_TERM,
            x_term = X_TERM
        )
        datahenkan.execute()

    if 'cot' in sys.argv:
        print('[Test] Converting data...')
        datahenkan = DataHenkan(
            data_file = COT_DATA_FILE,
            out_file = COT_OUT_FILE,
            outlier_th = OUTLIER_TH,
            sma_short_term = SMA_SHORT_TERM,
            sma_long_term = SMA_LONG_TERM,
            ema_short_term = EMA_SHORT_TERM,
            ema_long_term = EMA_LONG_TERM,
            macd_term = MACD_TERM,
            bol_term = BOL_TERM,
            sto_k_term = STO_K_TERM,
            sto_s_term = STO_S_TERM,
            sto_d_term = STO_D_TERM,
            x_term = X_TERM
        )
        datahenkan.execute()

#訓練を行う
if 'mes' in sys.argv:
    print('Making an estimator...')
    from P03_model_maker import ModelMaker
    maker = ModelMaker(
        est_dir = EST_DIR,
        est_file = EST_FILE,
        est_file_b = EST_FILEB,
        cls_file = CLS_FILE,
        info_file = INFO_FILE,
        graph_file = GRAPH_FILE,
        hist_file = HIST_FILE,
        hist_file_b = HIST_FILEB,
        train_data_file = CON_OUT_FILE,
        lstm_dims = LSTM_DIMS,
        dims = MES_DIMS,
        lr = MES_LR,
        min_lr = MES_MIN_LR,
        lstm_len = LSTM_LEN,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        valid_rate = VARID_RATE,
        es_patience = ES_PATIENCE,
        lr_patience = LR_PATIENCE,
        width = WIDTH
    )
    maker.execute()

#推定ステップを実行
if 'est' in sys.argv:
    print('Conducting estimation...')
    from P04_estimetor import Estimator
    estimator = Estimator(
        dst_dir = EST_DST_DIR,
        test_file = COT_OUT_FILE,
        est_fileA = EST_FILE,
        est_fileB = EST_FILEB,
        plot_file = PLOT_FILE,
        plot_dir = PLOT_DIR,
        lstm_len = LSTM_LEN,
        width = WIDTH
    )
    estimator.execute()

#バックテストの実行
if 'test' in sys.argv:
    print('Back test...')
    from P05_backtest import BackTest
    backtest = BackTest(
        est_fileA = EST_FILE,
        est_fileB = EST_FILEB,
        est_data_file = COT_OUT_FILE,
        save_dir = SAVE_DIR,
        save_file = SAVE_FILE,
        lstm_len = LSTM_LEN,
        width = WIDTH,
        x_width = X_WIDTH,
        lot = LOT,
        x_term = X_TERM,
        sfd_long = SFD_LONG,
        sfd_short = SFD_SHORT,
        cost = COST
    )
    backtest.execute()

#実環境
if 'running' in sys.argv:
    print("Start running bot")
    from P06_forwardtest import RunningBot
    runningbot = RunningBot(
        periods=CRY_PERIOD,
        est_fileA=EST_FILE,
        est_fileB=EST_FILEB,
        lstm_len=LSTM_LEN,
        width=WIDTH,
        x_width=X_WIDTH,
        x_term=X_TERM,
        lot=LOT,
        plt_dir=PLT_DIR,
        plt_file=PLT_FILE,
        OUTLIER_TH=OUTLIER_TH,
        SMA_SHORT_TERM=SMA_SHORT_TERM,
        SMA_LONG_TERM=SMA_LONG_TERM,
        EMA_SHORT_TERM=EMA_SHORT_TERM,
        EMA_LONG_TERM=EMA_LONG_TERM,
        MACD_TERM=MACD_TERM,
        BOL_TERM=BOL_TERM,
        STO_K_TERM=STO_K_TERM,
        STO_S_TERM=STO_S_TERM,
        STO_D_TERM=STO_D_TERM,
        X_TERM=X_TERM
    )
    runningbot.execute()