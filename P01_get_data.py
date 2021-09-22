# CryptoWatchからデータを取得し、保存する
# テスト用に用いる
import pandas as pd
import requests
import datetime


class GetCandleStick:
    def __init__(self, out_file, periods):
        self.out_file = out_file
        self.periods = periods

    def getcandlestick(self):
        after = 1514764800
        res = requests.get("https://api.cryptowat.ch/markets/bitflyer/btcfxjpy/ohlc",
                           params={"periods": self.periods, "after": after})
        res = res.json()
        res = res["result"]
        res = res[self.periods]
        data = pd.DataFrame(res, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'volume_2'])
        data['date'] = [datetime.datetime.fromtimestamp(n) for n in data['date']]

        return data

    def execute(self):
        data = self.getcandlestick()
        data.to_csv(self.out_file, index=False)
