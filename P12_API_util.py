# 注文出しやデータの取得を行う
import datetime
import time
import re

import pandas as pd
import pybitflyer

api_key = ''
api_secret = ''
api = pybitflyer.API(api_key=api_key, api_secret=api_secret)
product_code = "FX_BTC_JPY"


# 約定データを取得する関数
def get_exec_data(timedelta, pure_data=0, first=True, before_last_id=0):
    if first:
        response = {"status": "internalError in execute.py"}
        # 一回データを取得
        try:
            response = api.executions(product_code=product_code, count=500)
        except:
            print('a')
            pass
        while "status" in response:
            try:
                response = api.executions(product_code=product_code, count=500)
            except:
                print('a')
                pass
            time.sleep(5)
        # 一番古い時間と新しい時間を取得
        last_date = datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', response[-1]['exec_date'].replace('T', ' ')),
                                               "%Y-%m-%d %H:%M:%S")
        new_date = datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', response[0]['exec_date'].replace('T', ' ')),
                                              "%Y-%m-%d %H:%M:%S")

        # 一番古い値が2時間以上前の足になるまでデータを取得と合算を繰り返す
        while new_date - last_date < timedelta:
            last_id = response[-1]['id']
            add_response = {"status": "internalError in execute.py"}
            try:
                add_response = api.executions(product_code=product_code, count=500, before=last_id)
            except:
                print('a')
                pass
            while "status" in add_response:
                try:
                    add_response = api.executions(product_code=product_code, count=500, before=last_id)
                except:
                    pass
                time.sleep(5)
            response = response + add_response
            last_date = datetime.datetime.strptime(re.sub(r'[.][0-9]+', '',
                                                          response[-1]['exec_date'].replace('T', ' ')),
                                                   "%Y-%m-%d %H:%M:%S")

    else:
        response = {"status": "internalError in execute.py"}
        try:
            response = api.executions(product_code=product_code, count=500, after=before_last_id - 1)
        except:
            print('a')
            pass
        while "status" in response:
            try:
                response = api.executions(product_code=product_code, count=500, after=before_last_id - 1)
            except:
                print('a')
                pass
            time.sleep(5)

        while response[-1]['id'] > before_last_id:
            last_id = response[-1]['id']
            add_response = {"status": "internalError in execute.py"}
            try:
                add_response = api.executions(product_code=product_code, count=500, before=last_id,
                                              after=before_last_id - 1)
            except:
                pass
            while "status" in add_response:
                try:
                    add_response = api.executions(product_code=product_code, count=500, before=last_id,
                                                  after=before_last_id - 1)
                except:
                    pass
                time.sleep(5)
            response = response + add_response
        response = response + pure_data
        response = [n for n in response if
                    datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', n['exec_date'].replace('T', ' ')),
                                               "%Y-%m-%d %H:%M:%S") >
                    (datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', n['exec_date'].replace('T', ' ')),
                                                "%Y-%m-%d %H:%M:%S") - timedelta)]

    data = [(res['exec_date'], res['id'], res['price']) for res in response]
    data = pd.DataFrame(data, columns=['date', 'id', 'price'])
    data['date'] = [datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', n.replace('T', ' ')), "%Y-%m-%d %H:%M:%S") for n
                    in data['date']]
    data.set_index('date', inplace=True)
    data[::-1]

    # dataは計算に使う
    # resoponse は次のデータの取得に使う。だからデータの変形をする前のを返す
    return data, response


# 成行注文を出す関数
def market(side, size):
    response = {"status": "internalError in order.py"}
    try:
        response = api.sendchildorder(product_code=product_code, child_order_type="MARKET", side=side, size=size)
    except:
        pass
    while "status" in response:
        try:
            response = api.sendchildorder(product_code=product_code, child_order_type="MARKET", side=side, size=size)
        except:
            pass
        time.sleep(3)


# 証拠金の変動履歴を取得
def get_collateral_history(first_time=0, id=None):
    response = {"status": "internalError in get_collateral_history.py"}
    if id == None:
        try:
            response = api.getcollateralhistory(count=1)
        except:
            pass
        while "status" in response:
            try:
                response = api.getcollateralhistory(count=1)
            except:
                pass
        return response[0]['id']

    else:
        try:
            response = api.getcollateralhistory(count=500, after=id)
        except:
            pass
        while "status" in response:
            try:
                response = api.getcollateralhistory(count=1, after=id)
            except:
                pass
        if len(response) != 0:
            response = [(data['date'], data['change']) for data in response]
            response = pd.DataFrame(response, columns=['date', 'price'])
            response['date'] = [datetime.datetime.strptime(re.sub(r'[.][0-9]+', '', n.replace('T', ' ')),
                                                           "%Y-%m-%d %H:%M:%S") for n in response['date']]
            response.set_index('date', inplace=True)
            response.loc[first_time] = 0
            response[::-1]

        else:
            response = pd.DataFrame([0], columns=['price'], index=[first_time])

        return response


# 現在の板情報を得る関数
# フォワードテストで使うデータ
def get_board(ask_or_bid):
    # ask_or_bid　には 'asks' か 'bids'　を指定
    response = {"status": "internalError in get_board.py"}
    try:
        response = api.board(product_code=product_code)
    except:
        pass
    while "status" in response:
        try:
            response = api.board(product_code=product_code)
        except:
            pass

    return response[ask_or_bid][0]['price']
