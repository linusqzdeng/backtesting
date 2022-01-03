# -*- coding: UTF-8 -*-

import pandas as pd

import os
import datetime

class Contract:

    valid_contracts = ["IF00", "IH00", "IC00"]

    def __init__(self, contract: str, fromdate=None, todate=None):
        """
        Params
        ------
        - contract:
            name of the main contract
        - fromdate:
            datetime object specifies the start date
            default value is the start date of the whole dataset
        - todate:
            datetime object sepcifies the end date
            default value is tne end date of the whole dataset
        """
        # 合约设置
        if contract in self.valid_contracts:
            self.contract = contract
        else:
            print("Invalid contract name...")

        # 交易时间表
        time_filepath = os.path.join(
                os.path.abspath("."),
                "trading_time", self.contract + "_time.csv"
                )  # 5min data
        self.time_df = self.get_timedf(time_filepath)

        # 回测区间设置
        self._fromdate = fromdate or self.time_df.index[0].to_pydatetime().date()
        self._todate = todate or self.time_df.index[-1].to_pydatetime().date()

        # 保证金比例和合约乘数
        self.margin = self.set_margin(self.contract)
        self.mult = self.set_mult(self.contract)

        # 数据路径
        self.filepath = os.path.join(
                os.path.abspath("."),
                "5m_main_contracts", self.contract + ".csv"
                )  # 5min data

        # 初始资金
        self.startcash = 10_000_000

        # 平仓类型 
        self.closeout_type = 1  # 1代表平今仓，0代表止盈止损平仓

        # 滑点设置

        # 印花税设置
        self.stamp_duty = 0.001

    def get_timedf(self, path):
        """Return the time df that contains the trading bars per day"""
        timedf = pd.read_csv(path, index_col='date')
        timedf.index = pd.to_datetime(timedf.index)

        return timedf

    def set_margin(self, contract):
        # 保证金设置
        if contract in ["IH00", "IC00"]:
            margin = 0.10
        elif contract == "IF00":
            margin = 0.12

        return margin
    
    def set_mult(self, contract):
        # 乘数设置
        if contract in ["IF00", "IH00"]:
            mult = 300.0
        elif contract == "IC00":
            mult = 200.0

        return mult


contract_name = 'IF00'
fromdate = datetime.date(2015, 6, 16)
todate = datetime.date(2017, 4, 16)

def set_contract_var():
    return Contract(contract_name, fromdate, todate)

