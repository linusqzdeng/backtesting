# -*- coding: UTF-8 -*-
# Created on: 20220207
# Tested by: python 3.8.10
# Author: @Qizhong Deng

import pandas as pd
import numpy as np
import backtrader as bt
import empyrical as emp
import pyfolio as pyf

import os, sys
import datetime
import warnings

warnings.filterwarnings("ignore")


class Config:

    data_path = os.path.abspath("../data.csv")
    data = pd.read_csv(data_path, index_col="TRADE_DT", parse_dates=True)
    valid_contracts = ["IF00", "IH00", "IC00"]
    contract = valid_contracts[1]
    msi_path = os.path.abspath(f"./{contract}_msi.csv")
    msi_df = pd.read_csv(msi_path, index_col='Date', parse_dates=True)

    fromdate = datetime.date(2010, 4, 16)
    todate = datetime.date(2021, 12, 31)

    startcash = 10_000_000
    ctp_comm = 3.45 / 10000
    normal_comm = 0.23 / 10000
    stamp_duty = 0.0001

    is_ctp = False  # 平今仓

    def __init__(self):
        self.df = self.create_df(self.data, self.contract)
        self.df = self.add_msi(self.msi_df, self.df)
        self.time_df = self.create_timedf(self.data)
        self.mult = self.set_mult()
        self.margin = self.set_margin()

    def create_df(self, data, contract):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = data[data["S_INFO_CODE"] == contract]

        adj_cols = [
            "S_DQ_ADJOPEN",
            "S_DQ_ADJHIGH",
            "S_DQ_ADJLOW",
            "S_DQ_ADJCLOSE",
        ]
        pre_cols = ["S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE"]

        for pre_col, adj_col in zip(pre_cols, adj_cols):
            df.loc[:, adj_col] = round(df.loc[:, pre_col] * df.loc[:, "S_DQ_ADJFACTOR"], 1)
        df = df[adj_cols]

        return df

    def create_timedf(self, data):
        """Return the time data that contains the trading bars per day"""
        timedf = pd.DataFrame(index=data.index.date)
        timedf["time"] = data.index.time
        timedf = timedf.groupby(timedf.index).apply(lambda x: list(x["time"]))
        timedf.index = pd.to_datetime(timedf.index)

        return timedf
    
    def add_msi(self, msi, data):
        """Add daily msi column to dataframe"""
        out = data.copy()
        for date, today_msi in msi.iterrows():
            date = date.date().isoformat()
            out.loc[date, 'msi'] = float(today_msi)
        
        return out

    def set_margin(self):
        # 保证金设置
        if self.contract in ["IH00", "IC00"]:
            return 0.10
        elif self.contract == "IF00":
            return 0.12

        raise Exception("Unvalid contract name")

    def set_mult(self):
        # 乘数设置
        if self.contract in ["IF00", "IH00"]:
            return 300.0
        elif self.contract == "IC00":
            return 200.0

        raise Exception("Unvalid contract name")


metavar = Config()


class DataInput(bt.feeds.PandasData):

    lines = ('msi',)
    params = (
        ("nullvalue", np.nan),
        ("fromdate", metavar.fromdate),
        ("todate", metavar.todate),
        ("datetime", None),  # index of the dataframe
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ('msi', 4),  # market sentiment index
        ("volume", -1),
        ("openinterest", -1),
    )


class Logger:
    """将输出内容保存到本地文件"""

    def __init__(self, filename=f"./logs/{metavar.contract}trades.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass


class MySizer(bt.Sizer):

    params = ()

    def _getsizing(self, comminfo, cash, data, isbuy):
        pass


class MyCommInfo(bt.CommInfoBase):

    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("ctp_comm", metavar.ctp_comm),
        ("normal_comm", metavar.normal_comm),
        ("mult", metavar.mult),
        ("stamp_duty", metavar.stamp_duty),  # 印花税0.1%
        ("margin", metavar.margin),
        ("backtest_margin", 1.0),  # no leverage
    )

    def _getcommission(self, size, price, pseudoexec):
        """手续费=买卖手数*合约价格*手续费比例*合约乘数"""
        if metavar.is_ctp == True:
            self.p.commission = self.p.ctp_comm
        else:
            self.p.commission = self.p.normal_comm

        return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.margin


class MyStrats(bt.Strategy):

    params = (
        ("period", 50),
        ("stop_limit", 0.005),
        ("msi_threshold", 9 / 10000),
        ("target_percent", 0.10),
    )

    def __init__(self):
        self.dataopen = self.datas[0].open
        self.dataclose = self.datas[0].close
        self.datadatetime = self.datas[0].datetime
        self.datamsi = self.datas[0].msi

        self.order = None
        self.open_price = None
        self.buy_price = None
        self.sell_price = None

        self.open_bar = 1

    def log(self, txt, dt=None):
        dt = dt or self.datadatetime.datetime(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            margin_used = order.executed.price * abs(order.executed.size) * metavar.mult * metavar.margin

            if order.isbuy():
                self.log(
                    "LONG DETECTED @ {:.2f}, EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}".format(
                        order.created.price,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        margin_used,
                    )
                )
                self.buy_price = order.executed.price

            elif order.issell():
                self.log(
                    "SHORT DETECTED @ {:.2f}, EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}".format(
                        order.created.price,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        margin_used,
                    )
                )
                self.sell_price = order.executed.price

            if order.info:
                self.log(f"INFO {self.order.info['name']}")

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER CANCELED/MARGIN/REJECTED **CODE**: {order.getstatusname()}")

        # Write down if no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.bar_traded = len(self)
        self.log(f"OPERATION PROFIT {trade.pnl:.2f}, NET PROFIT {trade.pnlcomm:.2f}, TRADE AT BAR {self.bar_traded}")

    def start(self):
        pass

    def next(self):
        # Time management
        date = bt.num2date(self.datadatetime[0]).date().isoformat()
        time = bt.num2time(self.datadatetime[0])
        trade_time = metavar.time_df.loc[date]
        open_time = trade_time[0]
        close_time = [trade_time[-2], trade_time[-1]]  # 使用收盘时间前一个bar作为平今仓信号

        # 当日开盘价
        if time == open_time:
            self.open_price = self.dataopen[0]
            self.open_bar = len(self)

        bypass_conds = [
            self.order,  # 订单正在进行中
            len(self) < self.open_bar + self.p.period - 1,  # 平稳度观测区间
            self.datamsi[0] > self.p.msi_threshold  # 市场不平稳
            ]
        if any(bypass_conds):
            return

        # 满足上述条件后进行交易
        longsig = self.dataclose[0] >= self.open_price
        shortsig = self.dataclose[0] < self.open_price

        metavar.is_ctp = False
        if not self.position:
            if time not in  close_time:  # 临近收盘不建仓
                if longsig:
                    self.order = self.order_target_percent(target=self.p.target_percent)
                elif shortsig:
                    self.order = self.order_target_percent(target=-self.p.target_percent)
        else:
            # 平今仓
            if time == close_time[-2]:
                metavar.is_ctp = True
                self.order =  self.close()
                self.order.addinfo(name="CLOSE AT THE END OF THE DAY")
                return

            # 止损
            if self.position.size > 0:
                pct_change = (self.dataclose[0] - self.buy_price) / self.buy_price
                if pct_change < -self.p.stop_limit:
                    self.order = self.close()
                    self.order.addinfo(name="CLOSE DUE TO STOPLIMIT")
            else:
                pct_change = (self.dataclose[0] - self.sell_price) / self.sell_price
                if pct_change > self.p.stop_limit:
                    self.order = self.close()
                    self.order.addinfo(name="CLOSE DUE TO STOPLIMIT")

    def stop(self):
        pass


def normal_analysis(strats):
    # =========== for analysis.py ============ #
    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())
    pyfoliozer = strats.analyzers.getbyname("pyfolio")
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

    perf_df = pyf.timeseries.perf_stats(returns, positions=positions, transactions=transactions)

    returns.to_csv("./results/returns.csv")
    positions.to_csv("./results/positions.csv")
    transactions.to_csv("./results/transactions.csv")
    gross_lev.to_csv("./results/gross_lev.csv")
    perf_df.to_csv("./results/perf_df.csv")
    rets.to_csv("./results/timereturn.csv", index=True)
    # ======================================== #

    cumrets = emp.cum_returns(rets, starting_value=0)
    num_years = metavar.todate.year - metavar.fromdate.year
    ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    max_drawdown = emp.max_drawdown(rets)

    summary = pyf.create_simple_tear_sheet(rets)

    # 夏普比率
    yearly_trade_times = rets.shape[0] / num_years
    sharpe = emp.sharpe_ratio(rets, risk_free=0, annualization=yearly_trade_times)  # 4.5h 交易时间

    # 盈亏比
    mean_per_win = (rets[rets > 0]).mean()
    mean_per_loss = (rets[rets < 0]).mean()

    # 单次交易最大最小值
    day_ret_max = rets.max()
    day_ret_min = rets.min()

    results_dict = {
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "累计收益率": cumrets[-1],
        "年化收益率": ann_rets,
        "收益回撤比": ann_rets / -max_drawdown,
        "单日最大收益": day_ret_max,
        "单日最大亏损": day_ret_min,
        "交易次数": len(transactions),
        "获胜次数": round(sum(rets > 0), 0),
        "胜率": sum(rets > 0) / rets.shape[0],
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    print(results_df)
    print(perf_df)
    print(summary)


def opt_analysis(results):
    def get_analysis(result):
        analysers = {}
        analysers["thold_s"] = result.params.thold_s
        analysers["thold_l"] = result.params.thold_l

        # 返回参数
        rets = pd.Series(result.analyzers._TimeReturn.get_analysis())

        # 夏普比率
        cumrets = emp.cum_returns(rets, starting_value=0)
        max_drawdown = emp.max_drawdown(rets)
        num_years = metavar.todate.year - metavar.fromdate.year
        ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
        calmar = ann_rets / -max_drawdown
        yearly_trade_times = rets.shape[0] / num_years
        sharpe = emp.sharpe_ratio(rets, risk_free=0, annualization=yearly_trade_times)  # 4.5h 交易时间

        analysers["cumrets"] = cumrets[-1]
        analysers["ann_rets"] = ann_rets
        analysers["max_drawdown"] = max_drawdown
        analysers["calmar_ratio"] = calmar
        analysers["sharpe"] = sharpe

        return analysers

    opt_results = [get_analysis(i[0]) for i in results]
    opt_df = pd.DataFrame(opt_results)
    opt_df.to_csv("./results/opt_results.csv")

    print(opt_df)


def run():
    sys.stdout = Logger()

    # Initialisation
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MyStrats)

    data = DataInput(dataname=metavar.df)
    cerebro.adddata(data)

    comminfo = MyCommInfo()
    cerebro.addsizer(MySizer)
    cerebro.broker.setcash(metavar.startcash)
    cerebro.broker.addcommissioninfo(comminfo)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="_TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="_CalmarRatio")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    # Observers
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.TimeReturn)

    # Start backtesting
    print(f"开始资金总额 {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    # results = cerebro.run(maxcpus=1)
    print(f"结束资金总额 {cerebro.broker.getvalue():.2f}")

    strats = results[0]
    normal_analysis(strats)

    # opt_analysis(results)


if __name__ == "__main__":
    run()
    # print(metavar.df)
