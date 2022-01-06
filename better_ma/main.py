# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import backtrader as bt
import empyrical as emp

import datetime
import os, sys


class Config:
    """For global variable"""

    valid_contracts = ['IF00', 'IH00', 'IC00']

    def __init__(self):
        # 回测区间
        self._fromdate = datetime.date(2010, 4, 16)
        self._todate = datetime.date(2015, 4, 15)

        # 标的合约
        self.contract = 'IF00'
        self.freq = '10min'

        # 数据处理
        filepath = os.path.join(os.path.abspath('..'),'data.csv')
        raw_df = pd.read_csv(filepath, index_col='TRADE_DT')
        self.df = self.create_df(raw_df, self.contract, self.freq)

        # 交易参数
        self.mult = self.set_mult()
        self.margin = self.set_margin()

        self.stamp_duty = 0.001
        self.startcash = 10_000_000

        # 平今仓
        self.closeout_type = 1


    def create_df(self, raw_df, contract, freq):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = raw_df[raw_df['S_INFO_CODE'] == contract]

        # convert index to datetime object
        df.index = pd.to_datetime(df.index)

        # resampling
        df = df.resample(rule=freq, origin='end').last().dropna()

        # calculate adj prices
        adj_cols = ["S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE",]
        pre_cols = ["S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE"]

        for pre_col, adj_col in zip(pre_cols, adj_cols):
            df[adj_col] = round(df[pre_col] * df['S_DQ_ADJFACTOR'], 1)

        return df[adj_cols]

    def set_margin(self):
        # 保证金设置
        if self.contract in ["IH00", "IC00"]:
            margin = 0.10
        elif self.contract == "IF00":
            margin = 0.12
        else:
            raise ValueError('Invalid contract name')

        return margin
    
    def set_mult(self):
        # 乘数设置
        if self.contract in ["IF00", "IH00"]:
            mult = 300.0
        elif self.contract == "IC00":
            mult = 200.0
        else:
            raise ValueError('Invalid contract name')

        return mult

# global variable
metavar = Config()


class Logger:
    """将打印出的交易单保存为log文件"""

    def __init__(self, filename=f"./logs/{metavar.contract}trades.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass


class MainContract(bt.feeds.PandasData):
    params = (
        ("nullvalue", np.nan),
        ("fromdate", metavar._fromdate),
        ("todate", metavar._todate),
        ("datetime", None),  # index of the dataframe
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ("volume", -1),
        ("openinterest", -1),
    )


class FurCommInfo(bt.CommInfoBase):
    """定义期货的交易手续费和佣金"""

    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("commission", 3.45 / 10_000),
        ("mult", metavar.mult),
        ("stamp_duty", metavar.stamp_duty),  # 印花税0.1%
        ("margin", metavar.margin),
        ("backtest_margin", 1.0),  # no leverage for now
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        手续费=买卖手数*合约价格*手续费比例*合约乘数

        根据平仓类型`metavar.closeout_type`决定手续费比例
        - 平昨仓/止盈止损平仓: 0.23 / 10000
        - 平今仓: 3.45 / 10000
        """
        if metavar.closeout_type == 1:
            self.p.commission = 3.45 / 10000  # 平今仓
        else:
            self.p.commission = 0.23 / 10000  # 止盈止损平仓/开仓

        if size > 0:
            return abs(size) * price * self.p.commission * self.p.mult
        else:  # 卖出时考虑印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty) * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.backtest_margin


class BetterMA(bt.Strategy):
    params = (
        ('fast_sma', 60),
        ('slow_sma', 120),
        ('closeout_limit', 0.02),
        ('target_percent', 0.3),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime

        # 计算指标
        fast_sma = bt.ind.MovingAverageSimple(period=self.p.fast_sma)
        slow_sma = bt.ind.MovingAverageSimple(period=self.p.slow_sma)
        self.crossover = bt.ind.CrossOver(fast_sma, slow_sma)

        self.buy_price = None
        self.sell_price = None
        self.buy_create = None  # 金叉价格
        self.sell_create = None  # 死叉价格

        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            # 保证金占用
            margin_used = order.executed.price * abs(order.executed.size) * metavar.mult * metavar.margin

            if order.isbuy():
                # 记录订单完成时间
                self.ordermin = bt.num2time(self.datadatetime[0]).isoformat()
                self.log(f"LONG SIG DETECTED @ {order.created.price:.2f}")
                self.log(
                    f"BUY EXECUTED {order.executed.price:.2f}, SIZE {order.executed.size:.2f}, COST {order.executed.value:.2f}, COMMISSION {order.executed.comm:.2f}, MARGIN {margin_used:.2f}"
                )

                if order.info:
                    self.log(f"INFO {order.info['name']}")
                
                # 做多价格和金叉价格
                self.buy_price = order.executed.price
                # self.buy_create = order.created.price

            elif order.issell():
                self.ordermin = bt.num2time(self.datadatetime[0]).isoformat()
                self.log(f"SHORT SIG DETECTED @ {order.created.price:.2f}")
                self.log(
                    f"SELL EXECUTED {order.executed.price:.2f}, SIZE {order.executed.size:.2f}, COST {order.executed.value:.2f}, COMMISSION {order.executed.comm:.2f}, MARGIN {margin_used:.2f}"
                )

                if order.info:
                    self.log(f"INFO {order.info['name']}")

                # 做空价格和死叉价格
                self.sell_price = order.executed.price
                # self.sell_create = order.created.price

            self.bar_executed = len(self)

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER CANCELED/MARGIN/REJECTED **CODE**: {order.getstatusname()}")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"OPERATION PROFIT {trade.pnl:.2f}, NET PROFIT {trade.pnlcomm:.2f}")

    def start(self):
        pass
    
    def prenext(self):
        pass

    def next(self):
        # 当有正在处理中的订单
        if self.order:
            return

        if self.crossover == 1.0:  # 更新金叉价格
            self.buy_create = self.dataclose[0]
            self.log('GOLD CROSS UPDATED')
        if self.crossover == -1.0:  # 更新死叉价格
            self.sell_create = self.dataclose[0]
            self.log('DEAD CROSS UPDATED')

        # 策略逻辑
        if not self.position:
            # 同传统均线策略开仓逻辑
            if self.crossover == 1.0:  # 金叉
                self.order_target_percent(target=self.p.target_percent)

            elif self.crossover == -1.0:  # 死叉
                self.order_target_percent(target=-self.p.target_percent)
        else:
            # 改良平仓逻辑
            if self.position.size > 0:
                # if (self.dataclose[0] / self.buy_create - 1) < self.p.closeout_limit:
                if self.dataclose[0] < self.buy_create:
                    self.order_target_percent(target=0)
                    self.order.addinfo(name='CLOSE OUT BECAUSE OF STOP LIMIT')

            else:
                # if (self.dataclose[0] / self.sell_create - 1) > self.p.closeout_limit:
                if self.dataclose[0] > self.sell_create:
                    self.order_target_percent(target=0)
                    self.order.addinfo(name='CLOSE OUT BECAUSE OF STOP LIMIT')

    def stop(self):
        pass
    

if __name__ == "__main__":
    sys.stdout = Logger()

    data = MainContract(dataname=metavar.df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BetterMA)

    comminfo = FurCommInfo()
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.setcash(metavar.startcash)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="_TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="_Sharpe", timeframe=bt.TimeFrame.Minutes)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_Return", timeframe=bt.TimeFrame.Minutes)
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="_CalmarRatio")

    # backtesting
    init_msg = f"""
            回测对象: {metavar.contract}
            起始时间: {metavar._fromdate}
            终止时间: {metavar._todate}
            合约点值: {metavar.mult}
            最低保证金: {metavar.margin}
            开仓/平仓手续费: {0.23 / 10000:.4%}
            平今仓手续费: {3.45 / 10000:.4%}
            """

    print(init_msg)
    print(f"开始资金总额 {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strats = results[0]
    print(f"结束资金总额 {cerebro.broker.getvalue():.2f}")

    # cerebro.plot()

    # =========== for analysis.py ============ #
    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())
    rets.to_csv("timereturn.csv", index=True)
    # ======================================== #

    cumrets = emp.cum_returns(rets, starting_value=0)
    maxrets = cumrets.cummax()
    drawdown = (cumrets - maxrets) / maxrets
    max_drawdown = emp.max_drawdown(rets)
    calmar_ratio = emp.calmar_ratio(rets)

    num_years = metavar._todate.year - metavar._fromdate.year
    cumrets_final = emp.cum_returns_final(rets, starting_value=0)
    ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    yearly_trade_times = rets.shape[0] / num_years
    risk_free = 0.00
    sharpe = emp.sharpe_ratio(rets, risk_free=risk_free, annualization=yearly_trade_times)

    # 盈亏比
    mean_per_win = (rets[rets > 0]).mean()
    mean_per_loss = (rets[rets < 0]).mean()

    day_ret_max = pd.Series(strats.analyzers._TimeReturn.get_analysis()).describe()["max"]
    day_ret_min = pd.Series(strats.analyzers._TimeReturn.get_analysis()).describe()["min"]

    results_dict = {
        "年化夏普比率": sharpe,
        "最大回撤": max_drawdown, 
        "累计收益率": cumrets[-1],
         "年化收益率": ann_rets, 
        "收益回撤比": ann_rets / -max_drawdown,
        "单日最大收益": day_ret_max,
        "单日最大亏损": day_ret_min,
        "交易次数": int(rets.shape[0]),
        "获胜次数": sum(rets > 0),
        "胜率": sum(rets > 0) / rets.shape[0],
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    print(results_df)

