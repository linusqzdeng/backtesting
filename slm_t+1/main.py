# -*- coding: UTF-8 -*-
# Created on: 2022-1-10 10:44
# Tested by: python 3.8.10
# Author: @Qizhong Deng

import pandas as pd
import numpy as np
import backtrader as bt
import empyrical as emp

import os, sys
import datetime


class Config:

    # 上证综指
    # train_data = pd.read_csv(
            # os.path.abspath('./SSE_index.csv'),
            # delimiter=';',
            # index_col='date',
            # parse_dates=True
            # )

    # 股指期货主力合约
    raw_data = pd.read_csv(
            os.path.abspath('../data.csv'),
            index_col='TRADE_DT',
            parse_dates=True
            )

    contract = 'IF00'
    train_fromdate = datetime.date(2010, 4, 16)
    train_todate = datetime.date(2019, 12, 31)
    fromdate = datetime.date(2020, 1, 1)
    todate = datetime.date(2021, 12, 21)

    # lookback_period = 6
    startcash = 10_000_000

    mult = 300
    margin = 0.12
    commission = 0.23 / 10000
    stamp_duty = 0.001


    def __init__(self):
        self.lookback_period = 6
        # 历史数据始料库 1990/12/20 - 2015/9/11
        # self.train_df = self.create_pattern(
                # self.train_data, ref_col='close',
                # fromdate=self.train_fromdate, todate=None
                # )

        # 采用IF00作为历史数据库
        self.resampled_data = self.resample_df(
                self.cal_adjprices(self.raw_data, self.contract),
                'daily'
                )
        self.resampled_data = self.create_pattern(self.resampled_data, ref_col='S_DQ_ADJCLOSE')
        self.train_df = self.resampled_data.loc[self.train_fromdate:self.train_todate]
        self.test_df = self.resampled_data.loc[self.fromdate:self.todate]

        self.train_pat = self.get_patterns(self.train_df, self.lookback_period)

    def create_pattern(self, df: pd.DataFrame, ref_col: str, fromdate=None, todate=None):
        """Numerical pattern 1 for price goes down, 2 for price goes up"""

        fromdate = fromdate or datetime.date(2010, 4, 16) 
        todate =todate or datetime.date(2021, 12, 21) 

        df = df.loc[fromdate:todate]
        df['ret'] = df[ref_col].pct_change()
        df['pat'] = df['ret'].apply(lambda x: 1 if x <= 0 else 2) 
        df = df.dropna()

        return df

    def cal_adjprices(self, df: pd.DataFrame, contract: str):
        """Calculate the adjusted prices for specific contract"""

        df = df[df["S_INFO_CODE"] == contract]
        adj_cols = ["S_DQ_ADJOPEN","S_DQ_ADJHIGH","S_DQ_ADJLOW","S_DQ_ADJCLOSE",]
        pre_cols = ["S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE"]

        for pre_col, adj_col in zip(pre_cols, adj_cols):
            df.loc[:, adj_col] = round(
                df.loc[:, pre_col] * df.loc[:, "S_DQ_ADJFACTOR"], 1
            )

        return df[adj_cols]

    def resample_df(self, df: pd.DataFrame, freq: str):
        """Resample the dataframe to specific frequency"""

        adj_cols = ["S_DQ_ADJOPEN","S_DQ_ADJHIGH","S_DQ_ADJLOW","S_DQ_ADJCLOSE",]
        methods = ['first', 'max', 'min', 'last']
        agg_dict = {col:method for col, method in zip(adj_cols, methods)}

        if freq == "weekly":
            df = df.resample("W-MON").agg(agg_dict).dropna()
        elif freq == "daily":
            df = df.resample("D").agg(agg_dict).dropna()

        return df

    def get_patterns(self, df: pd.DataFrame, n: int):
        """
        Return a pd.Series that contains all
        patterns with specific n days look back
        """

        pat_list = [df['pat'].iloc[i:i + n].values for i in range(len(df) - n + 1)]
        pat_list = [np.array2string(pat, separator=',') for pat in pat_list]
        pat = pd.Series(pat_list).value_counts()

        return pat

metavar = Config()


class Logger:
    """将打印出的交易单保存为log文件"""

    def __init__(self, filename=f"./logs/{metavar.contract}.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass


class DataInput(bt.feeds.PandasData):
    lines = ("pattern",)  # extending the datafeed
    params = (
        ("nullvalue", np.nan),
        ("fromdate", metavar.fromdate),
        ("todate", metavar.todate),
        ("datetime", None),  # index of the dataframe
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ("pattern", 5),
        ("volume", -1),
        ("openinterest", -1),
    )


class MySizer(bt.Sizer):
    """基于真实波动幅度头寸管理"""

    params = (
        ("period", 20),
        ("mult", metavar.mult),
        ("theta", 0.01),
        ("addpos_unit", 0.5),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        abs_vol = self.strategy.atr[0] * self.p.mult  # N * CN
        if self.strategy.addpos:
            abs_vol *= self.p.addpos_unit  # 0.5N * CN
        unit = self.broker.get_value() * self.p.theta // abs_vol  # 1 unit

        return unit


class MyCommInfo(bt.CommInfoBase):
    
    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("commission", metavar.commission),
        ("mult", metavar.mult),
        ("stamp_duty", metavar.stamp_duty),
        ("margin", metavar.margin),
        ("backtest_margin", 1.0),  # no leverage for now
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        手续费=买卖手数*合约价格*手续费比例*合约乘数
        """
        if size > 0:
            return abs(size) * price * self.p.commission * self.p.mult
        else:  # 卖出时考虑印花税
            return abs(size) * price * self.p.commission * self.p.mult
            # return (
                # abs(size)
                # * price
                # * (self.p.commission + self.p.stamp_duty)
                # * self.p.mult
            # )

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.backtest_margin


class MyStrats(bt.Strategy):
    
    params = (
        ("printout", True),
        ("target_percent", 0.45),
        ("stop_limit", 0.01),
        ("theta", 0.01),
        ("mult", metavar.mult),
    )

    def __init__(self):
        self.pat = []

        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime
        self.datapat = self.datas[0].pattern

        self.buyprice = None
        self.sellprice = None
        self.up_prob = None
        self.downprob = None

        self.order = None
        
        # for sizer
        self.atr = bt.ind.ATR(period=metavar.lookback_period)

    def log(self, txt, dt=None):
        dt = dt or self.datadatetime.date(0)
        if self.p.printout:
            print(f"{dt} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            # 保证金占用
            self.margin_used = (
                order.executed.price
                * abs(order.executed.size)
                * metavar.mult
                * metavar.margin
            )

            if order.isbuy():
                self.log(
                        "BUY EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, UP {:.2%} DOWN {:.2%}".format(
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        self.margin_used,
                        self.up_prob,
                        self.down_prob,
                    )
                )

                self.buyprice = order.executed.price

            elif order.issell():
                self.log(
                        "SELL EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f} UP {:.2%} DOWN {:.2%}".format(
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        self.margin_used,
                        self.up_prob,
                        self.down_prob,
                    )
                )

                self.sellprice = order.executed.price

            if order.info:
                self.log(f"**INFO** {order.info['name']}")

            self.bar_executed = len(self)

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER CANCELED/MARGIN/REJECTED **CODE** {order.getstatusname()}")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"OPERATION PROFIT {trade.pnl:.2f}, NET PROFIT {trade.pnlcomm:.2f}")

    def start(self):
        pass

    def next(self):
        bypass_conds = [
                self.order,
                len(self) < metavar.lookback_period,
                ]
        if any(bypass_conds):
            return
        
        self.pat = np.array(
                self.datapat.get(ago=0, size=metavar.lookback_period - 1),
                dtype=int
                )  # 过去n-1天的符号

        up_pat = np.array2string(np.append(self.pat, 2), separator=',')
        down_pat = np.array2string(np.append(self.pat, 1), separator=',')

        self.up_prob = metavar.train_pat[up_pat] / len(metavar.train_df)
        self.down_prob = metavar.train_pat[down_pat] / len(metavar.train_df)

        upsig = self.up_prob >= self.down_prob
        downsig = self.up_prob < self.down_prob

        if not self.position:
            # 未持仓，按信号建仓
            if upsig:
                self.order = self.order_target_percent(target=self.p.target_percent)
            elif downsig:
                self.order = self.order_target_percent(target=-self.p.target_percent)

        else:
            # 已持仓信号方向不一致，平仓后建仓
            pct_change = self.dataclose[0] / self.dataclose[-1] - 1
            if self.position.size > 0:
                if pct_change <= -self.p.stop_limit:
                    self.order = self.order_target_percent(target=0)
                    self.order.addinfo(name='CLOSE LONG POS')

                    if upsig:
                        self.order = self.order_target_percent(target=self.p.target_percent)
                    elif downsig:
                        self.order = self.order_target_percent(target=-self.p.target_percent)

                else:
                    # 平仓后开空
                    if downsig:
                        self.order = self.order_target_percent(target=-2 * self.p.target_percent)

            else:
                if (self.up_prob > self.down_prob) or (pct_change >= self.p.stop_limit):
                    self.order = self.order_target_percent(target=0)
                    self.order.addinfo(name='CLOSE SHORT POS')

                    self.order = self.order_target_percent(target=self.p.target_percent)

    def stop(self):
        pass

    def cal_size(self):
        """Calculate the size to order in terms of the ATR"""
        abs_vol = self.atr[0] * self.p.mult  # N * CN
        size = self.broker.get_value() * self.p.theta // abs_vol  # 1 unit
        
        return size


def normal_analysis(strats):
    # =========== for analysis.py ============ #
    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())
    rets.to_csv("timereturn.csv", index=True)
    # ======================================== #
    
    # 收益最大回撤
    cumrets = emp.cum_returns(rets, starting_value=0)
    max_drawdown = emp.max_drawdown(rets)

    # 夏普比率
    num_years = metavar.todate.year - metavar.fromdate.year
    ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    risk_free = 0
    sharpe = emp.sharpe_ratio(rets, risk_free=risk_free, period='daily')

    # 盈亏比
    mean_per_win = (rets[rets > 0]).mean()
    mean_per_loss = (rets[rets < 0]).mean()

    # 单次交易最大最小值
    day_ret_max = rets.max()
    day_ret_min = rets.min()

    results_dict = {
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "累计收益率": train_df[-1],
        "年化收益率": ann_rets,
        "收益回撤比": ann_rets / -max_drawdown,
        "单日最大收益": day_ret_max,
        "单日最大亏损": day_ret_min,
        "交易次数": round(len(rets), 0),
        "获胜次数": round(sum(rets > 0), 0),
        "胜率": sum(rets > 0) / len(rets),
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    results_df.to_clipboard()
    print(results_df)


def opt_analysis(results):

    def get_analysis(result):
        analysers = {}
        analysers['period'] = result.params.lookback_period

        # 返回参数
        rets = pd.Series(result.analyzers._TimeReturn.get_analysis()).values
        
        # 夏普比率
        cumrets = emp.cum_returns(rets, starting_value=0)
        num_years = metavar.todate.year - metavar.fromdate.year
        yearly_trade_times = len(rets) / num_years
        ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
        risk_free = 0
        sharpe = emp.sharpe_ratio(rets, risk_free=risk_free, annualization=yearly_trade_times)
        
        analysers['return'] = ann_rets
        analysers['max_drawdown'] = emp.max_drawdown(rets)
        analysers['calmar_ratio'] = emp.calmar_ratio(rets)
        analysers['sharpe'] = sharpe

        return analysers

    opt_results = [get_analysis(i[0]) for i in results]
    opt_df = pd.DataFrame(opt_results)
    opt_df.to_csv('opt_results.csv')

    print(opt_df)


def run():
    sys.stdout = Logger()

    # Initialisation
    # normal init
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MyStrats)

    # optimisation init
    # cerebro = bt.Cerebro(optdatas=True, optreturn=True)
    # cerebro.optstrategy(MyStrats, lookback_period=range(3, 11))

    data = DataInput(dataname=metavar.test_df)
    cerebro.adddata(data)

    comminfo = MyCommInfo()
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.setcash(metavar.startcash)
    # cerebro.addsizer(MySizer)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="_TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="_CalmarRatio")

    # Observers
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.TimeReturn)

    # Backtesting
    print(f"开始资金总额 {cerebro.broker.getvalue():.2f}")
    # normal run
    results = cerebro.run()
    # opt run
    # results = cerebro.run(maxcpus=1)
    print(f"结束资金总额 {cerebro.broker.getvalue():.2f}")
    
    # normal analysis
    strats = results[0]
    normal_analysis(strats)

    # opt analysis
    # opt_analysis(results)


if __name__ == "__main__":
    run()
    # print(metavar.train_pat)









