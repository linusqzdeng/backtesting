# -*- coding: UTF-8 -*-
# created on 2022-01-06 21:49
# author@ qizhong deng
# tested by python 3.8.10

import pandas as pd
import numpy as np
import backtrader as bt
import empyrical as emp
import pyfolio as pyf

import datetime
import os, sys
import warnings
from collections import defaultdict
from comminfo import IFCommInfo, IHCommInfo, ICCommInfo

warnings.filterwarnings("ignore")


class Config:
    """For global variable"""

    valid_contracts = ["IF00", "IH00", "IC00"]
    contract = valid_contracts[0]

    # 回测区间
    fromdate = datetime.date(2015, 4, 16)
    todate = datetime.date(2021, 12, 31)
    filepath = os.path.abspath("../data.csv")
    data = pd.read_csv(filepath, index_col="TRADE_DT", parse_dates=True)

    # 交易参数
    stamp_duty = 0.001
    commission = 0.23 / 10000
    mult = 300
    margin = 0.12
    startcash = 10_000_000 * 3

    def __init__(self):
        # 数据处理
        self.if00 = self.create_df(self.data, self.valid_contracts[0], 'daily')
        self.ih00 = self.create_df(self.data, self.valid_contracts[1], 'daily')
        self.ic00 = self.create_df(self.data, self.valid_contracts[2], 'daily')
        self.first_days = self.get_firstday(self.if00)  # keep records for the first day of the year

    def create_df(self, data, contract, freq):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = data[data["S_INFO_CODE"] == contract]
        df.index = pd.to_datetime(df.index)

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

        methods = ["first", "max", "min", "last"]
        agg_dict = {col: method for col, method in zip(adj_cols, methods)}

        if freq == "weekly":
            df = df.resample("W-MON").agg(agg_dict).dropna()
        elif freq == "daily":
            df = df.resample("D").agg(agg_dict).dropna()

        return df

    def set_margin(self):
        # 保证金设置
        if self.contract in ["IH00", "IC00"]:
            margin = 0.10
        elif self.contract == "IF00":
            margin = 0.12
        else:
            raise ValueError("Invalid contract name")

        return margin

    def set_mult(self):
        # 乘数设置
        if self.contract in ["IF00", "IH00"]:
            mult = 300.0
        elif self.contract == "IC00":
            mult = 200.0
        else:
            raise ValueError("Invalid contract name")

        return mult

    def get_firstday(self, df):
        """Return a list contains the first days for each year"""
        all_dt = df.loc[self.fromdate.isoformat() : self.todate.isoformat()].index.to_frame()
        yearly_dt = [g for _, g in all_dt.groupby(pd.Grouper(key="TRADE_DT", freq="Y"))]
        first_days = [df["TRADE_DT"].iloc[0].date().isoformat() for df in yearly_dt]

        return first_days


# global variable
metavar = Config()


class Logger:
    """将打印出的交易单保存为log文件"""

    savepath = f"./logs/multi_contracts.log"

    def __init__(self, filename=savepath):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        pass


class DataInput(bt.feeds.PandasData):
    params = (
        ("nullvalue", np.nan),
        ("fromdate", metavar.fromdate),
        ("todate", metavar.todate),
        ("datetime", None),  # index of the dataframe
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ("volume", -1),
        ("openinterest", -1),
    )


class DonchianChannels(bt.Indicator):
    '''
    Params Note:
      - `lookback` (default: -1)
        If `-1`, the bars to consider will start 1 bar in the past and the
        current high/low may break through the channel.
        If `0`, the current prices will be considered for the Donchian
        Channel. This means that the price will **NEVER** break through the
        upper/lower channel bands.
    Refers to: https://www.backtrader.com/recipes/indicators/donchian/donchian/
    '''

    alias = ('DCH', 'DonchianChannel',)
    lines = ('dcm', 'dch', 'dcl',)  # dc middle, dc high, dc low
    params = (
        ('period', 20),  # default value (could be modified)
        ('lookback', -1),  # consider current bar or not
    )

    def __init__(self):
        hi, lo = self.data.high, self.data.low
        if self.p.lookback:  # move backwards as needed
            hi, lo = hi(self.p.lookback), lo(self.p.lookback)

        self.l.dch = bt.ind.Highest(hi, period=self.p.period)
        self.l.dcl = bt.ind.Lowest(lo, period=self.p.period)
        self.l.dcm = (self.l.dch + self.l.dcl) / 2.0  # avg of the above


class TurtleSizer(bt.Sizer):

    params = (
        ("theta", 0.01),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        # 最大保证金约束
        available_margin = self.strategy.available_margin[data]
        price = self.strategy.dataclose[data][0]
        max_size = available_margin // (price * comminfo.p.mult * comminfo.p.margin)

        # 真实波动率约束
        atr = self.strategy.atr[data][0]
        abs_vol = atr * comminfo.p.mult  # N * CN
        current_value = self.broker.get_value([data]) + self.strategy.start_value[data]
        atr_size = (current_value * self.p.theta) // abs_vol  # 1 unit 

        return min(max_size, atr_size)


class Turtle(bt.Strategy):
    """Turtle trading system"""

    params = (
        ("s1_longperiod", 20),
        ("s1_shortperiod", 10),
        ("s2_longperiod", 55),
        ("s2_shortperiod", 20),
        ("is_s1", True),
        ("bigfloat", 6),
        ("drawback", 1),
        ("closeout", 2),
        ("max_pos", 6),
        ("theta", 0.01),
    )

    def __init__(self):
        # s1 or s2
        if self.p.is_s1:
            longperiod = self.p.s1_longperiod
            shortperiod = self.p.s1_shortperiod
        else:
            longperiod = self.p.s2_longperiod
            shortperiod = self.p.s2_shortperiod

        # if True, ignore the next 20days breakthrough
        # look for a 55days breakthrough
        self.ignore = False

        self.donchian_long = {}
        self.donchian_short = {}
        self.dataopen = defaultdict(lambda: 0)
        self.datahigh = defaultdict(lambda: 0)
        self.datalow = defaultdict(lambda: 0)
        self.dataclose = defaultdict(lambda: 0)
        self.longsig = defaultdict(lambda: 0)
        self.shortsig = defaultdict(lambda: 0)
        self.longexit = defaultdict(lambda: 0)
        self.shortexit = defaultdict(lambda: 0)
        self.atr = defaultdict(lambda: 0)
        for d in self.datas:
            # 突破通道
            long_channel = DonchianChannels(d, period=longperiod)
            short_channel = DonchianChannels(d, period=shortperiod)
            self.donchian_long[d], self.donchian_short[d] = {}, {}
            self.donchian_long[d]['hh'] = long_channel.dch  # highest high
            self.donchian_long[d]['ll'] = long_channel.dcl  # lowest low
            self.donchian_short[d]['hh'] = short_channel.dch
            self.donchian_short[d]['ll'] = short_channel.dcl

            # ohlc
            self.dataopen[d] = d.open
            self.datahigh[d] = d.high
            self.datalow[d] = d.low
            self.dataclose[d] = d.close

            # 交易信号
            self.longsig[d] = bt.ind.CrossUp(self.dataclose[d](0), self.donchian_long[d]['hh'])
            self.shortsig[d] = bt.ind.CrossDown(self.dataclose[d](0), self.donchian_long[d]['ll'])
            self.longexit[d] = bt.ind.CrossDown(self.dataclose[d](0), self.donchian_short[d]['ll'])
            self.shortexit[d] = bt.ind.CrossUp(self.dataclose[d](0), self.donchian_short[d]['hh'])

            # atr
            self.atr[d] = bt.ind.ATR(d, period=20)


        # 头寸管理
        self.pos_count = defaultdict(lambda: 0)
        self.margin_used = defaultdict(lambda: 0)
        self.max_margin = defaultdict(lambda: metavar.startcash * 0.4)
        self.start_value = defaultdict(lambda: metavar.startcash / 3)  # 10m each
        self.available_margin = defaultdict(lambda: 0)

        # 订单管理
        self.buyprice = defaultdict(lambda: None)
        self.sellprice = defaultdict(lambda: None)
        self.order = defaultdict(lambda: None)

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            # Basic info of current order
            dt, dn = self.datetime.date(), order.data._name
            mult = order.comminfo.p.mult
            margin = order.comminfo.p.margin

            # 保证金占用
            margin_used = order.executed.price * abs(order.executed.size) * mult * margin
            self.margin_used[order.data] += margin_used

            if order.isbuy():
                self.log(
                    "LONG CREATED FOR {}, @ {:.2f}, EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, CURPOS {}".format(
                        dn,
                        order.created.price,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        margin_used,
                        self.getposition(order.data).size,
                    )
                )

                self.buyprice[order.data] = order.executed.price

            elif order.issell():
                self.log(
                    "SHORT FOR {}, CREATED @ {:.2f}, EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, CURPOS {}".format(
                        dn,
                        order.created.price,
                        order.executed.price,
                        order.executed.size,
                        order.executed.value,
                        order.executed.comm,
                        margin_used,
                        self.getposition(order.data).size,
                    )
                )

                self.sellprice[order.data] = order.executed.price

            if order.info:
                self.log(f"**INFO** {order.info['name']}")

            self.bar_executed = len(self)

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER CANCELED/MARGIN/REJECTED **CODE** {order.getstatusname()}")

        self.order[order.data] = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f"OPERATION FOR {trade.data._name} PROFIT {trade.pnl:.2f}, NET PROFIT {trade.pnlcomm:.2f}")

    def start(self):
        pass

    def prenext(self):
        pass

    def next(self):
        # Loop through each asset
        for i, d in enumerate(self.datas):
            # 计算年内净值变动
            dt, dn = self.datetime.date().isoformat(), d._name
            self.reset_margin_and_startvalue(d, dt)
            self.available_margin[d] = self.cal_available_margin(d)
            pos = self.getposition(d).size

            if not pos and not self.order.get(d, None):
                if self.longsig[d]:
                    self.order[d] = self.buy(data=d)
                    self.pos_count[d] += 1
                elif self.shortsig[d]:
                    self.order[d] = self.sell(data=d)
                    self.pos_count[d] += 1
            else:
                # 大趋势判断: 若存在正向大趋势，执行紧缩性平仓
                if self.is_trend(d):
                    stop_limit = self.p.drawback
                else:
                    stop_limit = self.p.closeout

                if pos > 0:
                    price_change = self.dataclose[d][0] - self.buyprice[d]

                    # 多头加仓
                    if (price_change >= 0.5 * self.atr[d][0]) and (sum(self.pos_count.values()) < self.p.max_pos):
                        self.order[d] = self.buy(data=d)

                        if self.order[d] is not None:  # size != 0
                            self.pos_count[d] += 1
                            self.order[d].addinfo(name=f'LONG POSITION ADDED FOR {dn}, TOTAL POS ADD {self.pos_count[d]}')

                    # 多头平仓
                    if self.longexit[d] > 0:  # 赢利性退场
                        self.order[d] = self.close(data=d)
                        self.order[d].addinfo(name=f'WINNER CLOSE FOR {dn}')
                        self.pos_count[d] = 0
                        self.margin_used[d] = 0
                    elif price_change <= -stop_limit * self.atr[d][0]:  # 亏损性退场
                        self.order[d] = self.close(data=d)
                        self.order[d].addinfo(name=f'LOSER CLOSE FOR {dn}')
                        self.pos_count[d] = 0
                        self.margin_used[d] = 0

                else:
                    price_change = self.dataclose[d][0] - self.sellprice[d]

                    # 空头加仓
                    if (price_change <= -0.5 * self.atr[d][0]) and (sum(self.pos_count.values()) < self.p.max_pos):
                        self.order[d] = self.sell(data=d)
                        if self.order[d] is not None:  # size != 0
                            self.pos_count[d] += 1
                            self.order[d].addinfo(name=f'SHORT POSITION ADDED FOR {dn}, TOTAL POS ADD {self.pos_count[d]}')

                    # 空头平仓
                    if self.shortexit[d] > 0:  # 赢利性退场
                        self.order[d] = self.close(data=d)
                        self.order[d].addinfo(name=f'WINNER CLOSE FOR {dn}')
                        self.pos_count[d] = 0
                        self.margin_used[d] = 0
                    elif price_change >= stop_limit * self.atr[d][0]:  # 亏损性退场
                        self.order[d] = self.close(data=d)
                        self.order[d].addinfo(name=f'LOSER CLOSE FOR {dn}')
                        self.pos_count[d] = 0
                        self.margin_used[d] = 0

    def stop(self):
        pass

    def cal_available_margin(self, data):
        """Return the maximum margin that can be used to trade"""
        current_value = self.broker.get_value([data]) + self.start_value[data]
        ann_nav = current_value / self.start_value[data]

        if ann_nav >= 1.1:
            max_pct = 0.50
        elif 1.1 > ann_nav >= 1:
            max_pct = 0.40
        elif 1 > ann_nav >= 0.95:
            max_pct = 0.30
        else:
            max_pct = 0.10

        self.max_margin[data] = max_pct * (self.broker.get_value([data]) + self.start_value[data]) 
        available_margin = self.max_margin[data] - self.margin_used[data]

        return available_margin
    
    def reset_margin_and_startvalue(self, data, date):
        """Record the margin and start value of each asset at the year start"""
        if date in metavar.first_days:
            self.margin_used[data] = 0
            self.start_value[data] += self.broker.get_value([data])
    
    def is_trend(self, data):
        """Return True if the current movement of price strike the 'bigfloat' trend"""
        high = self.datahigh[data]
        buyprice = self.buyprice[data]
        sellprice = self.sellprice[data]
        atr = self.atr[data]

        uptrend = high[-1] - buyprice > self.p.bigfloat * atr[0] if buyprice else False
        downtrend = sellprice - high[-1] < -self.p.bigfloat * atr[0] if sellprice else False

        return any([uptrend, downtrend])


def run():
    sys.stdout = Logger()

    # initialisation
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Turtle)

    # Add datafeeds and comminfos
    datalist = [
        (metavar.if00, IFCommInfo(), 'if00'),
        (metavar.ih00, IHCommInfo(), 'ih00'),
        (metavar.ic00, ICCommInfo(), 'ic00')
    ]
    for df, comminfo, name in datalist:
        data = DataInput(dataname=df)
        cerebro.adddata(data, name=name)
        cerebro.broker.addcommissioninfo(comminfo, name=name)

    cerebro.broker.setcash(metavar.startcash)
    cerebro.addsizer(TurtleSizer)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    # Backtesting
    print(f"开始资金总额 {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strats = results[0]
    print(f"结束资金总额 {cerebro.broker.getvalue():.2f}")

    # cerebro.plot(volume=False)

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
    max_drawdown = emp.max_drawdown(rets)

    ann_rets = emp.annual_return(rets, period="daily")
    calmar_ratio = ann_rets / -max_drawdown
    sharpe = emp.sharpe_ratio(rets, risk_free=0, period="daily")

    # 盈亏比
    mean_per_win = (rets[rets > 0]).mean()
    mean_per_loss = (rets[rets < 0]).mean()

    day_ret_max = rets.max()
    day_ret_min = rets.min()

    results_dict = {
        "年化夏普比率": sharpe,
        "最大回撤": max_drawdown,
        "累计收益率": cumrets[-1],
        "年化收益率": ann_rets,
        "收益回撤比": calmar_ratio,
        "单日最大收益": day_ret_max,
        "单日最大亏损": day_ret_min,
        "交易次数": len(transactions),
        "获胜次数": sum(rets > 0),
        "胜率": sum(rets > 0) / sum(rets != 0),
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }
    results_series = pd.Series(results_dict)
    print(pd.Series(results_series))
    print(perf_df)


if __name__ == "__main__":
    run()
