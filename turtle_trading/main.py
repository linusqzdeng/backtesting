# -*- coding: UTF-8 -*-
# created on 2022-01-06 21:49
# author@ qizhong deng
# tested by python 3.8.10

import pandas as pd
import numpy as np
import backtrader as bt
import empyrical as emp

import datetime
import os, sys


class Config:
    """For global variable"""

    # valid_contract = [16, 905, 300]  # IH, IC, IF
    valid_contracts = ['IF00', 'IH00', 'IC00']

    def __init__(self):
        # 回测区间
        self._fromdate = datetime.date(2010, 4, 16)
        self._todate = datetime.date(2021, 12, 21)

        # 标的合约
        self.contract = self.valid_contracts[0]

        # 数据处理
        # filepath = os.path.join(os.path.abspath('..'), 'index.csv')
        filepath = os.path.join(os.path.abspath('..'), 'data.csv')
        raw_df = pd.read_csv(filepath, index_col='TRADE_DT', parse_dates=True)
        self.df = self.create_df(raw_df, self.contract, 'daily')
        # self.firstday = self.get_firstday(self.df)  # keep records for the first day of the year
        self.first_days = self.get_firstday(self.df)

        # # 交易参数
        self.mult = self.set_mult()
        self.margin = self.set_margin()

        self.stamp_duty = 0.001
        self.startcash = 10_000_000
        self.yr_startcash = self.startcash  # startcash at the beginning of each year
        self.brokervalue = self.startcash  # keep track of the current value in broker account
        self.tt_margin_used = 0

        # 手续费
        self.commission = 0.23 / 10000

        # 通道参数 S1
        # self.longlen = 20
        # self.shortlen = 10

        # 通道参数 S2
        self.longlen = 50
        self.shortlen = 20

    def create_df(self, raw_df, contract, freq):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = raw_df[raw_df['S_INFO_CODE'] == contract]
        df.index = pd.to_datetime(df.index)

        adj_cols = ["S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE",]
        pre_cols = ["S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE"]

        for pre_col, adj_col in zip(pre_cols, adj_cols):
            df.loc[:, adj_col] = round(df.loc[:, pre_col] * df.loc[:, 'S_DQ_ADJFACTOR'], 1)

        df = df[adj_cols]

        methods = ['first', 'max', 'min', 'last']
        agg_dict = {col: method for col, method in zip(adj_cols, methods)}

        if freq == 'weekly':
            df = df.resample('W-MON').agg(agg_dict).dropna()
        elif freq == 'daily':
            df = df.resample('D').agg(agg_dict).dropna()

        return df

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

    def get_firstday(self, df):
        """Return a list contains the first days for each year"""
        all_dt = df.loc[self._fromdate.isoformat():self._todate.isoformat()].index.to_frame()
        yearly_dt = [g for _, g in all_dt.groupby(pd.Grouper(key='TRADE_DT', freq='Y'))]
        first_days = [df['TRADE_DT'].iloc[0].date().isoformat() for df in yearly_dt]

        return first_days


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


class StockIndex(bt.feeds.PandasData):
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


class TurtleSizer(bt.Sizer):
    """基于真实波动幅度头寸管理"""

    params = (
        ("mult", metavar.mult),
        ("theta", 0.01),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        abs_vol = self.strategy.atr[0] * self.p.mult  # N * CN
        unit = self.broker.get_value() * self.p.theta // abs_vol  # 1 unit
        
        return unit


class DonchianChannels(bt.Indicator):
    '''
    Params Note:
      - `lookback` (default: -1)
        If `-1`, the bars to consider will start 1 bar in the past and the
        current high/low may break through the channel.
        If `0`, the current prices will be considered for the Donchian
        Channel. This means that the price will **NEVER** break through the
        upper/lower channel bands.
    Reference: https://www.backtrader.com/recipes/indicators/donchian/donchian/
    '''

    alias = ('DCH', 'DonchianChannel',)
    lines = (
        'long_m', 'long_h', 'long_l',
        'short_m', 'short_h', 'short_l',
    ) 
    params = (
        ("longlen", metavar.longlen),
        ("shortlen", metavar.shortlen),
        ("lookback", -1),  # consider current bar or not
    )

    def __init__(self):
        hi, lo = self.data.high, self.data.low
        if self.p.lookback:  # move backwards as needed
            hi, lo = hi(self.p.lookback), lo(self.p.lookback)
        
        # long length
        self.l.long_h = bt.ind.Highest(hi, period=self.p.longlen)
        self.l.long_l = bt.ind.Lowest(lo, period=self.p.longlen)
        self.l.long_m = (self.l.long_h + self.l.long_l) / 2.0  # avg of the above

        # short length
        self.l.short_h = bt.ind.Highest(hi, period=self.p.shortlen)
        self.l.short_l = bt.ind.Lowest(lo, period=self.p.shortlen)
        self.l.short_m = (self.l.short_h + self.l.short_l) / 2.0  # avg of the above


class Turtle(bt.Strategy):
    """Turtle trading system"""

    params = (
        ("longlen", metavar.longlen),
        ("shortlen", metavar.shortlen),
        ("bigfloat", 6),
        ("drawback", 1),
        ("closeout", 2),
        ("addpos", 0.5),
        ("max_add", 3),
        ("theta", 0.01),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datadatetime = self.datas[0].datetime

        # 突破信号
        self.donchian = DonchianChannels()
        self.long_h = self.donchian.long_h
        self.long_l = self.donchian.long_l
        self.short_h = self.donchian.short_h
        self.short_l = self.donchian.short_l

        # 平仓
        self.stoplimit = 2

        # 头寸管理
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.longlen)
        self.addpos_count = 0
        self.tt_margin_used = 0
        self.ann_profit = 0

        self.curpos = 0
        self.buyprice = 0
        self.sellprice = 0
        self.order = None
        self.isclosed = False  # whether the order has been closed

    def log(self, txt, dt=None):
        dt = dt or self.datadatetime.date(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            # 保证金占用
            margin_used = (
                    order.executed.price
                    * abs(order.executed.size)
                    * metavar.mult
                    * metavar.margin
                    )
            self.tt_margin_used += margin_used
            metavar.tt_margin_used = self.tt_margin_used
            self.margin_pct = self.tt_margin_used / self.broker.get_value()

            if order.isbuy():
                self.log(
                        "LONG DETECTED @ {:.2f}, HH {:.2f}, LL {:.2f}, NET VALUE {:.2f}".format(
                    order.created.price,
                    self.long_h[-1],
                    self.long_l[-1],
                    self.broker.get_value(),
                    )
                )
                self.log(
                        "BUY EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, TT_MARGIN {:.2%}, CURPOS {}".format(
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    margin_used,
                    self.margin_pct,
                    self.position.size,
                    )
                )

                self.buyprice = order.executed.price

            elif order.issell():
                self.log(
                    "SHORT DETECTED @ {:.2f}, HH {:.2f}, LL {:.2f} NET VALUE {:.2f}".format(
                    order.created.price,
                    self.long_h[-1],
                    self.long_l[-1],
                    self.broker.get_value(),
                    )
                )
                self.log(
                        "SELL EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, TT_MARGIN {:.2%}, CURPOS {}".format(
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    margin_used,
                    self.margin_pct,
                    self.position.size,
                    )
                )

                self.sellprice = order.executed.price

            if order.info:
                self.log(f"**INFO** {order.info['name']}")

            self.bar_executed = len(self)

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER CANCELED/MARGIN/REJECTED **CODE** {order.getstatusname()}")

        if self.isclosed:
            self.tt_margin_used = 0
            self.isclosed = False

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
        bypass_conds = [self.order,
                len(self) < metavar.longlen]
        if any(bypass_conds):
            return

        # record the net value at the beginnign of each year
        metavar.brokervalue = self.broker.get_value()
        today = self.datadatetime.date(0).isoformat()
        if today in metavar.first_days:
            metavar.yr_startcash = self.broker.get_value()

        # 是否实行紧缩平仓
        # if self.is_trend():
            # self.stoplimit = self.p.drawback
        # else:
            # self.stoplimit = 2

        # 建仓信号
        if not self.position:
            # 通道突破
            if self.dataclose[0] > self.long_h[0]:
                self.order = self.buy()
                self.order.addinfo(name='LONG POS CREATED')
            elif self.dataclose[0] < self.long_l[0]:
                self.order = self.sell()
                self.order.addinfo(name='SHORT POS CREATED')

        else:
            # 平仓条件
            if self.position.size > 0:
                price_change = self.dataclose[0] - self.buyprice

                # 多头加仓
                if (price_change >= self.p.addpos * self.atr[0])\
                        and (self.addpos_count < self.p.max_add):
                    self.order = self.buy()
                    self.order.addinfo(name='ADD LONG POSITION')
                    self.addpos_count += 1

                # 多头平仓
                if (self.dataclose[0] < self.short_l)\
                        or (price_change <= -self.stoplimit * self.atr[0]):
                    self.order = self.close()
                    self.order.addinfo(name='CLOSE LONG DUE TO STOP LIMIT')
                    self.addpos_count = 0
                    self.isclosed = True

            else:
                price_change = self.dataclose[0] - self.sellprice

                # 空头加仓
                if (price_change <= -self.p.addpos * self.atr[0])\
                        and (self.addpos_count < self.p.max_add):
                    self.order = self.sell()
                    self.order.addinfo(name='ADD SHORT POSITION')
                    self.addpos_count += 1

                # 空头平仓
                if (self.dataclose[0] > self.short_h)\
                        or (price_change >= self.stoplimit * self.atr[0]):
                    self.order = self.close()
                    self.order.addinfo(name='CLOSE SHORT DUE TO STOP LIMIT')
                    self.addpos_count = 0
                    self.isclosed = True

    def stop(self):
        pass

    def get_size(self):
        """Calculate size based on ATR"""
        abs_vol = self.atr[0] * metavar.mult  # N * CN
        unit = self.broker.get_value() * self.p.theta // abs_vol  # 1 unit
        
        return unit


    def is_trend(self):
        """判断是否有大趋势，如有则实行紧缩平仓参数"""
        if self.position.size > 0:
            if self.datahigh[0] - self.buyprice > self.p.bigfloat:
                return True
        elif self.position.size < 0:
            if self.datahigh[0] - self.sellprice < -self.p.bigfloat:
                return True

        return False


class FurCommInfo(bt.CommInfoBase):
    """定义期货的交易手续费和佣金"""

    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("commission", metavar.commission),
        ("mult", metavar.mult),
        ("stamp_duty", metavar.stamp_duty),  # 印花税0.1%
        ("margin", metavar.margin),
        ("backtest_margin", 1.0),  # no leverage
    )

    def _getcommission(self, size, price, pseudoexec):
        """手续费=买卖手数*合约价格*手续费比例*合约乘数"""
        return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        ann_profit = abs(metavar.brokervalue / metavar.yr_startcash)
        max_margin = self.max_margin(ann_profit) - metavar.tt_margin_used
        margin = price * self.p.mult * self.p.margin

        # return min(max_margin, margin)
        return margin

    def max_margin(self, ann_profit):
        """根据净值范围调整最大保证金比例"""
        if ann_profit >= 1.1:
            max_pct = 0.3
        elif 1.1 > ann_profit >= 1:
            max_pct = 0.2
        elif 1 > ann_profit > 0.95:
            max_pct = 0.1
        else:
            max_pct = 0.05

        return max_pct * metavar.brokervalue

def run():
    sys.stdout = Logger()

    # initialisation
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Turtle)

    data = StockIndex(dataname=metavar.df)
    cerebro.adddata(data)

    cerebro.broker.setcash(metavar.startcash)
    comminfo = FurCommInfo()
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addsizer(TurtleSizer)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="_TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    cerebro.addanalyzer(bt.analyzers.Calmar, _name="_CalmarRatio")

    # Backtesting
    init_msg = f"""
            回测对象: {metavar.contract}
            起始时间: {metavar._fromdate}
            终止时间: {metavar._todate}
            合约点值: {metavar.mult}
            最低保证金: {metavar.margin}
            开仓/平仓手续费: {0.23 / 10000:.4%}
            """

    print(init_msg)
    print(f"开始资金总额 {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strats = results[0]
    print(f"结束资金总额 {cerebro.broker.getvalue():.2f}")

    # cerebro.plot(volume=False)

    # =========== for analysis.py ============ #
    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())
    rets.to_csv("timereturn.csv", index=True)
    # ======================================== #

    cumrets = emp.cum_returns(rets, starting_value=0)
    max_drawdown = emp.max_drawdown(rets)

    ann_rets = emp.annual_return(rets, period='daily')
    calmar_ratio = ann_rets / -max_drawdown
    sharpe = emp.sharpe_ratio(rets, risk_free=0, period='daily')

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
        "交易次数": sum(rets != 0),
        "获胜次数": sum(rets > 0),
        "胜率": sum(rets > 0) / sum(rets != 0),
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    print(results_df)


if __name__ == "__main__":
    run()


