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
        self._todate = datetime.date(2020, 4, 15)

        # 标的合约
        self.contract = self.valid_contracts[0]
        self.freq = 'daily'  # daily or weekly

        # 数据处理
        # filepath = os.path.join(os.path.abspath('..'), 'index.csv')
        filepath = os.path.join(os.path.abspath('..'), 'data.csv')
        raw_df = pd.read_csv(filepath, index_col='TRADE_DT')
        self.df = self.create_df(raw_df, self.contract, self.freq)

        # # 交易参数
        self.mult = self.set_mult()
        self.margin = self.set_margin()

        self.stamp_duty = 0.001
        self.startcash = 10_000_000

        # 手续费
        self.commission = 0.23 / 10000

        # 通道参数 S1
        self.longlen = 20
        self.shortlen = 10

        # 改良止损参数
        self.bigfloat = 4
        self.drawback = 1

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

        agg_dict = {'S_DQ_ADJOPEN': 'first',
          'S_DQ_ADJHIGH': 'max',
          'S_DQ_ADJLOW': 'min',
          'S_DQ_ADJCLOSE': 'last'}

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
        ("backtest_margin", 1.0),  # no leverage for now
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        手续费=买卖手数*合约价格*手续费比例*合约乘数
        """
        if size > 0:
            return abs(size) * price * self.p.commission * self.p.mult
        else:  # 卖出时考虑印花税
            return abs(size) * price * (self.p.commission + self.p.stamp_duty) * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.backtest_margin


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

    plotinfo = dict(subplot=False)  # plot along with data
    plotlines = dict(
        long_m=dict(ls='--'),  # dashed line
        long_h=dict(_samecolor=True),  # use same color as prev line (dcm)
        long_l=dict(_samecolor=True),  # use same color as prev line (dch)
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
        ("bigfloat", metavar.bigfloat),
        ("drawback", metavar.drawback),
        ("addpos_percent", 0.02),
        ("addpos_max", 4),
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

        # 头寸管理
        self.atr = bt.ind.ATR(data, period=self.p.longlen)
        self.addpos = False  # 加仓
        self.addpos_count = 0
        self.tight_closeout = False  # 紧缩型止损

        self.buyprice = None
        self.sellprice = None
        self.order = None

        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

    def log(self, txt, dt=None):
        dt = dt or self.datadatetime.date(0)
        print(f"{dt} - {txt}")

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"ORDER SUBMITTED/ACCEPTED **CODE** {order.getstatusname()}")
            return

        # 处理已完成订单
        if order.status == order.Completed:
            # 保证金占用
            margin_used = order.executed.price * abs(order.executed.size) * metavar.mult * metavar.margin

            if order.isbuy():
                self.log(
                    "BUY EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}".format(
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    margin_used,
                    )
                )

                self.buyprice = order.executed.price

            elif order.issell():
                self.log(
                    "SELL EXECUTED @ {:.2f}, SIZE {:.2f}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}".format(
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    margin_used,
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

    def prenext(self):
        pass

    def next(self):
        if self.order:
            return

        self.addpos = False
        normal = 2 * self.atr[0]  # 普通平仓阈值
        tight = self.p.drawback * self.atr[0]  # 大趋势紧缩平仓阈值

        threhold = normal
        if self.tight_closeout:
            threshold = tight
        
        # 建仓信号
        if not self.position:
            if self.dataclose[0] > self.long_h[0]:
                self.order = self.buy()
            elif self.dataclose[0] < self.long_l[0]:
                self.order = self.sell()

        else:
            # 平仓条件
            if self.position.size > 0:
                price_change = self.dataclose[0] - self.buyprice
                # 多头加仓
                if (price_change / self.buyprice > self.p.addpos_percent)\
                        and (self.addpos_count < self.p.addpos_max):
                    self.addpos = True
                    self.order = self.buy()
                    self.order.addinfo(name='ADD LONG POSITION')

                # 多头平仓
                if price_change < -threhold:
                    self.order = self.close()

                    if self.tight_closeout:
                        self.order.addinfo(name='CLOSE OUT DUE TO STOP LIMIT (TIGHT)')
                        self.tight_closeout = False

                    self.order.addinfo(name='CLOSE OUT DUE TO STOP LIMIT')
                    self.addpos_count = 0

                # 触发大趋势止损
                if price_change > self.p.bigfloat * self.atr[0]:
                    self.tight_closeout = True

            else:
                price_change = self.dataclose[0] - self.sellprice
                # 空头加仓
                if (price_change / self.sellprice < -self.p.addpos_percent)\
                        and (self.addpos_count < self.p.addpos_max):
                    self.addpos = True
                    self.order = self.sell()
                    self.order.addinfo(name='ADD SHORT POSITION')

                # 空头平仓
                if price_change > threhold:
                    self.order = self.close()
                    
                    if self.tight_closeout:
                        self.order.addinfo(name='CLOSE OUT DUE TO STOP LIMIT (TIGHT)')
                        self.tight_closeout = False

                    self.order.addinfo(name='CLOSE OUT DUE TO STOP LIMIT')
                    self.addpos_count = 0

                # 触发大趋势止损
                if price_change < -self.p.bigfloat * self.atr[0]:
                    self.tight_closeout = True

            # 记录加仓次数
            if self.addpos:
                self.addpos_count += 1


    def stop(self):
        pass


if __name__ == "__main__":
    sys.stdout = Logger()

    # Initialisation
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
    maxrets = cumrets.cummax()
    drawdown = (cumrets - maxrets) / maxrets
    max_drawdown = emp.max_drawdown(rets)
    calmar_ratio = emp.calmar_ratio(rets)

    num_years = metavar._todate.year - metavar._fromdate.year
    # ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    ann_rets = emp.annual_return(rets, period='daily')
    yearly_trade_times = rets.shape[0] / num_years
    sharpe = emp.sharpe_ratio(rets, risk_free=0, period='daily')

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
        "交易次数": sum(rets != 0),
        "获胜次数": sum(rets > 0),
        "胜率": sum(rets > 0) / sum(rets != 0),
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    print(results_df)


