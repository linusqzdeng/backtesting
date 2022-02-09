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

warnings.filterwarnings("ignore")


class Config:
    """For global variable"""

    valid_contracts = ["IF00", "IH00", "IC00"]
    contract = valid_contracts[0]

    # 回测区间
    fromdate = datetime.date(2010, 1, 1)
    todate = datetime.date(2021, 12, 31)
    filepath = os.path.abspath("../data.csv")
    data = pd.read_csv(filepath, index_col="TRADE_DT", parse_dates=True)

    # 交易参数
    stamp_duty = 0.001
    commission = 0.23 / 10000
    startcash = 10_000_000

    def __init__(self):
        # 数据处理
        self.df = self.create_df(self.data, self.contract, "daily")
        self.first_days = self.get_firstday(self.df)  # keep records for the first day of the year

        # 根据合约更改保证金和乘数
        self.mult = self.set_mult()
        self.margin = self.set_margin()

    def create_df(self, raw_df, contract, freq):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = raw_df[raw_df["S_INFO_CODE"] == contract]
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

    savepath = f"./logs/{metavar.contract}trades.log"

    def __init__(self, filename=savepath):
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


class Turtle(bt.Strategy):
    """Turtle trading system"""

    params = (
        ("s1_longperiod", 20),
        ("s1_shortperiod", 10),
        ("s2_longperiod", 55),
        ("s2_shortperiod", 15),
        ("bigfloat", 6),
        ("drawback", 1),
        ("closeout", 2),
        ("max_add", 3),
        ("theta", 0.01),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datadatetime = self.datas[0].datetime

        # if True, ignore the next 20days breakthrough
        # look for a 55days breakthrough
        self.ignore = False

        # 突破通道
        self.s1_long_h = bt.ind.Highest(self.datahigh(-1), period=self.p.s1_longperiod)
        self.s1_long_l = bt.ind.Lowest(self.datalow(-1), period=self.p.s1_longperiod)
        self.s1_short_h = bt.ind.Highest(self.datahigh(-1), period=self.p.s1_shortperiod)
        self.s1_short_l = bt.ind.Lowest(self.datalow(-1), period=self.p.s1_shortperiod)
        self.s1_longsig = bt.ind.CrossOver(self.dataclose(0), self.s1_long_h)
        self.s1_shortsig = bt.ind.CrossOver(self.s1_long_l, self.dataclose(0))
        self.s1_longexit = bt.ind.CrossOver(self.s1_short_l, self.dataclose(0))
        self.s1_shortexit = bt.ind.CrossOver(self.dataclose(0), self.s1_short_h)

        self.s2_long_h = bt.ind.Highest(self.datahigh(-1), period=self.p.s2_longperiod)
        self.s2_long_l = bt.ind.Lowest(self.datalow(-1), period=self.p.s2_longperiod)
        self.s2_short_h = bt.ind.Highest(self.datahigh(-1), period=self.p.s2_shortperiod)
        self.s2_short_l = bt.ind.Lowest(self.datalow(-1), period=self.p.s2_shortperiod)
        self.s2_longsig = bt.ind.CrossOver(self.dataclose(0), self.s2_long_h)
        self.s2_shortsig = bt.ind.CrossOver(self.s2_long_l, self.dataclose(0))
        self.s2_longexit = bt.ind.CrossOver(self.s2_short_l, self.dataclose(0))
        self.s2_shortexit = bt.ind.CrossOver(self.dataclose(0), self.s2_short_h)

        # 头寸管理
        self.tr = bt.ind.TR(self.datas[0])
        self.atr = bt.ind.ATR(self.datas[0], period=20)
        self.pos_count = 0
        self.margin_used = 0
        self.max_margin = metavar.startcash * 0.2
        self.start_value = metavar.startcash
        self.ann_profit = 0

        self.curpos = 0
        self.buyprice = None
        self.sellprice = None
        self.order = None

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
            margin_used = order.executed.price * abs(order.executed.size) * metavar.mult * metavar.margin
            self.margin_used += margin_used
            self.margin_pct = self.margin_used / self.broker.get_value()

            if order.isbuy():
                self.log(
                    "LONG CREATED @ {:.2f}, EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, TT_MARGIN {:.2%}, CURPOS {}".format(
                        order.created.price,
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
                    "SHORT CREATED @ {:.2f}, EXECUTED @ {:.2f}, SIZE {}, COST {:.2f}, COMMISSION {:.2f}, MARGIN {:.2f}, TT_MARGIN {:.2%}, CURPOS {}".format(
                        order.created.price,
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
        self.log(
            "HH {:.2f}, LL {:.2f}, OPEN {:.2f}, CLOSE {:.2f}, VALUE {:.2f}, CASH {:.2f}".format(
                self.s2_long_h[-1],
                self.s2_long_l[-1],
                self.dataopen[0],
                self.dataclose[0],
                self.broker.get_value(),
                self.broker.get_cash(),
            )

        )
        
        # 计算年内净值变动
        today = bt.num2date(self.datadatetime[0]).date().isoformat()
        self.ann_profit = self.cal_ann_profit(today)
        self.margin_used = self.reset_margin(today)
        self.max_margin = self.cal_max_margin(self.ann_profit)

        # 跳过当前bar的条件
        bypass_conds = [
            self.order,
            len(self) < self.p.s1_longperiod,
            self.margin_used >= self.max_margin
            ]
        if any(bypass_conds):
            return

        # if not self.ignore:
        # longsig = self.s1_longsig
        # shortsig = self.s1_shortsig
        # longexit = self.s1_longexit
        # shortexit = self.s1_shortexit
        # else:
        # longsig = self.s2_longsig
        # shortsig = self.s2_shortsig
        # longexit = self.s2_longexit
        # shortexit = self.s2_shortexit

        # # only use s2 system
        longsig = self.s2_longsig
        shortsig = self.s2_shortsig
        longexit = self.s2_longexit
        shortexit = self.s2_shortexit

        size = self.get_size(self.max_margin, self.margin_used, self.dataclose[0])

        if not self.position:
            # 通道突破
            if longsig > 0:
                self.order = self.buy(size=size)
            elif shortsig > 0:
                self.order = self.sell(size=size)

        else:
            # 大趋势判断
            # 若存在正向大趋势，执行紧缩性平仓
            uptrend = self.datahigh[-1] - self.buyprice > self.p.bigfloat * self.atr[0] if self.buyprice else False
            downtrend =  self.sellprice - self.datahigh[-1] < -self.p.bigfloat * self.atr[0] if self.sellprice else False
            if uptrend or downtrend:
                stop_limit = self.p.drawback
            else:
                stop_limit = self.p.closeout

            if self.position.size > 0:
                price_change = self.dataclose[0] - self.buyprice

                # 多头加仓
                if (price_change >= 0.5 * self.atr[0]) and (self.pos_count < self.p.max_add):
                    self.order = self.buy(size=size)
                    self.pos_count += 1

                # 多头平仓
                if longexit > 0:  # 赢利性退场
                    self.order = self.close()
                    self.order.addinfo(name='WINNER CLOSE')
                    self.pos_count = 0
                    self.ignore = True
                    self.margin_used = 0
                elif price_change <= -stop_limit * self.atr[0]:  # 亏损性退场
                    self.order = self.close()
                    self.order.addinfo(name='LOSER CLOSE')
                    self.pos_count = 0
                    self.ignore = False
                    self.margin_used = 0

            else:
                price_change = self.dataclose[0] - self.sellprice

                # 空头加仓
                if (price_change <= -0.5 * self.atr[0]) and (self.pos_count < self.p.max_add):
                    self.order = self.sell(size=size)
                    self.pos_count += 1

                # 空头平仓
                if shortexit > 0:  # 赢利性退场
                    self.order = self.close()
                    self.order.addinfo(name='WINNER CLOSE')
                    self.pos_count = 0
                    self.ignore = True
                    self.margin_used = 0
                elif price_change >= stop_limit * self.atr[0]:  # 亏损性退场
                    self.order = self.close()
                    self.order.addinfo(name='LOSER CLOSE')
                    self.pos_count = 0
                    self.ignore = False
                    self.margin_used = 0

    def stop(self):
        pass

    def get_size(self, max_margin, margin_used, price):
        """Calculate size based on ATR"""
        # 最大保证金约束
        available_margin = max_margin - margin_used
        margin_size = available_margin // (price * metavar.mult * metavar.margin)

        # 真是波动率约束
        n = self.atr[0]
        abs_vol = n * metavar.mult  # N * CN
        atr_size = (self.broker.get_value() * self.p.theta) // abs_vol  # 1 unit

        return min(atr_size, margin_size)
    
    def cal_ann_profit(self, date):
        """计算每年的净值变动"""
        if date in metavar.first_days:
            self.start_value = self.broker.get_value()

        return self.broker.get_value() / self.start_value
    
    def reset_margin(self, date):
        """重制每年保证金占用"""
        if date in metavar.first_days:
            self.margin_used = 0
        
        return self.margin_used

    def cal_max_margin(self, ann_profit):
        """根据净值范围调整最大保证金比例"""
        if ann_profit >= 1.1:
            max_pct = 0.50
        elif 1.1 > ann_profit >= 1:
            max_pct = 0.40
        elif 1 > ann_profit >= 0.95:
            max_pct = 0.30
        else:
            max_pct = 0.10

        return max_pct * self.broker.get_value()


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
        return price * self.p.mult * self.p.margin


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
    # cerebro.addsizer(TurtleSizer)

    # Analysers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    # Backtesting
    init_msg = f"""
            回测对象: {metavar.contract}
            起始时间: {metavar.fromdate}
            终止时间: {metavar.todate}
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
    # print(metavar.first_days)
