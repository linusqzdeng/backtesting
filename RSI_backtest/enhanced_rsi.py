import csv
import datetime
import os, sys

import backtrader as bt
import empyrical as emp
import numpy as np
import pandas as pd


class Config:

    valid_contracts = ["IF00", "IH00", "IC00"]
    contract = valid_contracts[0]

    data = os.path.abspath('../data.csv')
    df = pd.read_csv(data, index_col='TRADE_DT', parse_dates=True)
    fromdate = datetime.date(2010, 4, 16)
    todate = datetime.date(2015, 12, 21)

    startcash = 10_000_000
    stamp_duty = 0.001
    is_ctp = False  # 是否平今仓

    def __init__(self):
        # 交易时间表
        self.shortlen_df = self.create_df(self.df, self.contract, '5Min')
        self.longlen_df = self.create_df(self.df, self.contract, '15Min')
        self.time_df = self.create_timedf(self.shortlen_df)

        # 保证金比例和合约乘数
        self.margin = self.set_margin(self.contract)
        self.mult = self.set_mult(self.contract)

    def get_timedf(self, path):
        """Return the time df that contains the trading bars per day"""
        timedf = pd.read_csv(path, index_col='date')
        timedf.index = pd.to_datetime(timedf.index)

        return timedf

    def set_margin(self, contract):
        # 保证金设置
        if contract in ["IH00", "IC00"]:
            return 0.10
        elif contract == "IF00":
            return 0.12
        
        raise TypeError('Unvalid contract name')
    
    def set_mult(self, contract):
        # 乘数设置
        if contract in ["IF00", "IH00"]:
            return 300.0
        elif contract == "IC00":
            return 200.0

        raise TypeError('Unvalid contract name')

    def create_df(self, raw_df, contract, freq):
        """
        Convert the raw .csv file into the target
        dataframe with specified frequency
        """
        df = raw_df[raw_df["S_INFO_CODE"] == contract]

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

        df = df.resample(freq).agg(agg_dict).dropna()

        return df

    def create_timedf(self, df):
        """Return the time df that contains the trading bars per day"""
        timedf = pd.DataFrame(index=df.index.date)
        timedf["time"] = df.index.time
        timedf = timedf.groupby(timedf.index).apply(lambda x: list(x["time"]))
        timedf.index = pd.to_datetime(timedf.index)

        return timedf

metavar = Config()


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


class MainContract(bt.feeds.PandasData):
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


class FurCommInfo(bt.CommInfoBase):
    """定义期货的交易手续费和佣金"""

    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("commission", 3.45 / 10_000),  # 万分之3.45手续费
        ("mult", metavar.mult),  # 乘数300
        ("stamp_duty", metavar.stamp_duty),  # 印花税0.1%
        ("margin", metavar.margin),
        ("backtest_margin", 1.0),  # no leverage for now
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        手续费=买卖手数*合约价格*手续费比例*合约乘数

        根据平仓类型`var.closeout_type`决定手续费比例
        - 平昨仓/止盈止损平仓: 0.23 / 10000
        - 平今仓: 3.45 / 10000
        """
        if metavar.is_ctp == True:
            self.p.commission = 3.45 / 10000  # 平今仓
        else:
            self.p.commission = 0.23 / 10000  # 止盈止损平仓/开仓

        return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.margin


class FurSizer(bt.Sizer):
    """基于真是波动幅度的头寸管理"""

    params = (
        ("theta", 0.01),  # 风险载荷
        ("mult", metavar.mult),  # 合约乘数
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        abs_vol = self.strategy.atr[0] * self.p.mult  # N * CN
        unit = self.broker.get_value() * self.p.theta // abs_vol  # 1 unit

        return unit


class EnhancedRSI(bt.Strategy):
    params = (
        ("period", 14),  # 参考研报
        ("thold_l", 50),
        ("thold_s", 80),
        ("stop_limit", 0.02),
        ("target_percent", 0.40),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

    def __init__(self):
        # 保存收盘价、开盘价、日期
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime

        # 设置指标
        self.rsi_s = bt.ind.RSI(self.datas[0], period=self.p.period, safediv=True)
        self.rsi_l = bt.ind.RSI(self.datas[1], period=self.p.period, safediv=True)
        self.atr = bt.ind.ATR(self.datas[0], period=self.p.period)

        # 做多信号：长期RSI > L & 短期RSI > S
        # 做空信号：长期RSI < 100-L & 短期RSI < 100-S
        self.longsig = bt.ind.And(self.rsi_l > self.p.thold_l, self.rsi_s > self.p.thold_s)
        self.shortsig = bt.ind.And(self.rsi_l < 100 - self.p.thold_l, self.rsi_s < 100 - self.p.thold_s)

        # 处理等待中的order
        self.order = None
        self.open_price = metavar.shortlen_df.iloc[0]['S_DQ_ADJOPEN']

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            margin_used = (
                order.executed.price
                * abs(order.executed.size)
                * metavar.mult
                * metavar.margin
            )

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

            if order.info:
                self.log(f"INFO {self.order.info['name']}")

        # 处理问题清单
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(
                f"ORDER CANCELED/MARGIN/REJECTED **CODE**: {order.getstatusname()}"
            )

        # Write down if no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.bar_traded = len(self)
        self.log(
            f"OPERATION PROFIT {trade.pnl:.2f}, NET PROFIT {trade.pnlcomm:.2f}, TRADE AT BAR {self.bar_traded}"
        )

    def start(self):
        # Observers数据写入本地文件
        self.mystats = csv.writer(open("results.csv", "w"))
        self.mystats.writerow(
            [
                "datetime",
                "drawdown",
                "maxdrawdown",
                "timereturn",
                "value",
                "cash",
                "pnlplus",
                "pnlminus",
            ]
        )

    def next(self):
        date = bt.num2date(self.datadatetime[0]).date().isoformat()
        time = bt.num2time(self.datadatetime[0])
        trade_time = metavar.time_df.loc[date]
        open_time = trade_time[0]
        close_time = trade_time[-2]  # 使用收盘时间前一个bar作为平今仓信号

        # 记录当天开盘价
        if time == open_time:
            self.open_price = self.dataopen[0]

        # 跳过当前交易的条件
        bypass_conds = [
            self.order,
            len(self) < self.p.period,
        ]
        if any(bypass_conds):
            return

        # 策略逻辑
        metavar.is_ctp = False 
        if not self.position and time != close_time:  # 临近收盘不建仓
            if self.longsig:
                self.order = self.order_target_percent(target=self.p.target_percent)
                # self.order = self.buy()
            elif self.shortsig:
                self.order = self.order_target_percent(target=-self.p.target_percent)
                # self.order = self.sell()
        else:
            # 判断是否平今仓
            if time == close_time and self.position:
                metavar.is_ctp = True
                self.order = self.close()
                self.order.addinfo(name="CLOSE OUT AT THE END OF THE DAY")
            else:
                # 止损平仓
                pct_change = self.dataclose[0] / self.open_price - 1  # 基于每日开盘价收益率
                cur_pos = self.broker.getposition(data=self.datas[0]).size
                long_close_sig = cur_pos > 0 and (pct_change < -self.p.stop_limit)  # 持有多头且下跌超过阈值
                short_close_sig = cur_pos < 0 and (pct_change > self.p.stop_limit)  # 持有空头且上涨超过阈值

                if long_close_sig or short_close_sig:
                    self.order = self.close()
                    self.order.addinfo(name="CLOSE OUT DUE TO STOPLIMIT")

        # Observers数据写入本地文件
        self.write_obs(-1)

    def stop(self):
        # Observers数据写入本地文件 - 最后一个bar
        self.write_obs(0)


    def write_obs(self, t):
        self.mystats.writerow(
            [
                self.datadatetime.datetime(t).strftime("%Y-%m-%d %H:%M:%S"),
                f"{self.stats.drawdown.drawdown[0]:.2f}",
                f"{self.stats.drawdown.maxdrawdown[0]:.2f}",
                f"{self.stats.timereturn.timereturn[0]:.2f}",
                f"{self.stats.broker.value[0]:.2f}",
                f"{self.stats.broker.cash[0]:.2f}",
                f"{self.stats.trades.pnlplus[0]:.2f}",
                f"{self.stats.trades.pnlminus[0]:.2f}",
            ]
        )


def run():
    # 保存回测交易单到本地
    sys.stdout = Logger()

    # Initiate the strategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EnhancedRSI)

    # Load datas
    data0 = MainContract(dataname=metavar.shortlen_df)
    data1 = MainContract(dataname=metavar.longlen_df)

    # Add data feeds
    cerebro.adddata(data0, name="short")
    cerebro.adddata(data1, name="long")

    # Add analyser
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="_Sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_AnnualReturn")

    # Add observers
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.TimeReturn)

    # Broker setup
    cerebro.broker.setcash(metavar.startcash)
    # cerebro.addsizer(FurSizer)

    # IF00 commission
    comminfo = FurCommInfo()
    cerebro.broker.addcommissioninfo(comminfo)

    # Output log file
    cerebro.addwriter(bt.WriterFile, out="log.csv", csv=True)
    init_msg = f"""
            策略: 改良长短RSI
            回测对象: {metavar.contract}
            起始时间: {metavar.fromdate}
            终止时间: {metavar.todate}
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

    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())

    # =========== for analysis.py ============ #
    rets.to_csv("timereturn.csv", index=True)
    # ======================================== #

    cumrets = emp.cum_returns(rets, starting_value=0)
    num_years = metavar.todate.year - metavar.fromdate.year
    ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    max_drawdown = emp.max_drawdown(rets)
    calmar_ratio = emp.calmar_ratio(rets)

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
        "交易次数": round(rets.shape[0], 0),
        "获胜次数": round(sum(rets > 0), 0),
        "胜率": sum(rets > 0) / rets.shape[0],
        "盈亏比": abs(mean_per_win / mean_per_loss),
    }

    results_df = pd.Series(results_dict)
    print(results_df)

if __name__ == "__main__":
    run()
