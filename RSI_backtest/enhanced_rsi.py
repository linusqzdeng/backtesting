import csv
import datetime
import os, sys

import backtrader as bt
import empyrical as emp
import numpy as np
import pandas as pd

import config
from ast import literal_eval


metavar = config.set_contract_var()

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
        ("fromdate", metavar._fromdate),
        ("todate", metavar._todate),
        ("datetime", None),  # index of the dataframe
        ("open", 0),
        ("high", 1),
        ("low", 2),
        ("close", 3),
        ("volume", 4),
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
        if metavar.closeout_type == 1:
            self.p.commission = 3.45 / 10000  # 平今仓
        else:
            self.p.commission = 0.23 / 10000  # 止盈止损平仓/开仓

        if size > 0:
            return abs(size) * price * self.p.commission * self.p.mult
        else:  # 卖出时考虑印花税
            return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.backtest_margin


class FurSizer(bt.Sizer):
    """基于真是波动幅度的头寸管理"""

    params = (
        ("theta", 0.02),  # 风险载荷
        ("adj_func", 1.0),  # 头寸调整函数
        ("fund", 100_000),  # 配置资金
        ("mult", 300),  # 合约乘数
        ("period", 11),  # 回测窗口
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        # price_change = abs(data.close[self.p.period] - data.close[0])
        # k_std = np.std(data.close.get(ago=0, size=self.p.period))  # 计算当前时间点前period天收盘价的标准差
        # size = self.p.adj_func * self.p.theta * self.p.fund // (k_std * price_change * self.p.mult) if price_change != 0 else 0

        tr = [
            max(data.high[0], data.close[-i]) - min(data.low[0], data.close[-i]) for i in range(1, self.p.period + 1)
        ]  # true range
        atr = sum(tr) / self.p.period  # 真实波动幅度
        size = self.p.adj_func * self.p.theta * self.p.fund // (atr * self.p.mult)

        return min(size, data.volume[0])  # 取计算所得值和当天成交量的最小值


class EnhancedRSI(bt.Strategy):
    params = (
        ("period", 14),  # 参考研报
        ("thold_l", 60),
        ("thold_s", 70),
        ("closeout_limit", 0.02),
        ("target_percent", 0.30),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt} - {txt}")

    def __init__(self):
        # 保存收盘价、开盘价、日期
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datadatetime = self.datas[0].datetime

        # 交易时间表
        filepath = os.path.join(os.curdir, "trading_time", "IF00_time.csv")
        self.time_df = pd.read_csv(filepath, index_col="date")

        # 设置指标
        self.rsi_s = bt.ind.RSI_SMA(self.datas[0], period=self.p.period, safediv=True)
        self.rsi_l = bt.ind.RSI_SMA(self.datas[1], period=self.p.period, safediv=True)

        # 处理等待中的order
        self.order = None
        # self.ordermin = None

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
        today = bt.num2date(self.datadatetime[0]).date()
        trading_period = literal_eval(self.get_tradetime(today).values[0])
        open_time = trading_period[0]
        close_time = trading_period[-2]  # 使用收盘时间前一个bar作为平今仓信号
        now = bt.num2time(self.datadatetime[0]).isoformat()

        # 记录当天开盘价
        if now == open_time:
            self.open_price = self.dataopen[0]

        # 跳过当前交易的条件
        bypass_conds = [
                    self.order,
                    # now == self.ordermin,
                    today == metavar._fromdate  # 跳过第一天
                    ]
        if any(bypass_conds):
            return

        # 交易信号
        # 做多信号：长期RSI > L & 短期RSI > S
        # 做空信号：长期RSI < 100-L & 短期RSI < 100-S
        long_sig = (self.rsi_l > self.params.thold_l) and (self.rsi_s > self.params.thold_s)
        short_sig = (self.rsi_l < 100 - self.params.thold_l) and (self.rsi_s < 100 - self.params.thold_s)

        # 策略逻辑
        metavar.closeout_type = 0
        if not self.position and now != close_time:  # 收盘前一个bar不建仓
            if long_sig:
                self.order = self.order_target_percent(target=self.p.target_percent)
            elif short_sig:
                self.order = self.order_target_percent(target=-self.p.target_percent)
        else:
            # 判断是否平今仓
            if now == close_time and self.position:
                metavar.closeout_type = 1
                self.order = self.close()
                self.order.addinfo(name="CLOSE OUT AT THE END OF THE DAY")
            else:
                # 止损平仓
                pct_change = self.dataclose[0] / self.open_price - 1  # 基于每日开盘价收益率
                cur_pos = self.broker.getposition(data=self.datas[0]).size
                long_close_sig = cur_pos > 0 and (pct_change < -self.p.closeout_limit)  # 持有多头且下跌超过阈值
                short_close_sig = cur_pos < 0 and (pct_change > self.p.closeout_limit)  # 持有空头且上涨超过阈值

                if long_close_sig or short_close_sig:
                    self.order = self.order_target_percent(target=0)
                    self.order.addinfo(name="CLOSE OUT DUE TO STOPLIMIT")

        # Observers数据写入本地文件
        self.write_obs(-1)

    def stop(self):
        # Observers数据写入本地文件 - 最后一个bar
        self.write_obs(0)

    def notify_order(self, order):
        # 不处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 处理已完成订单
        if order.status == order.Completed:
            margin_used = order.executed.price * abs(order.executed.size) * metavar.mult * metavar.margin
            # self.ordermin = bt.num2time(self.datadatetime[0]).isoformat()

            if order.isbuy():
                self.log(f"LONG SIG DETECTED @ {order.created.price:.2f}")
                self.log(
                    f"BUY EXECUTED {order.executed.price:.2f}, SIZE {order.executed.size:.2f}, COST {order.executed.value:.2f}, COMMISSION {order.executed.comm:.2f}, MARGIN {margin_used:.2f}"
                )
            elif order.issell():
                self.log(f"SHORT SIG DETECTED @ {order.created.price:.2f}")
                self.log(
                    f"SELL EXECUTED {order.executed.price:.2f}, SIZE {order.executed.size:.2f}, COST {order.executed.value:.2f}, COMMISSION {order.executed.comm:.2f}, MARGIN {margin_used:.2f}"
                )

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

    def get_tradetime(self, today) -> list:
        """获取当日所有交易时间点"""
        return metavar.time_df.loc[today.isoformat()]

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


def data_cleansing(filepath):
    """返回短期和长期价格序列"""
    # 保留字段OHLC, Volume
    cols = [
        "S_DQ_ADJOPEN",
        "S_DQ_ADJHIGH",
        "S_DQ_ADJLOW",
        "S_DQ_ADJCLOSE",
        "S_DQ_VOLUME",
    ]

    # Read .csv file and set TRADE_DT as index
    short_df = pd.read_csv(filepath, index_col="TRADE_DT")

    # 过滤字段
    short_df = short_df[cols]

    # Change TRADE_DT into datetime type
    short_df.index = pd.to_datetime(short_df.index)
    long_df = short_df.resample(rule="15min", origin="start").last().dropna()

    return short_df, long_df


if __name__ == '__main__':
    # 保存回测交易单到本地
    sys.stdout = Logger()

    # For handling input data
    short_df, long_df = data_cleansing(metavar.filepath)

    # Initiate the strategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EnhancedRSI)

    # Load datas
    data0 = MainContract(dataname=short_df)
    data1 = MainContract(dataname=long_df)

    # Add data feeds
    cerebro.adddata(data0, name="short")
    cerebro.adddata(data1, name="long")

    # Add analyser
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name="_TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="_DrawDown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_Return")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="_Sharpe")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="_AnnualSharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="_AnnualReturn")  # Annualisation

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

    rets = pd.Series(strats.analyzers._TimeReturn.get_analysis())

    # =========== for analysis.py ============ #
    rets.to_csv('timereturn.csv', index=True)
    # ======================================== #

    cumrets = emp.cum_returns(rets, starting_value=0)
    maxrets = cumrets.cummax()
    drawdown = (cumrets - maxrets) / maxrets
    max_drawdown = emp.max_drawdown(rets)
    calmar_ratio = emp.calmar_ratio(rets)

    # 夏普比率
    num_years = metavar._todate.year - metavar._fromdate.year
    yearly_trade_times = rets.shape[0] / num_years
    # ann_rets = cumrets[-1] ** (1 / num_years) - 1
    ann_rets = (1 + cumrets[-1]) ** (1 / num_years) - 1
    ann_std = emp.annual_volatility(rets)
    risk_free = 0
    sharpe = emp.sharpe_ratio(rets, risk_free=risk_free, annualization=yearly_trade_times)

    # 盈亏比
    mean_per_win = (rets[rets > 0]).mean()
    mean_per_loss = (rets[rets < 0]).mean()

    annual_rets = pd.Series(strats.analyzers._Return.get_analysis())
    annual_sharpe = strats.analyzers._Sharpe.get_analysis()["sharperatio"]
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
    results_df.to_clipboard()
    print(results_df)

