# Customised commission info class for
# each of three main contracts

import backtrader as bt

class FurCommInfo(bt.CommInfoBase):
    """定义期货的交易手续费和佣金"""

    params = (
        ("stocklike", False),
        ("commtype", bt.CommInfoBase.COMM_PERC),  # 按比例收取手续费
        ("percabs", True),  # 0.0002 = 0.2%
        ("commission", 0.23 / 10000),
        ("backtest_margin", 1.0),  # no leverage
    )

    def _getcommission(self, size, price, pseudoexec):
        """手续费=买卖手数*合约价格*手续费比例*合约乘数"""
        return abs(size) * price * self.p.commission * self.p.mult

    def get_margin(self, price):
        """每笔交易保证金=合约价格*合约乘数*保证金比例"""
        return price * self.p.mult * self.p.margin


class IFCommInfo(FurCommInfo):

    params = (
        ("mult", 300),
        ("margin", 0.12),
    )


class IHCommInfo(FurCommInfo):

    params = (
        ("mult", 300),
        ("margin", 0.10),
    )


class ICCommInfo(FurCommInfo):

    params = (
        ("mult", 200),
        ("margin", 0.12),
    )