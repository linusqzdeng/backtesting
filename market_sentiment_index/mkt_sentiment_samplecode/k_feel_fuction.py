# -*- coding: utf-8  -*-
# @Time    : 2020/3/12 11:31 上午
# @Author  : XuWenFu
# Sample refered to https://pan.baidu.com/s/15a2xPL6hr-1ZSEtclL2TAA


import pandas as pd
import empyrical as emp
from matplotlib import pyplot as plt

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)

# color可以百度颜色代码
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # 显示中文标签
plt.rcParams["font.serif"] = ["KaiTi"]
plt.rcParams["axes.unicode_minus"] = False


def cal_feel_stable(datas: list):
    """计算市场情绪的平稳度"""
    aria_datas = datas
    up_backs = []
    down_backs = []
    n = 0
    for data in range(n, len(aria_datas)):
        for data2 in range(n + 1, len(aria_datas)):
            if aria_datas[data2] - aria_datas[data] != 0:
                up = (aria_datas[data] - aria_datas[data2]) / aria_datas[data]
                up_backs.append(up)
                down = -(aria_datas[data] - aria_datas[data2]) / aria_datas[data]
                down_backs.append(down)

    # print('up_backs:', len(up_backs), up_backs)
    # print('down_backs', len(down_backs), down_backs)

    # 求平均最大回撤
    max_up_backs = []
    for up_data in range(len(up_backs)):
        if up_data + 1 == len(up_backs):
            break
        if up_backs[up_data] > 0 and up_backs[up_data + 1] < 0:
            max_up_backs.append(up_backs[up_data])

    # print('max_up_backs', max_up_backs)
    up_avg_back = sum(max_up_backs) / len(max_up_backs)
    # print('up_avg_back', up_avg_back)

    # 求反向最大回撤
    min_down_backs = []
    for down_data in range(len(down_backs)):
        if down_data + 1 == len(down_backs):
            break
        if down_backs[down_data] > 0 and down_backs[down_data + 1] < 0:
            min_down_backs.append(down_backs[down_data])

    # print('min_down_backs', min_down_backs)
    down_avg_back = sum(min_down_backs) / len(min_down_backs)
    # print('down_avg_back', down_avg_back)

    mdd, rev_mdd = [], []
    for idx, p in enumerate(datas):
        if idx + 1 == len(datas):
            break
        max_diff = p - min(datas[idx + 1:])
        min_diff = p - max(datas[idx + 1:])
        if max_diff != 0:
            mdd_per_bar = max_diff / p
            mdd.append(mdd_per_bar)
        if min_diff != 0:
            rev_mdd_per_bar = -min_diff / p
            rev_mdd.append(rev_mdd_per_bar)
        else:
            continue
    pass

    # 求今天情绪平稳度
    if up_avg_back - down_avg_back > 0:
        feel_stable = down_avg_back
    else:
        feel_stable = up_avg_back

    # print('市场情绪平稳度feel_stable：', feel_stable)
    return feel_stable


def buy_more(re_balance, day_buy: list, day_sell: list, security_money_ratio, onedays: list, day_damges=False):
    """买多"""
    # 计算买入的数量
    # 买入金额
    buy_money = onedays[51]
    # 计算保证金金额
    security_money = buy_money * 100 * security_money_ratio

    # 现有持股数量
    hold = (re_balance // security_money) * 100
    # 剩余金额
    re_balance = re_balance - security_money - hold * buy_money * spare

    # 记录当天买卖情况 #剩余金额，买入金额，持有情况，手续费花费
    day_buy.append(re_balance)
    day_buy.append(buy_money)
    day_buy.append(hold)
    day_buy.append(hold * buy_money * spare)

    # 剩余交易时间
    re_ondays = onedays[51:]
    for re_data in re_ondays:
        # 计算损失金额
        damages_money = (buy_money - re_data) / buy_money

        # 如果今日做多行情中，达到止损点，就卖出
        if damages_money >= damges:
            sell_money = re_data
            re_balance = re_balance + security_money - hold * sell_money * spare + (sell_money - buy_money) * hold
            day_damges = True
            break

    if not day_damges:
        # 否则，就以当日的收盘价卖出
        sell_money = re_ondays[-1]
        re_balance = re_balance + security_money - hold * sell_money * spare + (sell_money - buy_money) * hold

    # 记录当天卖出情况
    day_sell.append(re_balance)
    day_sell.append(sell_money)
    day_sell.append(hold)
    day_sell.append(hold * sell_money * spare)
    hold = 0

    # 记录当天买卖情况
    return day_buy, day_sell, day_damges, re_balance


def sell_less(re_balance, day_buy: list, day_sell: list, security_money_ratio, onedays: list, day_damges=False):
    """卖空"""
    # 计算买入的数量
    # 买入金额
    buy_money = onedays[51]
    # 计算保证金金额
    security_money = buy_money * 100 * security_money_ratio

    # 现有持股数量
    hold = (re_balance // security_money) * 100
    # 剩余金额
    re_balance = re_balance - security_money - hold * buy_money * spare

    # 记录当天买卖情况 #剩余金额，买入金额，持有情况，手续费花费
    day_buy.append(re_balance)
    day_buy.append(buy_money)
    day_buy.append(hold)
    day_buy.append(hold * buy_money * spare)

    # 剩余交易时间
    re_ondays = onedays[51:]
    for re_data in re_ondays:
        # 计算损失金额
        damages_money = (re_data - buy_money) / buy_money
        # 如果今日做多行情中，达到止损点，就卖出
        if damages_money >= damges:
            sell_money = re_data
            re_balance = re_balance + security_money - hold * sell_money * spare - (sell_money - buy_money) * hold
            day_damges = True
            break

    if not day_damges:
        # 否则，就以当日的收盘价卖出
        sell_money = re_ondays[-1]
        re_balance = re_balance + security_money - hold * sell_money * spare + (buy_money - sell_money) * hold

    # 记录当天卖出情况
    day_sell.append(re_balance)
    day_sell.append(sell_money)
    day_sell.append(hold)
    day_sell.append(hold * sell_money * spare)
    hold = 0

    # 记录当天买卖情况
    return day_buy, day_sell, day_damges, re_balance


def get_date(times):
    """生成年月日时间"""
    dates = []
    for time in times:
        if time not in dates:
            dates.append(time)
    return dates


if __name__ == "__main__":
    """此为文件执行入口，从此处执行文件"""

    print("-------------基于市场情绪平稳度日内股指期货交易策略回测开始--------------------------")
    # 读取并处理转换数据
    daily = pd.read_csv("./data.csv", index_col=0, parse_dates=True)
    daily.index.name = "Date"
    daily["Open"] = daily["open"]
    daily["High"] = daily["high"]
    daily["Low"] = daily["low"]
    daily["Close"] = daily["close"]
    daily["Volume"] = daily["volume"]
    daily = daily[["Open", "High", "Low", "Close", "Volume"]]
    # daily.drop_duplicates()
    print("总共多少条数据：", daily.shape)
    print("总共交易多少天：", daily.shape[0] // 270)
    print("查看前5条数据：\n", daily.head(5), "\n", daily.tail(5))

    # 将收盘价格转换为列表
    usedf = daily.groupby(by=[daily.index.year, daily.index.month, daily.index.day], as_index=True,)[
        "Close"
    ].apply(lambda x: x.tolist())
    usedf = usedf.tolist()

    # 开仓和关仓处理
    print("设置初始金额为：50万元")
    print("设置初始观察期：50分钟/日")
    print("设置波动阀值为：9/10000")
    print("设置双边交易成本为；1.5/10000")
    print("设置初始止损为：0.5%")
    print("保证金比例，按照1：1设置")
    # 保证金比例，按照1：1设置
    security_money_ratio = 1
    # 余额50万，持有数量0
    base_money = 500000
    re_balance = 500000
    hold = 0
    # 设置止损为0.5%
    damges = 0.0055
    # 设置情绪平稳度为万9
    mov = 0.0009
    # 手续费为万1.5
    spare = 0.00015
    # 记录交易结果金额
    record_balances = []
    # 记录当天交易情况
    records = []

    # 开始交易
    for onedays in usedf:

        # 数据校验，查看是否满足当日交易条件
        if re_balance < re_balance / 5:
            break
        if re_balance < (onedays[0] * 101 * security_money_ratio):
            print("自有资金量不足，无法完成购买，请充值")
            break

        day_buy = []
        day_sell = []
        # 当日做多
        buy_set = False
        sell_set = False
        # 计算波动率
        feel_stable = cal_feel_stable(onedays[0:50])

        # 如果波动率小，并且是上涨趋势，则开始开仓买多
        if feel_stable < mov and onedays[51] > onedays[0]:
            buy_set = True
            day_buy, day_sell, day_damges, re_balance = buy_more(
                re_balance=re_balance,
                day_buy=day_buy,
                day_sell=day_sell,
                onedays=onedays,
                security_money_ratio=security_money_ratio,
            )

        # 如果波动率小，并且是下跌趋势，则开始开仓卖空
        if feel_stable < mov and onedays[51] < onedays[0]:
            sell_set = True
            day_buy, day_sell, day_damges, re_balance = sell_less(
                re_balance=re_balance,
                day_buy=day_buy,
                day_sell=day_sell,
                onedays=onedays,
                security_money_ratio=security_money_ratio,
            )

        else:
            day_buy = [re_balance, 0, 0, 0]
            day_sell = [re_balance, 0, 0, 0]
            day_damges = False

        # 记录当天买卖情况
        day_cals = [buy_set, sell_set, feel_stable, day_damges, re_balance] + day_buy + day_sell
        records.append(day_cals)
        record_balances.append(re_balance)

    # figure图形图标的意思，在这里就是指我们画的图
    # 通过实例化一个figure并且传递参数
    # 在图像模糊的时候可以传入dpi参数，让图片更加清晰
    # fig = plt.figure(figsize=(20, 8), dpi=80)
    x = range(0, len(record_balances), 1)
    y1 = [(i / base_money) for i in record_balances]
    y2 = [closes[-1] for closes in usedf]
    y2 = [i / y2[1] for i in y2]
    plt.figure()
    plt.plot(x, y1, label="策略收益率", color="r")
    plt.plot(x, y2, label="股价波动率", color="b")
    plt.title("股指期货情绪平稳度交易策略")
    plt.xlabel("时间线")
    plt.ylabel("收益率/股价波动率曲线")
    plt.legend()
    plt.show()

    times = get_date(daily.index.date)
    records_df = pd.DataFrame(records, index=times)
    records_df.columns = [
        "买多",
        "卖空",
        "平稳度",
        "是否平仓",
        "当日余额",
        "买入后剩余金额",
        "买入价",
        "买入持仓",
        "买入手续费",
        "卖出初始金额",
        "卖出价",
        "卖出持仓",
        "卖出手续费",
    ]
    print("查看交易结果：\n", records_df.head(20))
    print("最终收益率：", record_balances[-1] / record_balances[0])
    records_df.to_excel("./每日交易结果.xlsx")
    print("-------------基于市场情绪平稳度日内股指期货交易策略回测结束--------------------------")
