import pandas as pd
import numpy as np


def create_df(data, contract):
    """
    Convert the raw .csv file into the target
    dataframe with specified frequency
    """
    df = data[data["S_INFO_CODE"] == contract]

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

    return df


def daily_msi(df):
    # dates = np.unique(df.index.date)
    df = df['S_DQ_ADJCLOSE']
    df = df.groupby(by=df.index.date).apply(msi)
    df.to_csv('./msi.csv', index=True, index_label='Date')


def msi(p):
    """
    Calculate the market sentiment index of a given time

    Params
    ------
    - prices: pd.Series
        Prices series of the observed period

    Returns
    -------
    - msi: float
    """
    p = p[:50]
    mdd, rev_mdd = [], []
    for i in range(len(p)):
        price_change = (p[: i + 1] - p[i]) / p[: i + 1]
        mdd.append(np.max(price_change))
        rev_mdd.append(-np.min(price_change))

    avg_mdd = np.mean(mdd)
    avg_rev_mdd = np.mean(rev_mdd)

    return min(avg_mdd, avg_rev_mdd)


if __name__ == "__main__":
    path = "../data.csv"
    data = pd.read_csv(path, index_col="TRADE_DT", parse_dates=True)
    df = create_df(data, "IF00")
    daily_msi(df)
