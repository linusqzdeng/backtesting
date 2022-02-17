# Test the correlation of three main contracts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


datapath = '../data.csv'
data = pd.read_csv(datapath, index_col='TRADE_DT', parse_dates=True)
contracts = ['IF00', 'IH00', 'IC00']

df = pd.DataFrame(columns=contracts)
for c in contracts:
    contract_df = data[data['S_INFO_CODE'] == c].loc[:, 'S_DQ_CLOSE']
    df[c] = contract_df

df = df.dropna()
rets = df.pct_change().dropna()
rets.corr().to_clipboard()
print(rets.corr())


def plot_rets(rets):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax = rets.plot()
    fig.tight_layout()
    plt.title('Price curve of main contracts')
    plt.show()

