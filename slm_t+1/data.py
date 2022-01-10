# -*- coding: UTF-8 -*-
# Data collection program

import pandas as pd
import numpy as np

from datetime import date

# ts.set_token(TUSHAHRE_TOKEN)
# pro = ts.pro_api()

# start_date = ''
# end_date = '19960101'
# # df = pro.daily(ts_code='000001.SH', end_date=end_date)
# df = pro.stock_basic(exchange='SSE')

train_fromdate = date(1995, 1, 3)
train_todate = date(2004, 12, 31)
test_fromdate = date(2005, 1, 1)
test_todate = date(2013, 12, 31)

df = pd.read_csv('SSE_index.csv', delimiter=';', index_col='date', parse_dates=True)

def create_pattern(df):
    """Numerical pattern 1 for price goes down, 2 for price goes up"""
    df['ret'] = df['close'].pct_change()
    df['pat'] = df['ret'].apply(lambda x: 1 if x <= 0 else 2) 
    df = df.dropna()

    return df

def get_patterns(df, n):
    pat_list = [df['pat'].iloc[i:i + n].values for i in range(len(df) - n + 1)]

    return pd.Series(pat_list)


pat_df = create_pattern(df)

print(pat_df)
print(get_patterns(pat_df, 3))


