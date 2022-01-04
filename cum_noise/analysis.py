import pandas as pd
import numpy as np
import empyrical as emp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import datetime
import config


# initialise the global var
metavar = config.set_contract_var()

# initialise pyplot settings
plt.rcParams["axes.unicode_minus"] = False  # display minus sign correctly
plt.rcParams["font.family"] = ["Heiti TC"]  # for Chinese characters


class Factory:
    # Reproduce intuitive plots for timeseries data analysis

    params = {
            'figsize': (12, 8)
            }

    def __init__(self, results_df: pd.DataFrame, prices_df: pd.DataFrame):
        """
        Params
        ------
        results_df: Timereturn dataframe from the backtest programe
        prices_df: Dataframe of the underlying asset's adj close

        Attributes
        ----------
        - timereturn:
        - detailed_rets:
        - ann_rets:
        - month_rets:
        - cumrets:
        - drawdown:
        """
        # essential attributes
        self.timereturn = results_df
        self.detailed_rets = self.sep_month_and_year(self.timereturn)

        self.ann_rets, self.ann_mean = self.cal_annual_rets(self.detailed_rets)
        self.month_rets = self.cal_monthly_rets(self.detailed_rets)

        self.cumrets = emp.cum_returns(self.timereturn['timereturn'], starting_value=1.0)
        maxrets = self.cumrets.cummax()
        self.drawdown = (self.cumrets - maxrets) / maxrets

        self.ann_mdd, self.calmar = self.cal_annual_mdd_calmar(self.detailed_rets)
        
        self.prices_df = self.normalise_prices(self.timereturn, prices_df)

    def normalise_prices(self, timereturn, prices_df):
        """
        Normalise the prices dataframe starting from 1 in order to
        compare the trend between prices and cumrets over the same timeframe

        Params
        ------
        timereturn: Timereturn dataframe from the backtesting programe
        prices_df: Dataframe of the underlying asset's adj.close

        Retruns
        -------
        prices_df: Prices dataframe with the same timeframe
        """
        prices_df.index = pd.to_datetime(prices_df.index)
        fromdate, todate = timereturn.index[0].date().isoformat(), timereturn.index[-1].date().isoformat()
        prices_df = prices_df.loc[fromdate:todate]['S_DQ_ADJCLOSE']

        return prices_df
        
    def sep_month_and_year(self, df):
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        return df
    
    def cal_annual_rets(self, df):
        """
        Convert minute timereturn to annual returns

        Params
        ------
        df: Timereturn dataframe in minutes basis

        Returns
        -------
        ann_rets: Annual timereturn dataframe
        ann_mean: Mean value of the annual return
        """
        years = df['year'].unique()
        ann_rets = {}

        # scale the min returns to annual return
        for year in years:
            min_rets = df[df['year'] == year]['timereturn']
            ann_rets[year] = emp.cum_returns_final(min_rets, starting_value=1) - 1  # (1 + rets) ** (1 / num_year) - 1

        ann_rets = pd.Series(ann_rets)
        ann_mean = ann_rets.mean()
        
        return ann_rets, ann_mean

    def cal_annual_mdd_calmar(self, df):
        """
        Calculate the maximum drawdown and calmar ratio
        for each year's return

        Params
        ------
        df: Timereturn dataframe in minutes basis

        Returns
        -------
        ann_mdd: Annual maximum drawdown level
        ann_calmar: Annual calmar raito
        """
        years = df['year'].unique()
        ann_mdd = {} 
        ann_calmar = {}

        for year in years:
            min_rets = df[df['year'] == year]['timereturn']
            ann_rets = emp.cum_returns_final(min_rets, starting_value=1) - 1
            ann_mdd[year] = emp.max_drawdown(min_rets)
            ann_calmar[year] = ann_rets / -ann_mdd[year]

        ann_mdd = pd.Series(ann_mdd)
        ann_calmar = pd.Series(ann_calmar)

        return ann_mdd, ann_calmar
    
    def cal_monthly_rets(self, df):
        """
        Convert timereturn dataframe in minutes to montly basis

        Params
        ------
        df: Timereturn dataframe in minutes basis

        Returns
        -------
        monthly_rets: Monthly timereturn dataframe
        """
        years = df['year'].unique()
        monthly_rets = pd.DataFrame(columns=df['year'].unique(), index=range(1, 13))

        for year in years:
            year_df = df[df['year'] == year]
            months = year_df['month'].unique()
            
            # scale the min returns to monthly returns
            for month in months:
                min_rets = year_df[year_df['month'] == month]['timereturn']
                rets = emp.cum_returns_final(min_rets, starting_value=1) - 1  # (1 + rets) ** (1 / num_month) - 1
                monthly_rets.loc[month,year] = rets

        monthly_rets = monthly_rets.astype(float).T
        monthly_rets = monthly_rets.replace(np.nan, 0)

        return monthly_rets
        
    def plot_annrets_mdd_calmar(self, ann_rets, ann_mean, ann_mdd, ann_calmar):
        """
        Create a combination of bar chart and line chart that
        illustrate the annual return, max drawdown and calmar
        ratio for each year backtesting
        
        Params
        ------
        ann_rets: Timereturn dataframe in annual basis
        ann_mean: Mean return across the timeframe
        ann_mdd: Annual maximum drawdown
        ann_calmar: Annual calmar ratio

        Returns
        -------
        bar chart for annual returns
        """
        fig, ax1 = plt.subplots(figsize=self.params['figsize'])
        ax1 = sns.barplot(
            x=ann_mdd.index, y=ann_mdd.values,
            color='brown', capsize=0.3, label='最大回撤'
        )

        ax2 = ax1.twinx()
        ax2.plot(
                ax1.get_xticks(), ann_calmar.values,
                color='tab:green', label='收益回撤比',
                marker='o', linewidth=3
                )
        
        ax3 = ax3 = sns.barplot(
            x=ann_rets.index, y=ann_rets.values,
            color='tab:blue', capsize=0.3,
            ax=ax1, label='年化收益率'
        )

        # average line
        ax3.axhline(ann_mean, alpha=0.8, linewidth=4,
                   dashes=(5, 2), color='black', label=f'平均值 {ann_mean:.2%}')
        
        # solid line in the x axis
        ax3.axhline(0, color='black')
        
        # set axis labels
        ax1.set_xlabel('年份')
        ax1.set_ylabel('最大回撤')
        ax2.set_ylabel('收益回撤比')
        ax3.set_ylabel('收益率')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()

        # format y axis to percentage style
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper right')
        # plt.legend(h1, l1, fontsize=12, loc='upper right')
        plt.title('年化收益率', fontsize=14)
        fig.tight_layout()

        plt.savefig(f'./images/{metavar.contract}rets_mdd_calmar.png')
        plt.show()

    def plot_month_rets_heatmap(self, monthly_rets):
        """
        Plot the monthly return for each year in heatmap

        Params
        ------
        monthly_rets: Monthly timereturn dataframe

        Returns
        -------
        Heatmap of monthly return
        """
        fig, ax = plt.subplots(figsize=self.params['figsize'])
        values = monthly_rets.values  # np.array of the monthly returns

        ax = sns.heatmap(monthly_rets, cmap='RdBu', cbar=True)

        # make sure the annot is in the center of the block
        # while also in a percentage format
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                # do not display the nan in annot
                if values[i, j] != 0:
                    text = ax.text(j + 0.5, i + 0.5, f"{values[i, j]:.2%}",
                                    ha='center', va='center', color="black")

        ax.set_ylabel('年份')
        ax.set_xlabel('月份')
        plt.title('月度收益率', fontsize=14)
        fig.tight_layout()

        plt.savefig(f'./images/{metavar.contract}month_rets_heatmap.png')
        plt.show()

    def plot_cumrets_dd_prices(self, cumrets, dd, prices):
        """
        Plot a line graph that shows the trends among cumulative returns,
        drawdown and the underlying's prices over the whole period
        
        Params
        ------
        cumrets: pd.DataFrame of the cumulative returns
        dd: pd.DataFrame of the return drawdown
        prices: pd.DataFrame of the underlying asset's ADJCLOSE
        
        Returns
        -------
        Line graph of params trends over the timeframe 
        """
        fig, ax1 = plt.subplots(figsize=self.params['figsize'])
        
        # cumulative returns
        cumrets.plot(ax=ax1, rot=45, grid=False, label="净值曲线", color="brown", linewidth=2)

        # drawdown and prices
        ax2 = ax1.twinx()
        ax2 = dd.plot.area(grid=False, label="回撤情况", alpha=0.3, color="tab:blue", linewidth=1)

        ax3 = ax1.twinx()
        prices.plot(
                ax=ax3, grid=False, label='价格曲线',
                alpha=0.4, color='grey', linewidth=2,
                )
        ax3.spines['right'].set_position(('outward', 60))  # outer axis

        ax1.set_xlabel("日期")
        ax1.set_ylabel("净值")
        # ax2.set_ylabel('回撤情况')
        # ax3.set_ylabel('价格曲线')

        ax1.yaxis.set_ticks_position("left")
        ax2.yaxis.set_ticks_position("right")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()

        plt.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=12, ncol=1, loc="upper right")
        plt.title(f"{metavar.contract}回测结果")
        plt.margins(x=0)
        fig.tight_layout()

        plt.savefig(f'./images/{metavar.contract}rets_dd_prices.png')
        plt.show()


if __name__ == '__main__':
    rets_file = "./timereturn.csv"
    prices_file = f"./1m_main_contracts/{metavar.contract}.csv"

    rets_df = pd.read_csv(rets_file)
    rets_df = rets_df.rename(columns={'Unnamed: 0': 'Date', '0': 'timereturn'})
    rets_df.set_index('Date', inplace=True)
    rets_df.index = pd.to_datetime(rets_df.index)
    prices_df = pd.read_csv(prices_file, index_col='TRADE_DT')

    ind = Factory(rets_df, prices_df)
    
    # calculate annually and monthly returns
    ann_rets, ann_mean = ind.ann_rets, ind.ann_mean
    monthly_rets = ind.month_rets

    # calculate annually max_drawdown and calmar ratio
    ann_mdd, ann_calmar = ind.ann_mdd, ind.calmar

    # plot the bar chart and heatmap
    ind.plot_annrets_mdd_calmar(ann_rets, ann_mean, ann_mdd, ann_calmar)
    ind.plot_month_rets_heatmap(monthly_rets)
    ind.plot_cumrets_dd_prices(ind.cumrets, ind.drawdown, ind.prices_df)

