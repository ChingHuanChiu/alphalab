import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# def ARIMA(df):
#     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#     import statsmodels.api as sm
#     from statsmodels.tsa.arima_model import ARIMA
#
#     #ACF和PACF圖
#     fig = plt.figure(figsize=(12, 8))
#     ax1 = fig.add_subplot()
#     fig = plot_acf(df)
#     ax2 = fig.add_subplot()
#     fig = plot_pacf(df)




def ETS(x, model='multiplicative'):
    '''

  檢視資料是否有季節性:
    '''
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(x, model=model)
    result.plot()


def CAPM(asset_return, market_return):
    '''
    找出CAPM的beta值與alpha值
    '''
    from scipy import stats
    beta, alpha, r_value, p_value, std_err = stats.linregress(asset_return, market_return)
    return beta, alpha


def portfolio_val(data, symbol:list, capital=100000, rf=0.00):
    symbol_df = []
    for s in symbol:
        symbol_df.append(data[data[symbol] == s])

    for stock_df in symbol_df:

        stock_df['Cumulative Return'] = stock_df['close']/stock_df.iloc[0]['close']

    weight = [1/(len(symbol_df))] * len(symbol_df)
    for stock_df, alloc in zip(symbol_df, weight):
        stock_df['Allocation'] = stock_df['Cumulative Return'] * alloc
        stock_df['Position Values'] = stock_df['Allocation'] * capital

    portfolio_val = pd.DataFrame(columns=symbol)
    for s in symbol_df:
        portfolio_val = pd.concat([portfolio_val, s['Position Values']], axis=1)
    portfolio_val['Total'] = portfolio_val.sum(axis=1)
    portfolio_val['Daily Return'] = portfolio_val['Total'].pct_change(1)
    portfolio_val['Cumulative  Daily Return'] = 100 * (portfolio_val['Total'][-1] / portfolio_val['Total'][0] - 1)

    daily_mean_return = portfolio_val['Daily Return'].mean()
    daily_mean_std = portfolio_val['Daily Return'].std()
    SR = daily_mean_return - rf / daily_mean_std
    ASR = (252 ** 0.5) * SR
    return portfolio_val, ASR


class portfolio():
    '''
    data:股票收盤價的資料, index=日期
    '''
    def __init__(self, data, stock: list):
        self.stock = stock
        self.data = data

        self.new_df = pd.DataFrame(columns=self.stock)
        for s in self.stock:
            self.new_df[s] = self.data[self.data.symbol == s]['close']

        self.log_re = np.log(self.new_df / self.new_df.shift(1)).dropna(axis=0)

    def opt_weight_monte_carlo(self, num_times=1500, plot=False):


        all_weights = np.zeros((num_times, len(self.new_df.columns)))
        re_arr = np.zeros(num_times)
        vol_arr = np.zeros(num_times)
        sharpe_arr = np.zeros(num_times)

        for idx in range(num_times):
            # 創建隨機權重
            weights = np.array(np.random.random(len(self.stock)))
            # 重新平衡權重
            weights = weights / np.sum(weights)
            # 保存權重
            all_weights[idx, :] = weights
            # 預期收益
            re_arr[idx] = np.sum((self.log_re.mean() * weights) * 252)
            # 預期波動率
            vol_arr[idx] = np.sqrt(np.dot(weights.T, np.dot(self.log_re.cov() * 252, weights)))
            # 夏普比率
            sharpe_arr[idx] = re_arr[idx] / vol_arr[idx]
            #夏普比率最大的權重組合
            opt_weights = all_weights[sharpe_arr.argmax(), :]

            max_sr_re = re_arr[sharpe_arr.argmax()]
            max_sr_vol = vol_arr[sharpe_arr.argmax()]

        if plot:
            plt.figure(figsize=(12, 8))
            plt.scatter(vol_arr, re_arr, c=sharpe_arr, cmap='viridis')
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('Volatility')
            plt.ylabel('Return')
            # 在最大的夏普比率加給紅點
            plt.scatter(max_sr_vol, max_sr_re, c='red', s=50, edgecolors='black')
        return opt_weights


    def opt_weight(self, weights):

        def get_re_vol_sr(weights):

            weights = np.array(weights)
            # 回報
            exp_re = np.sum(self.log_re.mean() * weights) * 252
            # 波動率
            exp_vol = np.sqrt(np.dot(weights.T, np.dot(self.log_re.cov() * 252, weights)))
            # 夏普比率
            SR = exp_re / exp_vol
            # 返回以上資料
            return np.array([exp_re, exp_vol, SR])

        # 　把夏普比率變為負值
        def neg_sr(weights):
            return get_re_vol_sr(weights)[2] * -1
        # 根據最小化函數的慣例，應該是一個為條件返回零的函數
        # type是限制的種類, 'eq'表示均等
        # fun是限定約束的函數

        # def check_sum(weights):
        #     return np.sum(weights) - 1
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple([(0, 1)]*len(weights))
        init_weights = [1. / len(weights)] * len(weights)
        opt_results = minimize(neg_sr, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
        opt_weights = opt_results.x

        return opt_weights


if __name__ == "__main__":

    import yfinance as yf
    O = yf.Ticker('o').history(start='2020-01-01', end='2020-05-06')
    O['symbol'] = ['o'] * len(O)

    VYM = yf.Ticker('MMM').history(start='2020-01-01', end='2020-05-06')
    VYM['symbol'] = ['MMM'] * len(VYM)
    dataa = pd.concat([O, VYM], axis=0)
    dataa.columns = [c.lower() for c in dataa.columns]
    port = portfolio(dataa, ['o', 'MMM'])
    w1 = port.opt_weight(weights=[1, 1]).round(3)
    ww = port.opt_weight_monte_carlo(plot=True)