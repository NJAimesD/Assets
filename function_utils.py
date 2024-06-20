"""
    Code for simultating the efficient frontier 
    of the Markowitz modelling approach.
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

plt.style.use("ggplot")


def portfolio_variance(x, mtx_var_covar):
    """
        Portfolio Variance Function
    """
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_covar, x))
    return variance

class Asset:

    def __init__(self, ric, **kwargs):
        
        self.ric = ric
        self._start_date = kwargs.get("start_date") or '2021-01-01'
        self._history = self._get_history(self._start_date)
        self.expected_return = self._get_return()[0].round(6)
        self.variance = self._get_variance().values[0][0].round(6)
        self.compoundedreturns = np.log(self._history).diff()

    def __repr__(self):
        return f"Asset({self.ric}, start_date:{self._start_date})"
    
    def __str__(self):
        return self.ric

    def _get_history(self, start_date):
        history = yf.download([self.ric], start=start_date, progress=False)
        history = history.drop(
            ["Open", "Low", "Close", "High", "Volume"], axis=1
        ).fillna(.5)

        return history

    def _get_return(self):
        
        history = self._history
        returnslog = np.log(history)
        compoundedreturns = returnslog.diff()
        cretunrsmean = compoundedreturns.dropna().mean(axis=0)
        cretunrsmeandf = cretunrsmean.to_frame()
        expectedreturn = (np.exp(cretunrsmeandf)) - 1

        expectedreturn = expectedreturn.rename(
            {0: 'expected_return'}, 
            axis=1
        )

        highexreturn = expectedreturn['expected_return']
        highexreturn = pd.DataFrame(highexreturn,columns = ['expected_return'])
        expected_return = np.array(highexreturn['expected_return'])
        
        return expected_return

    def _get_variance(self):
        history = self._history
        returnslog = np.log(history)
        compoundedreturns = returnslog.diff()
        covreturns = compoundedreturns.cov()
        return covreturns

    def plot_time_series(self):
        series = self.compoundedreturns

        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize=(20, 5))
        
        ax[0].plot(series, color='darkblue')
        
        ax[0].axhline(series['Adj Close'].mean(), color='darkred')
        ax[1].hist(
            series['Adj Close'], 
            color='orange',
            bins=np.arange(series['Adj Close'].min(), series['Adj Close'].max(), .001)
        )
        ax[0].set_facecolor("gainsboro")
        ax[1].set_facecolor("gainsboro")

        fig.suptitle(f'Log Returns {self.ric}')
        return fig, ax

class Portfolio:
    """
        Object that handles Porfolio building, metrics such 
        as expected return, portfolio variance, and covariance
        matrix.
    """
    def __init__(self, assets=[], weights=[], **kwargs):
        self._weights = weights
        self._start_date = kwargs.get("start_date") or '2021-01-01'
        self._assets = [Asset(x, start_date=self._start_date) for x in assets]
        #self._history = self._get_history(self._start_date)

    def __repr__(self):
        assets = ", ".join([str(x) for x in self._assets])
        weights = ", ".join([str(x) for x in self._weights])
        return f"Portfolio(Assets=[{assets}], Weights=[{weights}])"
    
    def __str__(self):
        assets = ", ".join([str(x) for x in self._assets])
        weights = ", ".join([str(x) for x in self._weights])
        return f"Portfolio(Assets=[{assets}], Weights=[{weights}])"

    def get_history(self):
        data = pd.DataFrame()
        for asset in self._assets:
            data[asset.ric] = asset._history['Adj Close']
        return data.reset_index()

    def get_cov_matrix(self):
        data = pd.DataFrame()
        for asset in self._assets:
            data[asset.ric] = asset.compoundedreturns['Adj Close']
        
        cov_matrix = data.cov()

        return cov_matrix

    def get_variance(self):
        
        cov_mat = self.get_cov_matrix()
        variance = np.matmul(
            np.transpose(self._weights), np.matmul(cov_mat, self._weights)
        )
        return variance

    def get_return(self):
        returns = [asset.expected_return for asset in self._assets]
        expected_return = np.dot(np.array(returns), np.array(self._weights))
        return expected_return.round(6)

class PortfolioManager:
    """
        Object that handles Porfolio Optimization, with several
        optimization methods.
    """
    def __init__(self, type, portfolio):
        self.type = type
        self._base_portfolio = portfolio
        self._base_assets = [str(x) for x in portfolio._assets]
        self._base_returns = np.array([asset.expected_return for asset in self._base_portfolio._assets])
        self._asset_count = len(self._base_portfolio._assets)
        self._x0 = [1 / self._asset_count] * (self._asset_count)
        self._base_cov_mat = self._base_portfolio.get_cov_matrix()
        self._min_returns = np.min(self._base_returns)
        self._max_returns = np.max(self._base_returns)
        plt.style.use("ggplot")
        #x0 = [1 / len(symbolslist)] * len(symbolslist)

    def __repr__(self):
        return f"PortfolioManager(base_portfolio={self._base_portfolio}, {self.type})"

    def get_markowitz_weights(self, target_return):
        
        non_negative = [(0, None) for i in range(self._asset_count)]
        l1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]   # unitary in norm L2
        markowitz = [{"type": "eq", "fun": lambda x: self._base_returns.dot(x) - target_return}] 

        optimal_result = op.minimize(
            fun=portfolio_variance, 
            x0=self._x0,\
            args=(self._base_cov_mat),\
            constraints=(l1_norm + markowitz),\
            bounds=non_negative,
            options={'maxiter': 5000}
        )

        weights = np.array(optimal_result.x)
        weights /= sum(abs(weights))

        return weights
    
    def get_efficient_frontier(self):
        target_returns = self._min_returns + np.linspace(0.05,0.95,100) * (self._max_returns-self._min_returns)
        weights = [
            self.get_markowitz_weights(x) for x in target_returns
        ]

        portfolios = [
            Portfolio(self._base_assets, list(weight)) for weight in weights
        ]

        variances = [
            portfolio.get_variance() for portfolio in portfolios
        ]

        return target_returns, variances

    def plot_efficient_frontier(self):
        
        target_returns, variances = self.get_efficient_frontier()
        
        fig, ax = plt.subplots(
            figsize=(8, 4), dpi=180
        )

        ax.set_facecolor("paleturquoise")
        
        fig.set_facecolor("azure")
        
        ax.scatter(
            self._base_portfolio.get_variance(),
            self._base_portfolio.get_return(),
            color='darkred',
            label='Base Portfolio'
        )

        ax.plot(
            variances,
            target_returns,
            label="Efficient Frontier",
            color='darkcyan'
        )

        ax.set_title(
            f"Markowitz Porfolio for assets {self._base_assets}"
        )

        ax.set_xlabel(
            "Volatilty"
        )


        ax.set_ylabel(
            "Return"
        )
        ax.legend()

        return fig, ax
    
def get_weights(n):
    """
        Returns a vector of size n, with weights, the sum should be 1.
    """

    search_space = np.linspace(0, 1, 1_000_000)
    cumulative_weights = 0
    vector_weight = []

    for i in range(n - 1):
        weight = np.random.choice(list(search_space)) ### uniform distribution.
        vector_weight.append(weight)
        cumulative_weights = cumulative_weights + weight
        search_space = np.linspace(0, 1 - cumulative_weights, 1_000_000)

    last_weight = 1 - cumulative_weights
    vector_weight.append(last_weight)
    return vector_weight
