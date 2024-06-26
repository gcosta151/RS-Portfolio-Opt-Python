# portfolio library
#
# Prepared by:    Giorgio Costa
# Last revision:  11-Apr-2024
#
#===============================================================================
# Import packages
#===============================================================================
import pandas as pd
import numpy as np
from scipy.stats import gmean
from tqdm import tqdm

from rsopt import hmm
from rsopt import factormodel as fm
from rsopt import optimization as opt

#===============================================================================
# Class Portfolio
#===============================================================================
class Portfolio:
    """Portfolio object 

    Inputs
    ------
    name: str
        Name of the portfolio
    assets: list
        List of assets in the portfolio
    features: list
        List of features used to build the factor model
    model: str
        Optimization model used to construct the portfolio
    regimeswitching: bool
        Boolean indicating whether to use regime switching
    constraints: list
        Set of constraints to impose on the optimization model
    solver: str
        Optimization solver to use

    Attributes
    ----------
    name: str
        Name of the portfolio
    assets: list
        List of assets in the portfolio
    features: list
        List of features used to build the factor model
    model: opt.{model}
        Optimization model object used to construct the portfolio
    wealth: pd.DataFrame
        Timeseries with the portfolio wealth evolution during the backtest
    stats: PortfolioStatistics
        PortfolioStatistics object with the portfolio summary statistics
    """
    def __init__(self, 
                 name:str,
                 assets:list,
                 features:list,
                 model:str,
                 regimeswitching:bool,
                 constraints=['longonly'], 
                 solver='SCS', 
                 ):
        self.name = name
        self.assets = assets
        self.features = features
        self.regimeswitching = regimeswitching
        self.wealth = None
        self.stats = None
        self.model = eval('opt.'+model)(regimeswitching, 
                                        constraints, 
                                        len(assets), 
                                        solver
                                        )

#===============================================================================
# Class PortfolioStatistics
#===============================================================================
class PortfolioStatistics:
    """PortfolioStatistics object 
    Calculate the summary statistics of a portfolio backtest

    Inputs
    ------
    wealth: pd.DataFrame
        Timeseries with the portfolio wealth evolution during the backtest
    freq: str
        Frequency of observations
    riskfree: pd.Series
        Timeseries with the risk-free rate of return
    lookback: int
        Number of observations used to compute the rolling Sharpe ratio

    Attributes
    ----------
    mu: np.float64
        Average portfolio return (annualized)
    vol: np.float64 
        Average portfolio volatility (annualized)
    sharpe: np.float64
        Average portfolio Sharpe ratio (annualized)
    roll_sharpe: pd.DataFrame
        Rolling Sharpe ratio based on the lookback window (annualized)
    """
    def __init__(self, 
                 wealth:pd.DataFrame, 
                 freq:str, 
                 riskfree:pd.Series, 
                 lookback:int
                 ):
        prets = wealth.pct_change().sub(riskfree, axis=0).dropna()
        if freq.lower() == 'daily':
            freq = 252
        elif freq.lower() == 'weekly':
            freq = 52
        elif freq.lower() == 'monthly':
            freq = 12
        
        self.mu = prets.apply(lambda x: gmean(1 + x)) ** freq - 1
        self.vol = prets.std() * np.sqrt(freq)
        self.sharpe = self.mu / self.vol
        self.roll_sharpe = prets.rolling(lookback
                                         ).apply(lambda x: 
                                                 (gmean(1 + x) ** freq - 1)
                                                 / (x.std() * np.sqrt(freq))
                                                 ).dropna()

#===============================================================================
# function backtest
#===============================================================================
def backtest(portfolios:list, 
             data, 
             daterange: list,
             lookback:int=60,
             rebalfreq:int=6,
             rsfeatures:list=None,
             n_states:int=2
             ):
    """The backtest function conducts a historical backtest of the list of 
    portfolio objects provided. The backtest is conducted by optimizing and 
    rebalancing the portfolios over the desired date range.

    Inputs
    ------
    portfolios: list
        List of Portfolio objects to backtest
    data: HistoricalData
        HistoricalData object containing the feature and asset returns
    daterange: list
        List containing the start and end date of the backtest 
    lookback: int
        Number of months to use in the regression models for parameter 
        estimation
    rebalfreq: int
        Number of months between portfolio rebalancing periods
    rsfeatures: list
        List of features to be used to build the HMM
    n_states: int
        Number of states in the HMM

    Outputs
    -------
    Portfolio.wealth: pd.DataFrame
        The portfolio object is updated with the wealth evolution
    Portfolio.stats: PortfolioStatistics
        The portfolio object is updated with summary statistics
    """
    daterange = pd.date_range(daterange[0], 
                              daterange[1], 
                              freq=str(rebalfreq)+'MS'
                              )
    wealth = {p.name: [100.0] for p in portfolios}

    for sdate, edate in tqdm(zip(daterange[:-1], daterange[1:])):

        # Fit the HMM
        if rsfeatures is not None:
            rsmodel = hmm.HMM(data.frets[:sdate][rsfeatures], n_states)

        # Fit the factor model
        fmodel = fm.FactorModel(data, sdate, lookback, rsmodel)

        # Optimize the portfolios
        for p in portfolios:
            w = p.model.optimize(fmodel)
            cumrets = (data.arets.loc[sdate:edate,
                                     :].add(data.riskfree.loc[sdate:edate], 
                                            axis=0
                                            ) + 1).cumprod()
            cumrets = wealth[p.name][-1] * cumrets.multiply(w, axis=1).sum(axis=1)
            wealth[p.name].extend(cumrets.to_list())
    
    # Update the portfolio objects with the backtest results
    for p in portfolios:
        p.wealth = pd.DataFrame(wealth[p.name], 
                                index=data.arets.index[-len(wealth[p.name]):],
                                columns=[p.name]
                                )
        p.stats = PortfolioStatistics(p.wealth, 
                                      data.freq, 
                                      data.riskfree, 
                                      lookback
                                      )

################################################################################
# End