# Regime-switching applications for portfolio optimization
# Python version
#
# Prepared by:    Giorgio Costa
# Last revision:  27-Dec-2023
#
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
import pandas as pd
from rsopt import dataload as dl
from rsopt import portfolio as po

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
"""
Select the parameters for the portfolio optimization backtest
- daterange: Backtest start and end dates
- calibration: HMM calibration window (in years)
- lookback: lookback window used to calibrate the factor models (in months)
- rebalfreq: rebalance fequency (in months)
"""
daterange = [pd.to_datetime('2001-01-01'), pd.to_datetime('2023-01-01')]
calibration = pd.DateOffset(years=30)
lookback:int = 60
rebalfreq:int = 6

"""
Load historical data from Kenneth French's website (Fama-French data)
- sourcedata: load one of the following datasets as the asset data:
    - 10_Industry_Portfolios
    - 30_Industry_Portfolios
    - 49_Industry_Portfolios
- datafreq: data frequency can be specified as 
    - daily
    - weekly
    - monthly
"""
sourcedata = '30_Industry_Portfolios'
datafreq = 'weekly'
data = dl.HistoricalData(sourcedata, 
                         daterange, 
                         calibration, 
                         datafreq
                         )
#-------------------------------------------------------------------------------
# Conduct portfolio backtests
#-------------------------------------------------------------------------------
"""
Assign the parameters to be used in the regime-switching models 
- rsfeatures: list of the features to be used to fit the HMM
- n_states: numer of states to which to fit the HMM
"""
rsfeatures = ['Mkt-RF']
n_states:int = 2

"""
models: dictionary of the models to be used in the backtest. Each model is
identified by a key, which is also serves as the name of the portfolio.
For each entry in the dict, the value is a list with two elements.
- First element: investment strategy to use for portfolio construction
    - MinVar: minimum variance portfolio
    - MVO: mean-variance portfolio
    - RiskParity: risk parity portfolio
- Second element: flag indicating whether to use a regime-switching factor model
for parameter estimation
    - 0: no regime-switching
    - 1: regime-switching

constraints: List of constraints to be used in the optimization models. 
    Note: Currently, only the 'longonly' constraint is available. Removing 
    this constraint will allow for short selling.
"""
models = {'minvar': ['MinVar', 0],
          'rsminvar': ['MinVar', 1],
          'mvo': ['MVO', 0],
          'rsmvo': ['MVO', 1],
          'rp': ['RiskParity', 0],
          'rsrp': ['RiskParity', 1]
          }
constraints = ['longonly']
portfolios = []
for m in models:
    portfolios.append(po.Portfolio(m, 
                                   data.assets, 
                                   data.features,
                                   models[m][0],
                                   models[m][1],
                                   constraints)
                                   )
po.backtest(portfolios,
            data,
            daterange,
            lookback=lookback,
            rebalfreq=rebalfreq,
            rsfeatures=rsfeatures,
            n_states=n_states
            )
                
#-------------------------------------------------------------------------------
# Results
#-------------------------------------------------------------------------------
"""
Plot the wealth evolution and rolling Sharpe ratios of all portfolios
"""
wealth = [portfolios[i].wealth for i in range(len(portfolios))]
wealth = pd.concat(wealth, axis=1)
wealth.plot()

sharpe = [portfolios[i].stats.roll_sharpe for i in range(len(portfolios))]
sharpe = pd.concat(sharpe, axis=1)
sharpe.plot()

results = pd.DataFrame([[portfolios[i].stats.mu[0] 
                         for i in range(len(portfolios))],
                        [portfolios[i].stats.vol[0] 
                         for i in range(len(portfolios))],
                        [portfolios[i].stats.sharpe[0] 
                         for i in range(len(portfolios))]
                       ],
                       index=['Mean', 'Volatility', 'Sharpe'],
                       columns=[portfolios[i].name 
                                for i in range(len(portfolios))]
                       )

################################################################################
# End