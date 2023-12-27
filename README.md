## Portfolio Optimization under a Markov Regime-Switching Framework ##
 This repository contains the source code to conduct numerical experiments similar to those presented in the following two papers:
 - [Costa, G. and Kwon, R. H. (2019). Risk parity portfolio optimization under a Markov regime-switching framework. Quantitative Finance, 19(3), 453-47](https://www.tandfonline.com/doi/abs/10.1080/14697688.2018.1486036?journalCode=rquf20)
    - [Link to PDF](https://www.researchgate.net/profile/Giorgio-Costa-2/publication/326756996_Risk_parity_portfolio_optimization_under_a_Markov_regime-switching_framework/links/5e0992d74585159aa4a47d19/Risk-parity-portfolio-optimization-under-a-Markov-regime-switching-framework.pdf)
 - [Costa, G. and Kwon, R. H. (2020). A regime-switching factor model for mean–variance optimization. Journal of Risk, 22(4), 31-59](https://www.risk.net/journal-of-operational-risk/7535001/a-regime-switching-factor-model-for-mean-variance-optimization)
    - [Link to PDF](https://www.researchgate.net/profile/Giorgio-Costa-2/publication/341752309_A_Regime-Switching_Factor_Model_for_Mean-Variance_Optimization/links/61ddd756323a2268f9997b5f/A-Regime-Switching-Factor-Model-for-Mean-Variance-Optimization.pdf)
 
The work in these two papers pertains to a Markov regime-switching factor model that captures the cyclical nature of asset returns in modern financial markets. Maintaining a factor model structure allows us to easily derive the first two moments of the asset return distribution: the expected returns and covariance matrix. By design, these two parameters are calibrated under the assumption of having distinct market regimes. In turn, these regime-dependent parameters serve as the inputs during portfolio optimization, thereby constructing portfolios adapted to the current market environment. The proposed framework leads to a computationally tractable portfolio optimization problem, meaning we can construct large, realistic portfolios. 

## Dependencies ##
- Python v3.x
- pandas v2.0.x
- numpy v1.24.x
- scipy v1.11.x
- cvxpy v1.4.x
- tqdm v4.65.x
- pandas_datareader v0.10.x

## Usage ##
This repository contains all the files used to necessary to run the numerical experiments of the regime-switching portfolio optimization framework. To run the experiments. please refer to the main.py file. Anyone wishing to make any changes to the models may do so by tinkering with a copy of the code base. The code base is composed of the following files:
- dataload.py: Module to download data from [Kenneth French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Use this module to download feature returns from the Fama-French model, as well as the Industry Portfolios to serve as the asset returns.
- portfolio.py: Library with the Portfolio object, as well as the backtest function to evaluate multiple competing portfolios over time. 
- optimization.py: Library to construct optimal nominal and regime-switching portfolios. Three portfolio optimization models are currently available for use. The optimization models can be used to construct both nominal and regime-switching portfolios. 
   - mvo: Mean-variance optimization
   - minvar: Minimum variance optimization
   - rp: Risk parity portfolio optimization
- hmm.py: Library with the HMM object. The HMM object takes in time series data and implements the Baum-Welch algorithm to fit a hidden Markov model. 
- factormodel.py: Library with the FactorModel object. The FactorModel object is calibrated using linear regression (OLS) under both a single regime, as well as under the assumption of multiple regimes. 

## Licensing ##
Unless otherwise stated, the source code is copyright of Giorgio Costa and licensed under the Apache 2.0 License.
