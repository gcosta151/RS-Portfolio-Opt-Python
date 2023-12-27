# factormodel library
#
# Prepared by:    Giorgio Costa
# Last revision:  25-Dec-2023
#
#-------------------------------------------------------------------------------
# Import packages
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------
# class FactorModel
#-------------------------------------------------------------------------------
class FactorModel:
    """FactorModel object
    Use a linear regression model to estimate the asset means and covariance 
    matrix. If provided, use a hidden Markov model to calibrate a linear 
    regression model per state.

    Inputs
    ------
    data: HistoricalData object containing the timeseries of asset and feature
        returns
    lookback: Number of observations to use to conduct the regression
    rsmodel (Optional): Object containing a T x S timeseries of smoothed 
        probabilities for S states and T observations, as well as the 
        transition probabilities conditional on the current estimated state

    Outputs
    -------
    FactorModel object with the following fields
    mu: list of N x 1 vectors of intercepts (regular and regime-switching)
    sigma: list of N x N covariance matrices (regular and regime-switching)
    """

    def __init__(self, data, lastdate, lookback, rsmodel=None):

        self.mu = []
        self.sigma = []

        mu, sigma = self.fit(data, lastdate, lookback)
        self.mu.append(mu)
        self.sigma.append(sigma)
        if rsmodel is not None:
            mu_rs, sigma_rs = self.fit_rs(data, lastdate, lookback, rsmodel)
            self.mu.append(mu_rs)
            self.sigma.append(sigma_rs)
        
    def fit(self, data, lastdate, lookback):
        """Construct a factor model by performing a linear regression
        The factor model is used to estimate the asset means and 
        covariance matrix

        Inputs
        ------
        data: HistoricalData object containing the timeseries of asset and 
            feature returns
        lastdate: End date for the regression sample
        lookback: Number of observations to use to conduct the regression

        Outputs
        -------
        mu: N x 1 vector of intercepts 
        sigma: N x N covariance matrix  
        """
        X = data.frets.loc[:lastdate, :]
        Y = data.arets.loc[:lastdate, :]
        if X.shape[0] > lookback:
            X = X.iloc[-lookback:, :]
            Y = Y.iloc[-lookback:, :]

        # Use linear regression for parameter estimation
        mu, sigma = linreg(X.to_numpy(), Y.to_numpy())

        return mu, sigma
    
    def fit_rs(self, data, lastdate, lookback, rsmodel):
        """Construct a factor model for each state in the regime-switching 
        model (rsmodel)
        1. For each state s, perform a linear regression and estimate mu 
            and sigma
        2. Combine the state-specific mu and sigma into a 
            probability-weighted mu and sigma using the transition 
            probabilities conditional on the estimated current state

        Inputs
        ------
        data: HistoricalData object containing the timeseries of asset and 
            feature returns
        lastdate: End date for the regression sample
        lookback: Number of observations to use to conduct the regression
        rsmodel: Object containing a T x S timeseries of smoothed
            probabilities for S states and T observations, as well as the 
            transition probabilities conditional on the current estimated 
            state

        Outputs
        -------
        mu: N x 1 vector of intercepts
        sigma: N x N covariance matrix            
        """
        
        # Use linear regression for parameter estimation per state
        X = data.frets.loc[:lastdate, :]
        Y = data.arets.loc[:lastdate, :]
        gamma = rsmodel.gamma.loc[X.index, :]
        mu = []
        sigma = []
        for s in range(rsmodel.n_states):
            Xs = X[gamma.iloc[:,s] >= 0.5]
            Ys = Y[gamma.iloc[:,s] >= 0.5]
            if Xs.shape[0] > lookback:
                Xs = Xs.iloc[-lookback:, :]
                Ys = Ys.iloc[-lookback:, :]

            mu_s, sigma_s = linreg(Xs.to_numpy(), Ys.to_numpy())
            mu.append(mu_s)
            sigma.append(sigma_s)

        # Probability-weighted asset covariance matrix and mean vector
        prob = rsmodel.prob
        sigma = sum((prob[s] * sigma[s]) 
                    + (prob[s] * np.outer(mu[s], mu[s])) 
                    - (prob[s] * sum(prob[t] * np.outer(mu[s], mu[t])
                                        for t in range(rsmodel.n_states))
                                        )
                    for s in range(rsmodel.n_states)
                    )
        mu = sum(prob[s] * mu[s] for s in range(rsmodel.n_states))

        return mu, sigma
        
#-------------------------------------------------------------------------------
# Function linreg
#-------------------------------------------------------------------------------
def linreg(X, Y):
    """Regress matrix of targets Y against feature array X
    Note: Features are centered (de-meaned) by default

    Inputs
    ------
    Y: T x N matrix with T observations and M targets 
    X: T x M matrix with T observations and N features

    Outputs
    -------
    mu: N x 1 vector of intercepts 
    sigma: N x N covariance matrix 
    """
    n_obs, n_features = X.shape

    # Center (de-mean) the factors
    X = X - np.mean(X, axis=0)
    X = np.concatenate((np.ones((n_obs, 1)), X), axis=1)

    # Compute the coefficients of an OLS regression
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # Calculate the OLS residuals
    epsilon = Y - (X @ beta)

    # Vector of intercepts (equal to the means if using centered factors)
    mu = beta[0, :]

    # Matrix of factor loadings
    V = beta[1:, :]

    # Factor covariance matrix
    F = np.cov(X[:, 1:], rowvar=False)

    # Diagonal matrix of residual variance
    ssq = np.sum(epsilon ** 2, axis=0) / (n_obs - n_features - 1)
    D = np.diag(ssq)

    # Asset covariance matrix
    sigma = V.T @ (F @ V) + D
    sigma = posdef(sigma)

    return mu, sigma

#-------------------------------------------------------------------------------
# Function posdef
#-------------------------------------------------------------------------------
def posdef(B):
    """Find the nearest positive definite matrix to the input matrix A

    Inputs
    ------
    B: Square matrix 
    
    Outputs
    -------
    B: Nearest positive semidefinite matrix to B
    """
    counter = 0
    while np.min(np.linalg.eigvals(B)) < 0:
        counter += 1
        U, S, V = np.linalg.svd(B)
        H = V @ np.diag(S) @ V.T
        B = (B + H) / 2
        B = (B + B.T) / 2

        if counter > 500:
            print('posdef ran out of iterations to approximate the nearest \
                  PSD matrix')
            break

    return B

################################################################################
# End