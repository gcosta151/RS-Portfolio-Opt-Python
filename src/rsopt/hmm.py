# hmm library
#
# Prepared by:    Giorgio Costa
# Last revision:  11-Apr-2024
#
#===============================================================================
# Import packages
#===============================================================================
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

#===============================================================================
# Class HMM
#===============================================================================
class HMM:
    """HMM object
    Apply the Baum-Welch expectation-maximization algorithm to fit a HMM 
    to the given timeseries of factor returns

    Inputs
    ------ 
    frets: pd.DataFrame
        T x M Timeseries of feature returns for M features and T observations
    n_states: int
        Number of states to which to fit the HMM
    num_iters: int
        Maximum number of iterations to be performed. The algorithm will stop 
        early if it converges

    Attributes
    -------
    rsfeatures: list
        List of feature names
    n_features: int
        Number of features
    n_states: int
        Number of states
    n_obs: int
        Number of observations
    A: np.Array
        S x S transition probability matrix
    state: int
        The esitmated state at the end of the current time period
    prob: np.Array
        The transition probabilities of the current state
    gamma: pd.DataFrame
        T x S array of smoothed probabilities for S states with T 
        observations. The smoothed probabilities indicate the probability of
        being in state s for s = 1, ..., S.

    Methods
    -------
    fit_hmm(frets, n_iters):
        Apply the Baum-Welch expectation-maximization algorithm
    exp_step(frets, pi, A, mu, sigma):
        Expectation step in the Baum-Welch algorithm
    max_step(frets, gamma, xi):
        Maximization step in the Baum-Welch algorithm
    """
    def __init__(self, 
                 frets:pd.DataFrame, 
                 n_states:int, 
                 n_iters:int=150
                 ):
        self.rsfeatures = frets.columns.to_list()
        self.n_features = frets.shape[1]
        self.n_states = n_states
        self.n_obs = frets.shape[0]

        # Run Baum-Welch (EM) algorithm
        self.fit_hmm(frets, n_iters)

    #--------------------------------------------------------------------------
    # Method fit_hmm
    #--------------------------------------------------------------------------
    def fit_hmm(self, 
                frets:pd.DataFrame, 
                n_iters:int
                ):
        """Apply the Baum-Welch expectation-maximization algorithm

        Inputs
        ------ 
        frets: pd.DataFrame
            T x M Timeseries of feature returns for M features and T
        num_iters: int
            Maximum number of iterations to be performed. The algorithm will 
            stop early if it converges

        Outputs
        -------
        gamma: pd.DataFrame
            T x S array of smoothed probabilities for S states with T 
            observations. The smoothed probabilities indicate the probability
            of being in state s for s = 1, ..., S. 
        A: np.Array
            S x S transition probability matrix 
        state: int
            The esitmated state at the end of the current time period
        prob: np.Array
            The transition probabilities of the current state
        """
        frets_idx = frets.index
        frets = frets.to_numpy()

        # Initialize random parameters for the HMM
        A = np.random.rand(self.n_states, self.n_states) + 0.1
        A /= A.sum(axis=1, keepdims=True)
        pi = np.random.rand(self.n_states) + 0.1
        pi /= np.sum(pi)

        # Initialize the mean and variance
        mu = [np.mean(frets, axis=0) / (0.5 + np.random.rand()) 
              for i in range(self.n_states)
              ]
        sigma = [(np.cov(frets, rowvar=0) / (0.5 + np.random.rand()))
                 for i in range(self.n_states)
                 ]
        
        # Run Baum-Welch (EM) algorithm
        for i in range(n_iters):
            # Expectation step       
            gamma, xi = self.exp_step(frets, pi, A, mu, sigma)
            
            # Maximization step
            pi, A, mu, sigma = self.max_step(frets, gamma, xi)
            
            # Check for convergence
            if i > 2:
                if np.linalg.norm(A - A_prev) < 5e-6 and \
                    np.linalg.norm(pi - pi_prev) < 5e-6 and \
                    np.linalg.norm(np.array(mu) - np.array(mu_prev)) < 5e-6 and \
                    np.linalg.norm(np.array(sigma) - np.array(sigma_prev)) < 5e-6:
                    break
                elif i == 99:
                    print("HMM did not converge, consider increasing the \
                          default number of iterations")
            
            # Save previous parameters for convergence check
            A_prev = A.copy()
            pi_prev = pi.copy()
            mu_prev = mu.copy()
            sigma_prev = sigma.copy()

        # Sort such that the first state has the lowest volatility
        idx = np.argsort([np.trace(sigma[i]) for i in range(self.n_states)])
        gamma = gamma[:, idx]
        self.A = A[idx, :]

        # Estimated current regime and its transition probabilities
        self.state = np.argmax(gamma[-1,:])
        self.prob = A[self.state,:]

        self.gamma = pd.DataFrame(gamma, 
                                columns=["State " + str(i) 
                                        for i in range(self.n_states)],
                                index=frets_idx)
        
    #--------------------------------------------------------------------------
    # Method exp_step
    #--------------------------------------------------------------------------
    def exp_step(self, 
                 frets:np.array, 
                 pi:np.array, 
                 A:np.array, 
                 mu:list, 
                 sigma:list
                 ):
        """Expectation step (e-step) in the Baum-Welch algorithm
        The e-step computes the expected values of the hidden state parameters.
        It estiamtes the forward (fwd) and backward (bwd) probabilities. These 
        probabilities provide estimates of the likelihood of being in a 
        particular hidden state at each time step given the observed data.

        Inputs
        ------
        frets: np.array
            T x M timeseries of feature returns for M features and T 
            observations
        pi: np.array
            S x 1 vector of probabilities defining the initial state 
            distribution
        A: np.array
            S x S transition probability matrix
        mu: list
            List with S elements vector where each element is a M x 1 vector of 
            expected feature returns
        sigma: list
            List with S elements where each element is a M x M feature 
            covariance matrix

        Outputs
        -------
        gamma: np.array
            T x S Timeseries of smoothed (posterior) probabilities for S 
            states and T observations. The smoothed probabilities indicate the 
            probability of being in state s for s = 1, ..., S
        xi: np.array
            M x M x T-1 multidimensional array of joint probabilities at each 
            point in time
        """
        fwd = np.zeros((self.n_obs, self.n_states))
        bwd = np.zeros((self.n_obs, self.n_states))
        gamma = np.zeros((self.n_obs, self.n_states))
        xi = np.zeros((self.n_states, self.n_states, self.n_obs-1))
        density = np.zeros((self.n_obs, self.n_states))

        # PDF of observed data given the current estimates of mu and sigma
        for i in range(self.n_states):
            mvn = multivariate_normal(mean=mu[i], cov=sigma[i])
            density[:,i] = mvn.pdf(frets)

        # Forward pass
        fwd[0,:] = pi * density[0,:]
        fwd[0,:] /= np.sum(fwd[0,:])
        for t in range(1, self.n_obs):
            fwd[t,:] = density[t,:] * np.dot(A.T, fwd[t-1,:])
            fwd[t,:] /= np.sum(fwd[t,:])

        # Backward pass
        bwd[self.n_obs-1,:] = density[self.n_obs-1,:]
        bwd[self.n_obs-1,:] /= np.sum(bwd[self.n_obs-1,:])
        for t in range(self.n_obs-2, -1, -1):
            bwd[t,:] = density[t+1,:] * np.dot(A, bwd[t+1,:])
            bwd[t,:] /= np.sum(bwd[t,:])

        # Compute gamma and xi
        for t in range(self.n_obs):
            gamma[t,:] = fwd[t,:] * bwd[t,:]
            gamma[t,:] /= np.sum(gamma[t,:])
            if t < self.n_obs-1:
                xi[:,:,t] = A * np.outer(fwd[t,:], bwd[t+1,:] * density[t+1,:])
                xi[:,:,t] /= np.sum(xi[:,:,t])

        return gamma, xi
            
    #--------------------------------------------------------------------------
    # Function max_step
    #--------------------------------------------------------------------------
    def max_step(self, 
                 frets:np.array, 
                 gamma:np.array, 
                 xi:np.array
                 ):
        """Maximization step (m-step) in the Baum-Welch algorithm
        The m-step updates the model parameters based on the estimated 
        expectations arising from the e-step. The m-step aims to maximize the 
        likelihood of the observed data. 
        
        Inputs
        ------
        frets: np.array
            T x M timeseries of feature returns for M features and T 
            observations
        gamma: np.array
            T x S Timeseries of smoothed (posterior) probabilities for S 
            states and T observations. The smoothed probabilities indicate  
            the probability of being in state s for s = 1, ..., S
        xi: np.array
            M x M x T-1 multidimensional array of joint probabilities
            at each point in time

        Outputs
        -------
        pi: np.array
            S x 1 vector of probabilities defining the initial state 
            distribution
        A: np.array
            S x S transition probability matrix 
        mu: np.array
            List of S elements where each element is a M x 1 vector of expected 
            feature returns
        sigma: list
            List of S elements where each element is a M x M feature covariance 
            matrix
        """
        A = np.zeros((self.n_states, self.n_states))
        mu, sigma = [], []

        # Update initial state distribution
        pi = gamma[0,:]

        for i in range(self.n_states):
            # Update transition matrix
            A[i,:] = np.sum(xi[i,:,:], axis=1)
            A[i,:] /= np.sum(A[i,:])

            # Update mean and covariance matrix for each state
            mu.append(np.sum(gamma[:,i].reshape(-1,1) * frets, axis=0) / 
                        np.sum(gamma[:,i])
                        )
            diff = frets - mu[i]
            sigma.append((gamma[:,i].reshape(-1,1) * diff).T @ diff / 
                            np.sum(gamma[:,i])
                        )

        return pi, A, mu, sigma
    
################################################################################
# End