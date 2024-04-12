# optimization library
#
# Prepared by:    Giorgio Costa
# Last revision:  11-Apr-2024
#
#===============================================================================
# Import packages
#===============================================================================
import cvxpy as cp
import numpy as np

#===============================================================================
# Class minvar
#===============================================================================
class MinVar:
    """Minimum variance optimization model 

        min   w' * sigma * w 
        s.t.  sum(w) = 1
              w >= 0 (if long-only constraint selected)

        Variables:
            w: n x 1 vector of portfolio weights

        Parameters:
            sigma: n x n asset covariance matrix
    
    Inputs
    ------
    regimeswitching: bool 
        Boolean indicating whether to use regime switching
    constraints: list
        Set of constraints to impose on the optimization model
    n_assets: int
        Number of assets in the portfolio
    solver: str
        Optimization solver to use

    Attributes
    ----------
    regimeswitching: bool
        Boolean indicating whether a regime switching model was used
    solver: str
        Optimization solver used
    sqrt_sigma: cvxpy.Parameter
        Square root of the covariance matrix
    w: cvxpy.Variable
        Portfolio weights
    z: cvxpy.Variable
        Auxiliary variable
    problem: cvxpy.Problem
        Optimization problem

    Methods
    -------
    optimize(fmodel)
        Solve the optimization model
    """
    def __init__(self, 
                 regimeswitching:bool, 
                 constraints:list, 
                 n_assets:int, 
                 solver:str
                 ):
        self.regimeswitching = regimeswitching
        self.solver = solver
        self.sqrt_sigma = cp.Parameter((n_assets, n_assets))
        if 'longonly' in constraints:
            self.w = cp.Variable(n_assets, nonneg=True)
        else:
            self.w = cp.Variable(n_assets)
        self.z = cp.Variable(nonneg=True)
        constraints = [cp.SOC(self.z, self.sqrt_sigma @ self.w),
                       cp.sum(self.w) == 1
                       ]
        objective = cp.Minimize(self.z)
        self.problem = cp.Problem(objective, constraints)

    #--------------------------------------------------------------------------
    # Method optimize
    #--------------------------------------------------------------------------
    def optimize(self, fmodel):
        """Solve the optimization model
        
        Inputs
        ------
        fmodel: fm.FactorModel
            Factor model object. This object includes the covariance matrix and 
            the mean returns for the assets and features

        Outputs
        -------
        w: np.array
            n x 1 vector of portfolio weights
        """
        sigma = fmodel.sigma[self.regimeswitching]
        self.sqrt_sigma.value = np.linalg.cholesky(sigma).T
        self.problem.solve(solver=self.solver)
        return self.w.value
    
#===============================================================================
# Class MVO
#===============================================================================
class MVO:
    """Mean-Variance optimization model 

        min   w' * sigma * w 
        s.t.  sum(w) = 1
              mu' * w >= 1.05 * mean(mu)
              w >= 0 (if long-only constraint selected)

        Variables:
            w: n x 1 vector of portfolio weights

        Parameters:
            sigma: n x n asset covariance matrix
            mu: n x 1 vector of asset mean returns
    
    Inputs
    ------
    regimeswitching: bool 
        Boolean indicating whether to use regime switching
    constraints: list
        Set of constraints to impose on the optimization model
    n_assets: int
        Number of assets in the portfolio
    solver: str
        Optimization solver to use

    Attributes
    ----------
    regimeswitching: bool
        Boolean indicating whether a regime switching model was used
    solver: str
        Optimization solver used
    sqrt_sigma: cvxpy.Parameter
        Square root of the covariance matrix
    mu: cvxpy.Parameter
        Mean returns
    w: cvxpy.Variable
        Portfolio weights
    z: cvxpy.Variable
        Auxiliary variable
    problem: cvxpy.Problem
        Optimization problem

    Methods
    -------
    optimize(fmodel)
        Solve the optimization model
    """
    def __init__(self, 
                 regimeswitching:bool, 
                 constraints:list, 
                 n_assets:int, 
                 solver:str
                 ):
        self.regimeswitching = regimeswitching
        self.solver = solver
        self.sqrt_sigma = cp.Parameter((n_assets, n_assets))
        self.mu = cp.Parameter(n_assets)
        if 'longonly' in constraints:
            self.w = cp.Variable(n_assets, nonneg=True)
        else:
            self.w = cp.Variable(n_assets)
        self.z = cp.Variable(nonneg=True)
        constraints = [cp.SOC(self.z, self.sqrt_sigma @ self.w),
                       cp.sum(self.w) == 1,
                       self.mu.T @ self.w >= 1.05 * cp.sum(self.mu) / n_assets
                       ]
        objective = cp.Minimize(self.z)
        self.problem = cp.Problem(objective, constraints)

    #--------------------------------------------------------------------------
    # Method optimize
    #--------------------------------------------------------------------------
    def optimize(self, fmodel):
        """Solve the optimization model
        
        Inputs
        ------
        fmodel: fm.FactorModel
            Factor model object. This object includes the covariance matrix and 
            the mean returns for the assets and features

        Outputs
        -------
        w: np.array
            n x 1 vector of portfolio weights
        """
        sigma = fmodel.sigma[self.regimeswitching]
        self.sqrt_sigma.value = np.linalg.cholesky(sigma).T
        self.mu.value = fmodel.mu[self.regimeswitching]
        self.problem.solve(solver=self.solver)
        return self.w.value

#===============================================================================
# Class RiskParity
#===============================================================================
class RiskParity:
    """Risk parity optimization model 

        min   w' * sigma * w - sum(log(w))
        s.t.  w >= 0 

        Variables:
            w: n x 1 vector of portfolio weights

        Parameters:
            sigma: n x n asset covariance matrix

        Note: The risk parity model assumes that positions can only be held 
            long. This assumption allows us to model risk parity as a convex 
            problem.

    Inputs
    ------
    regimeswitching: bool 
        Boolean indicating whether to use regime switching
    constraints: list
        Set of constraints to impose on the optimization model
    n_assets: int
        Number of assets in the portfolio
    solver: str
        Optimization solver to use

    Attributes
    ----------
    regimeswitching: bool
        Boolean indicating whether a regime switching model was used
    solver: str
        Optimization solver used
    sqrt_sigma: cvxpy.Parameter
        Square root of the covariance matrix
    w: cvxpy.Variable
        Portfolio weights
    z: cvxpy.Variable
        Auxiliary variable
    problem: cvxpy.Problem
        Optimization problem

    Methods
    -------
    optimize(fmodel)
        Solve the optimization model
    """
    def __init__(self, 
                 regimeswitching:bool, 
                 constraints:list, 
                 n_assets:int, 
                 solver:str
                 ):
        self.regimeswitching = regimeswitching
        self.solver = solver
        self.sqrt_sigma = cp.Parameter((n_assets, n_assets))
        self.w = cp.Variable(n_assets, nonneg=True)
        self.z = cp.Variable(nonneg=True)
        constraints = [cp.SOC(self.z, self.sqrt_sigma @ self.w)]
        objective = cp.Minimize(cp.power(self.z, 2) 
                                - cp.sum(cp.log(self.w))
                                )
        self.problem = cp.Problem(objective, constraints)

    #--------------------------------------------------------------------------
    # Method optimize
    #--------------------------------------------------------------------------
    def optimize(self, fmodel):
        """Solve the optimization model
        
        Inputs
        ------
        fmodel: fm.FactorModel
            Factor model object. This object includes the covariance matrix and 
            the mean returns for the assets and features

        Outputs
        -------
        w: np.array
            n x 1 vector of portfolio weights
        """
        sigma = fmodel.sigma[self.regimeswitching]
        self.sqrt_sigma.value = np.linalg.cholesky(sigma).T
        self.problem.solve(solver=self.solver)
        w = self.w.value / np.sum(self.w.value)
        return w 

################################################################################
# End