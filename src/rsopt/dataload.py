# dataload library
#
# Prepared by:    Giorgio Costa
# Last revision:  11-Apr-2024
#
#===============================================================================
# Import packages
#===============================================================================
import pandas as pd
from pandas_datareader import get_data_famafrench as get_ff
import os

#===============================================================================
# Class HistoricalData
#===============================================================================
class HistoricalData:
    """HistoricalData object
    Load historical data from Kenneth French's data library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    
    Inputs
    ------
    sourcedata: str
        Name of the dataset to load
    daterange: list
        Start and end dates for the data
    calibration: int
        Number of years required to calibrate the HMM
    freq: str
        Frequency of the data (daily, weekly, monthly)
    use_cache: bool
        Boolean indicating whether to use cached data
    save_results: bool
        Boolean indicating whether to save the results to cache
    
    Attributes
    ----------
    arets: pd.DataFrame
        T x N array of asset returns for N assets and T observations
    frets: pd.DataFrame
        T x M array of factor returns for M factors and T observations
    riskfree: pd.Series
        T x 1 array of risk-free returns
    assets: list
        List of asset names
    features: list
        List of feature names

    Methods
    -------
    load_data(cache_path, use_cache, save_results)
    """
    def __init__(self,
                 sourcedata:str,
                 daterange:list, 
                 calibration:int, 
                 freq:str='weekly', 
                 use_cache:bool=True, 
                 save_results:bool=True
                 ):
        start = daterange[0] - calibration
        end = daterange[1]
        self.sourcedata = sourcedata
        self.daterange = daterange
        self.calibration = calibration
        self.start = start
        self.end = end
        self.freq = freq
        cache_path = './cache/'+sourcedata+'_'+freq+'.pkl'
        self.load_data(cache_path, use_cache, save_results)

    #--------------------------------------------------------------------------
    # Method load_data
    #--------------------------------------------------------------------------
    def load_data(self, 
                  cache_path:str, 
                  use_cache:bool, 
                  save_results:bool
                  ):
        """Load data from Kenneth French's data library
        
        Inputs
        ------
        cache_path: str 
            Path to the cache data file
        use_cache: bool
            Boolean indicating whether to use cached data
        save_results: bool
            Boolean indicating whether to save the results to cache

        Outputs
        ----------
        Appends the following attributes to the object:
        arets: pd.DataFrame 
            T x N array of asset returns for N assets and T observations
        frets: pd.DataFrame
            T x M array of factor returns for M factors and T observations
        riskfree: pd.Series
            T x 1 array of risk-free returns
        assets: list
            List of asset names
        features: list
            List of feature names
        """
        if use_cache and os.path.exists(cache_path):
            frets = pd.read_pickle('./cache/factor_'+self.freq+'.pkl')
            arets = pd.read_pickle(cache_path)
        else:
            dl_freq = '_daily'

            # Get asset data
            arets = get_ff(self.sourcedata+dl_freq, 
                           start=self.start, 
                           end=self.end
                           )[0] / 100
            # Get factor data 
            ff5 = get_ff('F-F_Research_Data_5_Factors_2x3'+dl_freq, 
                         start=self.start,
                         end=self.end
                         )[0]
            momentum = get_ff('F-F_Momentum_Factor'+dl_freq, 
                              start=self.start, 
                              end=self.end
                              )[0]
            shortterm = get_ff('F-F_ST_Reversal_Factor'+dl_freq, 
                               start=self.start, 
                               end=self.end
                               )[0]
            longterm = get_ff('F-F_LT_Reversal_Factor'+dl_freq, 
                              start=self.start, 
                              end=self.end
                              )[0]
            frets = pd.concat([ff5, 
                               momentum, 
                               shortterm, 
                               longterm
                               ], axis=1) / 100
            riskfree = frets['RF']

            # Adjust asset returns for risk-free rate
            arets = arets.sub(riskfree, axis=0)
            
            if self.freq == 'weekly' or self.freq == '_weekly':
                # Convert daily returns to weekly returns
                arets = arets.resample('W-FRI').agg(lambda x: 
                                                    (x + 1).prod() - 1
                                                    )
                frets = frets.resample('W-FRI').agg(lambda x: 
                                                    (x + 1).prod() - 1
                                                    )
            elif self.freq == 'monthly' or self.freq == '_monthly':
                # Convert daily returns to monthly returns
                arets = arets.resample('M').agg(lambda x: 
                                                (x + 1).prod() - 1
                                                )
                frets = frets.resample('M').agg(lambda x: 
                                                (x + 1).prod() - 1
                                                )
            if save_results:
                frets.to_pickle('./cache/factor_'+self.freq+'.pkl')
                arets.to_pickle(cache_path)
        
        self.arets = arets
        self.riskfree = frets['RF']
        self.frets = frets.drop(['RF'], axis=1)
        self.assets = [x.strip(' ') for x in arets.columns.to_list()]
        self.features = [x.strip(' ') for x in frets.columns.to_list()]

################################################################################
# End