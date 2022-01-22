import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

## Constants

MIN_DATASET_SIZE=1000
P_VALUE_CRITERION = 0.01
FILEPATH = "data/2005-tfldata-accidents.csv" # Note this is a local path - must change for server (tekeh)
TEST_FEATURES_TFL = ['Accident Severity', 'Road Type', 'Time', 'Light Conditions (Banded)', 'Road Surface', 'Weather'] ## The subset we will build the model on
DAYLIGHT_HORIZON = 1800

# Functions

def is_independent(dat, X, Y, Z=None, categorical=True):
    """
    Checks whether X|Z is independent of Y. Z is a set of covariates, whereas X, Y are single covariates
    dat : pandas dataframe
    X : column label, type == list
    Y : column label, type == list
    Z : column labels (1 for now, but will generalise), type == list

    Works for categorical data at the moment - continuous data is even easier
    """
    ## Construct contingency table and work out nominal dependence 
    res_list = []
    pvalues_list = []
    if Z is None:
        ## Testing direct dependence
        #tab = pd.crosstab(dat[X], dat[Y])
        table = sm.stats.Table.from_data(dat[X+Y]) 
        res = table.test_nominal_association()
        pval = res.pvalue

    else:
        ## Here e deal with the set Z not empty. Take const Z cuts of the data and look for indep.
        ## Create new column which is a tupled version of the control variables
        
        dat['Combined_Controls'] = list(zip(*[dat[x] for x in Z]))
        controlled_categories = dat['Combined_Controls'].unique()
        for cat in controlled_categories:
            #print(cat)
            try:
                dat_subset = dat[X+Y][dat['Combined_Controls'] == cat]
                if dat_subset.shape[0] < MIN_DATASET_SIZE:
                    continue
                table = sm.stats.Table.from_data( dat_subset ) 
                res = table.test_nominal_association()
                pvalues_list.append(res.pvalue)
                res_list.append(res)
            except:
                continue
        #print(pvalues_list)
        pval = max(pvalues_list) ## crude. Prob of observing assuming null hyp (two variables are independent)
    
    #print(pval)
    if pval < P_VALUE_CRITERION: ## 5% significance, say. Reject the null 
        #print(f"{X} | {Z} depends on {Y}")
        return 1, res_list, pvalues_list
    else:
        #print(f"{X} | {Z} does not depend on {Y}")
        return 0, res_list, pvalues_list

def tfl_preprocess(dat):
    """
    Helper function to preprocess TfL data fields
    """
    ## Time field
    dat.Time = pd.to_numeric(dat.Time.str[1:]) # postprocess for tfl data
    x = dat.Time < DAYLIGHT_HORIZON
    dat.Time = x.map({True:1, False:2})
    return dat ## for success

def mtg_preprocess(dat):
    dat.prices = [x['usd'] for x in dat.prices]
    return dat

if __name__ == "__main__":

    #Y = ['Road Surface']
    #X = ['Accident Severity']
    #Z = ['Weather']
    #bit, res_list, p_list = is_independent(dat, X, Y, Z)
    
    ## Testset TFL
    dat = pd.read_csv(FILEPATH)[TEST_FEATURES_TFL] ## truncated set
    dat = tfl_preprocess(dat)
    ## Test: Do all possible combs (with |Z| = 1) to see the results and intuitively evaluate
    from itertools import product, combinations
    for X, Y, Z in combinations(TEST_FEATURES_TFL, r=3):
        #print(X,Y,Z)
        bit, res_list, p_list = is_independent(dat, [X], [Y], [Z])

    ## Testet MTG
    ## Test
