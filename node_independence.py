import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm

def is_independent(dat, X, Y, Z=None):
    """
    Checks whether X|Z is independent of Y. Z is a set of covariates, whereas X, Y are single covariates
    dat : pandas dataframe
    X : column label
    Y : column label
    Z : column labels (1 for now, but will generalise)
    Works for categorical data at the moment
    """
    ## do the linear regression : STATSMODELS
    
    ## Construct contingency table and work out nominal dependence 
    if Z is None:
        ## Testing direct dependence
        tab = pd.crosstab(dat[X], dat[Y])
        table = sm.stats.Table(tab)
        res = table.test_nominal_association()
        pval = res.pvalue

    else:
        ## In this branch we deal with the set Z not empty. Easiest way is to take some const Z cut of the data
        controlled_categories = dat[Z].unique()
        pvalues_list = []
        for cat in controlled_categories:
            print(cat)
            xdat = dat[X][dat[Z] == cat]
            ydat = dat[Y][dat[Z] == cat]
            try:
                tab = pd.crosstab(xdat, ydat)
                table = sm.stats.Table(tab)
                res = table.test_nominal_association()
                pvalues_list.append(res.pvalue)
            except:
                continue
        print(pvalues_list)
        pval = max(pvalues_list) ## crude, but should work
    
    print(pval)
    if pval < P_VALUE_CRITERION: ## 5% significance, say
        return 1
    else:
        return 0


if __name__ == "__main__":
    P_VALUE_CRITERION = 0.05
    FILEPATH = "data/2005-tfldata-accidents.csv" # Note this is a local path - must change for server (tekeh)
    TEST_FEATURES = ['Accident Severity', 'Road Type', 'Time', 'Light Conditions (Banded)', 'Road Surface', 'Weather'] ## The subset we will build the model on

    dat = pd.read_csv(FILEPATH)[TEST_FEATURES] ## truncated set
    X, Y, Z = ['Accident Severity'], ['Road Surface'], ['Weather']
    #is_independent(dat, X[0], Y[0])
    is_independent(dat, X[0], Y[0], Z[0])


    #dat = dat[X+Z]
    #dat[X] = pd.factorize(dat[X])
    #dat[Z] = pd.factorize(dat[Z])

    #reg_y, enc_y = pd.factorize(dat[X])
    #reg_X1, enc_X1 = pd.factorize(dat[Y])
    #reg_X2, enc_X2 = pd.factorize(dat[Z])



    #reg_y = reg_y.reshape(-1,1)
    #reg_X1 = reg_X1.reshape(-1,1)
    #reg_X2 = reg_X2.reshape(-1,1)
    #reg_X = np.hstack((reg_X1, reg_X2))
    #reg_model = sm.OLS(reg_y, reg_X)
    #results = reg_model.fit()
    #print(results.summary())
