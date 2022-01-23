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
    Z : column labels (1 for now, but will generalise), type == list, optional
    categorical: bool, optional

    Works for categorical data at the moment - continuous data is even easier
    """
    ## Construct contingency table and work out nominal dependence 
    res_list = []
    pvalues_list = []
    if Z is None:
        ## Testing direct dependence
        #tab = pd.crosstab(dat[X], dat[Y])
        table = sm.stats.Table.from_data(dat[X+Y]) 
        table, status_code = remove_low_occupancy_cells(table)
        if status_code == 0:
            print(f"Association can not be determined. Likely too many gaps in the data table")
            return -1, [], []
        #print(table.table_orig.shape)
        if table.table_orig.shape[0] < 2 or table.table_orig.shape[1] < 2:
            print(f"Association can not be determined. Likely too many gaps in the data table")
            return -1, [], []
        #print(table.table_orig.shape)
        if np.squeeze(table.table_orig).ndim == 1:
            print(f"Association can not be determined. Likely too many gaps in the data table")
            return -1, [], []
        res = table.test_nominal_association()
        pval = res.pvalue

    else:
        ## Here e deal with the set Z not empty. Take const Z cuts of the data and look for indep.
        ## Create new column which is a tupled version of the control variables
        
        dat['Combined_Controls'] = list(zip(*[dat[x] for x in Z]))
        controlled_categories = dat['Combined_Controls'].unique()
        for cat in controlled_categories:
            #print(cat)
            
            dat_subset = dat[X+Y][dat['Combined_Controls'] == cat]
            if dat_subset.shape[0] < MIN_DATASET_SIZE:
                continue
            table = sm.stats.Table.from_data( dat_subset ) 
            table, status_code = remove_low_occupancy_cells(table)
            
            if status_code == 0:
                print(f"Association can not be determined. Likely too many gaps in the data table")
                return -1, [], []
            #print(table.table_orig.shape)
            if table.table_orig.shape[0] < 2 or table.table_orig.shape[1]< 2:
                print(f"Association can not be determined. Likely too many gaps in the data table")
                return -1, [], []
            if np.squeeze(table.table_orig).ndim == 1:
                print(f"Association can not be determined. Likely too many gaps in the data table")
                return -1, [], []
            res = table.test_nominal_association()
            pvalues_list.append(res.pvalue)
            res_list.append(res)
            #except:
            #    continue
        #print(pvalues_list)
        pval = max(pvalues_list) ## crude. Prob of observing assuming null hyp (two variables are independent)
    
    #print(pval)
    if pval < P_VALUE_CRITERION: ## 5% significance, say. Reject the null 
        print(f"{X} | {Z} is associated with {Y} (likely dependence, p={pval})")
        return False, res_list, pvalues_list
    else:
        print(f"{X} | {Z} is not associated with {Y} (likely independence, p={pval})")
        return True, res_list, pvalues_list

def remove_low_occupancy_cells(table, min_occupancy=5):
    """
    Removes cells in frequency tables with low occupancy as this screws up the independene test.  is often qutoed as the best min occcupancy.

    table: statsmodel table object
    min_occupancy: int, optional

    Returns:
    table
    status code: 1 okay, 0 failed
    """

    freq_df = table.table_orig.copy()
    freq_df[freq_df < min_occupancy] = np.nan
    ## Now delete rows and columns as appropriate
    ## Greedy algo
    while np.any(freq_df.isna()):
        ## If it becomes 1d, then the chi2 makes no sense

        row_count = freq_df.isna().sum(axis=0)
        col_count = freq_df.isna().sum(axis=1)

        row_idxmax = row_count.idxmax()
        col_idxmax = col_count.idxmax()

        try:
            if row_count[row_idxmax] < col_count[col_idxmax]:
                freq_df.drop(inplace=True, axis=0, labels=col_idxmax)
        except:
            return sm.stats.Table(freq_df)

        try:
            if row_count[row_idxmax] >= col_count[col_idxmax]:
                freq_df.drop(inplace=True, axis=1, labels=row_idxmax)
        except:
            return sm.stats.Table(freq_df)

        #print(freq_df)
    try: 
        table_new = sm.stats.Table(freq_df)
        return table_new, 1
    except:
        ## Empty df
        return table, 0


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
    ## Testset TFL
    dat = pd.read_csv(FILEPATH)[TEST_FEATURES_TFL] ## truncated set
    dat = tfl_preprocess(dat)
    ## Test: Do all possible combs (with |Z| = 1) to see the results and intuitively evaluate
    from itertools import product, combinations, permutations
    for X, Y in combinations(TEST_FEATURES_TFL, r=2):
        #print(X,Y,Z)
        #bit, res_list, p_list = is_independent(dat, [X], [Y], [Z])
        bit, res_list, p_list = is_independent(dat, [X], [Y])


    for X, Y, Z in permutations(TEST_FEATURES_TFL, r=3):
        #print(X,Y,Z)
        #bit, res_list, p_list = is_independent(dat, [X], [Y], [Z])
        bit, res_list, p_list = is_independent(dat, [X], [Y], [Z])

    ## Testet MTG
    ## Test
