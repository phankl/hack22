import numpy as np
import pandas as pd

## MODEL: X     ->  Y
##              f_x

SAMPS = 100000

fx = lambda x: x**2

X = np.random.normal(0, 5, size=SAMPS)
Y = fx(X) + np.random.normal(0, 1, size=SAMPS)

data_dict = {'X': X, 'Y': Y}
gen_data = pd.DataFrame(data_dict)
gen_data.to_csv("simple_pair_testset.csv")
