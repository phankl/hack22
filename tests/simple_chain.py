import numpy as np
import pandas as pd

## MODEL: X     ->   Y    ->      Z
##              f_x       f_Y

SAMPS = 10000

fx = lambda x: 0.1*x
fy = lambda y: 1 + y

X = np.random.uniform(0,1, size=SAMPS)
Y = fx(X) + np.random.normal(0,.1,size=(SAMPS))
Z = fy(Y) + np.random.normal(0,.1, size=(SAMPS))

## Split into two classes with roughly equal occupancy

X = (X < np.median(X)) + 1
Y = (Y < np.median(Y)) + 1
Z = (Z < np.median(Z)) + 1

data_dict = {'X': X, 'Y': Y, 'Z': Z}
gen_data = pd.DataFrame(data_dict)
gen_data.to_csv("simple_chain_testset.csv")
