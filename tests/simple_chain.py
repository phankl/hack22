import numpy as np
import pandas as pd

## MODEL: X     ->   Y    ->      Z
##              f_x       f_Y

SAMPS = 10000

fx = lambda x: x**2
fy = lambda y: 1 + y

X = np.random.normal(0,5, size=SAMPS)
Y = fx(X) + np.random.normal(0,1,size=(SAMPS))
Z = fy(Y) + np.random.normal(0,1, size=(SAMPS))

data_dict = {'X': X, 'Y': Y, 'Z': Z}
gen_data = pd.DataFrame(data_dict)
gen_data.to_csv("simple_chain_testset.csv")
