import numpy as np
import pandas as pd

## MODEL: X     ->   Z    <-      Y
##                  f_(xy)

SAMPS = 10000
fxy = lambda x, y: x + y

X = np.random.uniform(0,1, size=SAMPS)
Y = np.random.uniform(0,5,size=SAMPS)
Z = fxy(X,Y) + np.random.normal(0,.1, size=SAMPS)

X = (X< np.median(X))+1
Y = (Y< np.median(Y))+1
Z = (Z< np.median(Z))+1

data_dict = {'X': X, 'Y': Y, 'Z': Z}
gen_data = pd.DataFrame(data_dict)
gen_data.to_csv("simple_collider_testset.csv")
