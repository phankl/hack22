import numpy as np
import pandas as pd

## MODEL: X     ->   Z    <-      Y
##                  f_(xy)

SAMPS = 10000

fxy = lambda x, y: x**2 + y

X = np.random.normal(0,5, size=SAMPS)
Y = np.random.normal(0,5,size=SAMPS)
Z = fxy(X,Y) + np.random.normal(0,1, size=SAMPS)

gen_data = pd.DataFrame([X, Y, Z])
gen_data.to_csv("simple_collider_testset.csv")
