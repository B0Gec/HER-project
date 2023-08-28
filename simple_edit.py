import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pysindy as ps
import warnings
warnings.filterwarnings("ignore")

import sys, pysindy 
print(pysindy.__version__, sys.version)

from loadit import loaded, fnamei, fnameii, fnameiii


df, cols = loaded(fnamei)
print(df)


t = np.linspace(0, 1, 100)
t = np.array(df['t'])
x = np.array(df['total'])
y = np.array(df['Ni-'])
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y

threshold = 0.1
model = ps.SINDy(feature_names=["x", "y"],
                optimizer = ps.STLSQ(threshold=threshold))

print(warnings.filters)
model.fit(X, t=t)

model.print()