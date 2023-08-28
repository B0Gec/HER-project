import numpy as np
import warnings
# warnings.filterwarnings("ignore")
import pysindy as ps
import datetime
import warnings
# warnings.filterwarnings("ignore")

# print(pysindy.__version__, sys.version)
# print(warnings.filters)

from loadit import loaded, fnamei, fnameii, fnameiii

fname = fnamei
df = loaded(fname)
# print(df)

# t = np.linspace(0, 1, 100)
t = np.array(df['t'])
# x = np.array(df['total'])
# y = np.array(df['Ni-'])
# simple:
# x = 3 * np.exp(-2 * t)
# y = 0.5 * np.exp(t)

# X = np.stack((x, y), axis=-1)  # First column is x, second is y


# scale = 35  # fail
scale = 25
scale = 30
scale = 32
scale = 33
scale = 35
scale = 36
scale = 336

shift = 0  # standard
shift = 1   # ignored 'total' column
# shift = 14
# shift = 6
X = np.array(df[df.columns[1+shift:scale]])

threshold = 1
threshold = 2
threshold = 2.24
# threshold = 3
# threshold = 4
# threshold = 8
# threshold = 9
# threshold = 10
# threshold = 100
# threshold = 1e-1
# threshold = 1e-2
threshold = 2e-2
# threshold = 0.001
# threshold = 0.0012005   # good
# threshold = 0.0013005   # good

# threshold = 0.0014
# threshold = 0.00146
# threshold = 0.00146104  # good
# threshold = 0.00146504  # good
# threshold = 0.00146604  # good
# threshold = 0.0014665  # bad
# threshold = 0.00146704  # bad
# threshold = 0.00146804  # bad
# threshold = 0.00147
# threshold = 0.0015005   # zero

# threshold = 1e-3  # interesting output
# threshold = 1e-4
# threshold = 1e-5
# threshold = 1e-8
# threshold = 1e-9
# # threshold = 1.5e-10
# threshold = 2e-10
# threshold = 3e-10
# threshold = 5e-10
# threshold = 1e-10
# threshold = 1e-15

# model = ps.SINDy(feature_names=["x", "y"],
#                 optimizer = ps.STLSQ(threshold=threshold))
model = ps.SINDy(feature_names=df.columns[1+shift:scale],
                 # feature_library=ps.PolynomialLibrary(degree=1),
                 optimizer = ps.STLSQ(threshold=threshold))

model.fit(X, t=t)

# print(model.coefficients())
print(sum(abs(model.coefficients()[0, :])>0))

print(f'\nfilename: {fname}, threshold: {threshold}, date: {print(datetime.datetime.now())} \n')
model.print()

print(sum(abs(model.coefficients()[0, :])>0))

# model.print(lhs=["total"])


