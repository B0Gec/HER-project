# library: only +, i.e. linear
import numpy as np

import pysindy as ps
from sklearn.metrics import mean_squared_error

# nrows = 550
# ncols = 35
# x = np.linspace(0, 1, nrows)
#
# y = 2 * x + 1
# Alist = [[i for i in x]] + [[np.random.random() for i in x] for _ in range(ncols-1)]
# A = np.array(Alist).T
#

from loadit import loaded, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fnames = [fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii]
fname = fnameiv
# fname = fnames[0]
df = loaded(fname, ppn_only=True)
# print(df)
cols = list(df.columns[1:])
# cols = cols[:5]
# print(cols)
# 1/0

# threshold = 0.001
threshold = 0.05
threshold = 0.1
threshold = 0.009

# i = 2
for i in range(len(cols)):
    break
    col_y = cols[i]
    other_cols = cols[:i] + cols[i+1:]
    # colnames = [col_y] + other_cols
    colnames = other_cols
    y = df[col_y]
    z = df[other_cols]
    print(col_y)
    # print(col_y, other_cols)
    # print(colnames)
    # print(y)
    # print(z)
    # if i == 2:
    #     1/0


    A = z
    # # print(A.shape)
    # # print(A)
    # # 1/0
    #
    b = y
    # A = np.vstack( (x, z) ).T


    # model = ps.SINDy(feature_names=["x"]+[f"z{i}" for i in range(ncols-1)],
    model=ps.SINDy(feature_names=colnames,
                    feature_library=ps.PolynomialLibrary(degree=1),
                     optimizer = ps.STLSQ(threshold=threshold),
                     )

    model.fit(A, x_dot=b)
    model.print(precision=1)
    dt = np.mean(df['t'][1:].to_numpy() - df['t'][:-1].to_numpy())
    # dt = np.mean(t[1:] - t[:-1])
    # print(A.shape, dt.shape)

    mse = model.score(A, x_dot=b, metric=mean_squared_error)
    rmse = mse**0.5
    print(f'rmse: {rmse}')
    print()


######

# finale: isotopes ^60Ni vs Ni
# y =

b = df['^60Ni- ppn']
A = df[['Ni- ppn']]

# model = ps.SINDy(feature_names=["x"]+[f"z{i}" for i in range(ncols-1)],
model = ps.SINDy(feature_names=['Ni- ppn'],
                 feature_library=ps.PolynomialLibrary(degree=1),
                 optimizer=ps.STLSQ(threshold=threshold),
                 )

model.fit(A, x_dot=b)
model.print(precision=1)
print(model.coefficients())
dt = np.mean(df['t'][1:].to_numpy() - df['t'][:-1].to_numpy())
# dt = np.mean(t[1:] - t[:-1])
# print(A.shape, dt.shape)

mse = model.score(A, x_dot=b, metric=mean_squared_error)
rmse = mse ** 0.5
print(f'rmse: {rmse}')
print()

# print(b)
# print(A)
