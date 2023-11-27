# library: only +, i.e. linear
import numpy as np

import pysindy as ps

# nrows = 550
# ncols = 35
# x = np.linspace(0, 1, nrows)
#
# y = 2 * x + 1
# Alist = [[i for i in x]] + [[np.random.random() for i in x] for _ in range(ncols-1)]
# A = np.array(Alist).T
#

from loadit import loaded, fnamei, fnameii, fnameiii, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fname = fnameiv
fname = fnameiv
df = loaded(fname, ppn_only=True)
# print(df)
cols = list(df.columns[1:])
# cols = cols[:5]
# print(cols)
# 1/0

# i = 2
for i in range(len(cols)):
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
                     # optimizer = ps.STLSQ(threshold=threshold),
                     )

    model.fit(A, x_dot=b)
    model.print(precision=9)

