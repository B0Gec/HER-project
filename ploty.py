# from loadit import loaded
#
# import datetime
#
# import os
# from math import log
# from io import StringIO
# import sys

import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
# import pysindy as ps
# import numpy as np
# # print(pysindy.__version__, sys.version)
# # print(warnings.filters)

# from loadit import loaded, fnamei, fnameii, fnameiii, fnameiv, fnamevi
from loadit import loaded, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fname = fnameiv
fname = fnamevi
fname = fnameviii
fname = fnamex
fname = fnamexi
fname = fnamexii
df = loaded(fname, ppn_only=True, ignore_total=True)
# df = loaded(fnameiv, ppn_only=False, ignore_total=True)

# [df[i] for i in df.columns[1:]]

# plt.style.use('_mpl-gallery')

plt.title(fname)
fig, ax = plt.subplots()
ax.set_title(fname)
mp = plt.colormaps['hsv']
mp = plt.colormaps['gist_rainbow']
mp = plt.colormaps['twilight']
mp = plt.colormaps['tab20b']
# cycler(color=['c', 'm', 'y', 'k']) +
#                  cycler(lw=[1, 2, 3, 4]))

types = ['-', '--', '-.', ':']
def_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
from itertools import product
colstyle = list(product(types, def_colors))
print(colstyle)

print(len(def_colors))
cycle = []

cols = df.columns[1:][:555]
for n, i in enumerate(cols):
    # ax.plot(df['t'], df[i], label=i, color=mp(n/len(df.columns[1:])*0.9))
    # ax.plot(df['t'], df[i], label=i, color=(1.0, 0, 0, 1))
    # ax.plot(df['t'], [1 + n for i in df['t']], label=i, linestyle=colstyle[1], color=mp(n/len(cols)*0.999))
    # ax.plot(df['t'], [1 + n for i in df['t']], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
    ax.plot(df['t'], df[i], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
#     # ax.stackplot(df['t'], df[i], label=i)


# ax.stackplot(df['t'], *[df[i] for i in df.columns[1:]], labels=df.columns[1:])
ax.legend()
plt.show()


print(df)
