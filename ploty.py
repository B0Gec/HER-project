"""
Basic vizualization of the data by line plots.

In first half by matplotlib, in second half by seaborn.
In second half by seaborn there is a threshold for the data to be visualized.
Look files lineplots_*.png for the results.
"""
# from loadit import loaded
#
# import datetime
#
# import os
# from math import log
# from io import StringIO
# import sys

import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.metrics import mean_squared_error
# import pysindy as ps
import numpy as np
# # print(pysindy.__version__, sys.version)
# # print(warnings.filters)

# from loadit import loaded, fnamei, fnameii, fnameiii, fnameiv, fnamevi
from loadit import loaded, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fname = fnameiv
# fname = fnamevi
# fname = fnameviii
# fname = fnamex
# fname = fnamexi
# fname = fnamexii
fname_short = fname.split('NEG_Ni ')[-1].split('_AE')[0]
df = loaded(fname, ppn_only=True, ignore_total=True, notime=False)
# # df = loaded(fnameiv, ppn_only=False, ignore_total=True)
#
# # [df[i] for i in df.columns[1:]]
#
# # plt.style.use('_mpl-gallery')
#
# plt.title(fname)
# fig, ax = plt.subplots()
# ax.set_title(fname)
# mp = plt.colormaps['hsv']
# mp = plt.colormaps['gist_rainbow']
# mp = plt.colormaps['twilight']
# mp = plt.colormaps['tab20b']
# # cycler(color=['c', 'm', 'y', 'k']) +
# #                  cycler(lw=[1, 2, 3, 4]))
#
# types = ['-', '--', '-.', ':']
# def_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# from itertools import product
# colstyle = list(product(types, def_colors))
# print(colstyle)
#
# print(len(def_colors))
# cycle = []
#
# cols = df.columns[1:][:555]
# for n, i in enumerate(cols):
#     # ax.plot(df['t'], df[i], label=i, color=mp(n/len(df.columns[1:])*0.9))
#     # ax.plot(df['t'], df[i], label=i, color=(1.0, 0, 0, 1))
#     # ax.plot(df['t'], [1 + n for i in df['t']], label=i, linestyle=colstyle[1], color=mp(n/len(cols)*0.999))
#     # ax.plot(df['t'], [1 + n for i in df['t']], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
#     ax.plot(df['t'], df[i], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
# #     # ax.stackplot(df['t'], df[i], label=i)
#
# # ax.stackplot(df['t'], *[df[i] for i in df.columns[1:]], labels=df.columns[1:])
# ax.legend()
# plt.show()
#
#
# print(df)


##### seaborn try:
import seaborn as sns
# df = df.iloc[:1000, :2]
print(df)
# df = df[df.columns[:3]]
# df_wide = df.pivot(index="t", columns=df.columns)
# df = df.reindex(index=df['t'], data=df)
# df = df.reindex(index=df['t'], columns=df.columns[1:])
df.set_index('t', inplace=True)
print(df)

# print(df['C_3- ppn'])
# sns.relplot(data=df['C_3- ppn'], kind="line")
# plt.show()

# 1/0


# separation:
thrs = [10**5, 0.02, 0.002]
thr_idx = 2
thresh = thrs[thr_idx]
mean_thresh = 0.0002
viz_thresh = 0.0002


cols = list(df.columns)

# print(df[[col for col in cols if max(df[col]) < 0.02]])

selection = [col for col in cols if max(df[col]) < thresh]
mean_selection = [col for col in cols if max(df[col]) < thresh and np.mean(df[col]) < mean_thresh]
# selection = selection[:2]
print(f'number of fragments in the selection: {len(selection)}')


types = ['-', '--', '-.', ':']
# def_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
def_colors = sns.color_palette()
from itertools import product
colstyle = list(product(types, def_colors))

df = df[selection]
# dfnew.columns = [i[:(-4 if 'ppn' in i else -1)]  for i in selection]

fig, ax = plt.subplots(figsize=(15, 10))

# cols = selection[:49]
cols = mean_selection
for n, i in enumerate(cols):
    # sns.relplot(df['t'], df[i], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
    # sns.relplot(data=df[i], kind="line", linestyle=types[n], color=sns.color_palette()[n])
    # sns.relplot(data=df[i], label=i, kind='line', linestyle=colstyle[n][0], color=colstyle[n][1])
    # sns.lineplot(data=df[i], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])
    sns.lineplot(data=[min(j, viz_thresh) for j in df[i]], label=i, linestyle=colstyle[n][0], color=colstyle[n][1])

fig = plt.gcf()  # Get the current figure object
# Add padding to the top of the figure
fig.subplots_adjust(top=0.92, right=0.88)
title = f'Line plots revisited at threshold {thresh} with magnifying threshold {viz_thresh} for ' + fname_short + ' data'
# Add a title to the plot with bold, bigger letters
fig.suptitle(title, fontsize=16, )

plt.legend(bbox_to_anchor=(0.9985, 0.9))
##### End Of - seaborn try:


plt.show()
