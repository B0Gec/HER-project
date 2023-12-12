# old init_analisys.py

# library: only +, i.e. linear
import numpy as np
import matplotlib.pyplot as plt


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
# df = loaded(fname, ppn_only=True, notime=False)
# df = loaded(fname, non_ppn_only=True, ppn_only=False)
# df = loaded(fname, non_ppn_only=True, ppn_only=False, notime=False)
cols = list(df.columns)
print(df)

# df = loaded(fname, ppn_only=False, notime=False, ignore_total=False)
# print(df['total'])
# i1, i2 = 3, 4
# i1, i2 = 41, 42
# print(df.iloc[:, i1])
# print(df.iloc[:, i2])
# print(df.iloc[:, i1]/df['total'] - df.iloc[:, i2])
# print((df.iloc[:, i1]/df['total'] - df.iloc[:, i2])[50:80])
# print(list(abs(df.iloc[:, i1]/df['total'] - df.iloc[:, i2]) > 1e-6).index(True))
# # print((abs(df.iloc[:, 3]/df['total'] - df.iloc[:, 4]) > 1e-8)[:140])
# # print((df.iloc[:, 3]/df['total'] - df.iloc[:, 4])[:20])


# print(list(df.iloc[208:, 0] < df.iloc[208:, 1]).index(False))
# print(list(df.iloc[218:, 0] < df.iloc[218:, 1]).index(True))
# print(list(df.iloc[208:, 0] < df.iloc[208:, 1]))
# print(max(df.iloc[:, 0]))
# print(max(df.iloc[:, 1]))




t = df['t']
plt.plot(t, df.iloc[:, 19])
plt.plot(t, df.iloc[:, 20])
print(df.columns[19])
print(df.columns[20])

plt.show()
1/0
fig = plt.figure(figsize=(16, 16))
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)


x = np.arange(len(cols))
# means = [np.mean(df[col]) for col in cols]
# means = [min(np.mean(df[col], 0.01) for col in cols if np.mean(df[col]) < 0.01 else 0.01]
# means = [min(np.mean(df[col]), 0.01) for col in cols]
thresh = 100000.002
# thresh = 0.02
# thresh = 0.002

means = [min(np.mean(df[col]), thresh) for col in cols]
# vars = [np.var(df[col]) for col in cols]
vars = [np.var(df[col]) if (np.mean(df[col]) < thresh) else 0 for col in cols]


# plt.errorbar(x, means, yerr=vars, label='both limits (default)')
# plt.bar(x, means, yerr=vars, label='both limits (default)')

# print(df[[col for col in cols if max(df[col]) < 0.02]])
selection = [col for col in cols if max(df[col]) < thresh]
# selection = selection[:2]

# plt.boxplot(df[selection], labels=[i[:-5] for i in selection])

# plt.violinplot(np.log(df[selection]), showmeans=False, showmedians=True, showextrema=True, )
# , labels=[i[:-5] for i in selection])
# 1/0



# in following line is the code that sets xticks for violin plot:
# https://matplotlib.org/3.1.1/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
# plt.xticks([y + 1 for y in range(len(selection))], labels=[i[:-5] for i in selection])

# .set_ticks(, labels=[i[:-5] for i in selection])
# plt.set_xticks([y + 1 for y in range(len(all_data))],
#               labels=['x1', 'x2', 'x3', 'x4'])
# fig.set_xticks(labels=['x1', 'x2', 'x3', 'x4'])

# from functools import reduce
# xs = [[i for _ in range(df.shape[0])] for i in range(len(selection))]
# xs = sum(xs, [])
# ys = df[selection].values.flatten()
# print(ys)
#

# plt.scatter(xs, ys, label='both limits (default)')

for n, sel in enumerate(selection):
    # plt.scatter([sel[:-5]]*df.shape[0], df[sel], label=sel[:-5], marker='x')
    # plt.scatter([n+1]*df.shape[0], np.log(df[sel]), label=sel[:-5], marker='x')
    plt.scatter([n + 1] * df.shape[0], df[sel],  marker='x')



plt.violinplot(df[selection], showmeans=False, showmedians=True, showextrema=True, )
# plt.violinplot(np.log(df[selection]), showmeans=False, showmedians=True, showextrema=True, )
plt.xticks([y + 1 for y in range(len(selection))], labels=[i[:(-5 if 'ppn' in i else -1)]  for i in selection])
# plt.xticks([sel for sel in selection], labels=[i[:-5] for i in selection])


# violin plot:
# https://matplotlib.org/3.1.1/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
# plot violin plot
# axs[0].violinplot(all_data,
#                   showmeans=False,
#                   showmedians=True)
# axs[0].set_title('Violin plot')

# ys = np.log(df['C_3- ppn'])
# plt.violinplot(ys, showmeans=False, showmedians=True, showextrema=True, )
# plt.scatter([1 for i in ys], ys, marker='x')

plt.show()

