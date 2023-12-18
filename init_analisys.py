"""
# library: only +, i.e. linear

# basic fisualizations such as grouped (reordered) correlation matrix heatmap,
# line plots, scatter plots, and histograms

first  there is a clustered correlation matrix heatmap by seaborn, a bit lower unsuccessful DIY heatmap by matplotlib.,

then follows threshold selection for line plots, errbar boxplot and violin plots and scatter plots (matplotlib).

ends with swarm plot by seaborn.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from loadit import loaded, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fnames = [fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii]
fname = fnameiv
fname_short = fname.split('NEG_Ni ')[-1].split('_AE')[0]
# fname = fnames[0]
df = loaded(fname, ppn_only=True)
# df = loaded(fname, ppn_only=True, notime=False)
# df = loaded(fname, non_ppn_only=True, ppn_only=False)
cols = list(df.columns)
print(df)

# dfa = df.sort_values(df.columns)
# print(dfa)
# print(df.sort_index(axis=1))
# print(df)

# #### correlation matrix:
# import seaborn as sns
#
# # Create a correlation matrix
# correlation_matrix = df.corr()
#
#
# # Create a clustermap of the correlation matrix
# # sns.clustermap(correlation_matrix)
# # Create a clustermap of the correlation matrix with a custom color palette
#
# # fig, ax = plt.subplots(figsize=(10, 10))
# plt.title('correlation matrix for')
# fig, ax = plt.subplots()
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# sns.clustermap(correlation_matrix, cmap=cmap)
# # plt.title('nekaj')
# # sns.title('correlation matrix for')
# fig = plt.gcf()  # Get the current figure object
# # Add padding to the top of the figure
# fig.subplots_adjust(top=0.92)
#
# title = 'Reordered correlation matrix for ' + fname_short + ' data'
# # Add a title to the plot with bold, bigger letters
# fig.suptitle(title, fontsize=16, )
#
#
# # Show the plot
# plt.show()
# plt.show()
# 1/0
# #### end of seaborns' easy way.



# 1/0

#
# print(df.iloc[:3, 1])
# print(df.iloc[:3, 2])
# print(df.iloc[:3, 1].corr(df.iloc[:3, 2]))
# print(pd.DataFrame([1,2,3]).corr(method='pearson'))
# print(pd.DataFrame({'a': [1,2,3], 'b': [2,4,6]}))
# dfa = pd.DataFrame({'a': [1,2,3], 'b': [4,4,4]})
# mia, sia = dfa['a'].mean(), dfa['a'].std()
# mib, sib = dfa['b'].mean(), dfa['b'].std()
# print(mia, sia, mib, sib)
#
# print(([(dfa['a'][i] - mia)*(dfa['b'][i] - mib) for i in range(dfa.shape[0])]))
# print(sum([(dfa['a'][i] - mia)*(dfa['b'][i] - mib) for i in range(dfa.shape[0])]))
# print(sum([(dfa['a'][i] - mia)*(dfa['b'][i] - mib) for i in range(dfa.shape[0])])/((dfa.shape[0]-1)*sia*sib))
#
# 1/0
# print(dfa.corr())
# print(pd.DataFrame({'a': [1,2,3], 'b': [2,4,6]}).corr())



# ### correlation matrix heatmap: matplotlib
# dfcorr = df.corr()
# df = dfcorr
#
# # in the following lines we reorder the correlation matrix so that the most correlated variables are next to each other:
# # https://stackoverflow.com/questions/6313308/get-column-index-from-column-name-in-python-pandas
# # https://stackoverflow.com/questions/31384310/how-to-swap-columns-in-a-pandas-dataframe
# # df = df.corr().unstack().sort_values(ascending=False).drop_duplicates()
# # df = df.corr().unstack().sort_values(ascending=False)
# # print(dfcorr)
# # print(dfcorr[sorted(df.columns)])
#
# df = df.reindex(index=sorted(df.columns))
# print(df)
#
# # 1/0
#
# # correlation matrix reodrdering heuristic:
# # we reorder the correlation matrix so that the most correlated variables are next to each other:
# # we start with the first variable and then we look for the most correlated variable to it,
# # then we look for the most correlated variable to the second variable, etc.
# # The measure of how correlated two variables are is the absolute value of their correlation coefficient.
# # code for reordering:
# # for index, variable in enumerate(df.columns)[1:]:
# #     next_variable = df.corr[variable].sort_values(ascending=False)[1:].index[0]
#
#
# # print(df.corr()index='S- ppn']))
#
# # print(df.corr())
# # print(df.corr().iloc[0, 1])
# #
# # df = df.iloc[:, 3]
#
#
#
# # 1/0
# # corr = df.corr()
# # df.columns = [i[:-5] for i in df.columns]
# df = df.rename(columns={i: i[:-5] for i in df.columns})
#
# print(df)
# fig, ax = plt.subplots(figsize=(10, 10))
# im = ax.imshow(df.corr())
#
# # Create colorbar
# # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel('legend', rotation=-90, va="bottom")
#
# plt.title('correlation matrix for ' + fname_short + ' data')
# # ax.set_title("Harvest of local farmers (in tons/year)")
#
# ax.set_xticks(np.arange(len(df.columns)), labels=df.columns)
# ax.set_yticks(np.arange(len(df.columns)), labels=df.columns)
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
#
# # # Loop over data dimensions and create text annotations.
# for i in range(len(df.columns)):
#     for j in range(len(df.columns)):
#         text = ax.text(j, i, f'{df.corr().iloc[i, j]:.0}',
#                        ha="center", va="center", color="w",
#                        fontsize='xx-small')
#
# fig.tight_layout()
# # plt.show()
#
#
# plt.show()
#
# # sns.heatmap(df.corr(), annot=False)
# # plt.title('heatmap')
#
# # plt.show()
#
# 1/0


# means = [np.mean(df[col]) for col in cols]
# means = [min(np.mean(df[col], 0.01) for col in cols if np.mean(df[col]) < 0.01 else 0.01]
# means = [min(np.mean(df[col]), 0.01) for col in cols]
# thresh = 100000.002
# thresh = 0.02
# thresh = 0.002
thrs = [10**5, 0.02, 0.002]
thr_idx = 1
thresh = thrs[thr_idx]

#### box plot?:
means = [min(np.mean(df[col]), thresh) for col in cols]
# vars = [np.var(df[col]) for col in cols]
vars = [np.var(df[col]) if (np.mean(df[col]) < thresh) else 0 for col in cols]



# plt.errorbar(x, means, yerr=vars, label='both limits (default)')
# plt.bar(x, means, yerr=vars, label='both limits (default)')
#### End Of - box plot?:

# print(df[[col for col in cols if max(df[col]) < 0.02]])
selection = [col for col in cols if max(df[col]) < thresh]
# selection = selection[:2]
print(f'number of fragments in the selection: {len(selection)}')

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



##### Start - violin and scatter plot:
# plt.subplots(figsize=(13, 13))
# # 14, 7, 12
#
# if thr_idx > 0:
#     plt.plot([0+1, len(selection)], [thresh, thresh], 'r--', label='threshold')
# if thr_idx < len(thrs)-1:
#     plt.plot([0+1, len(selection)], [thrs[thr_idx+1], thrs[thr_idx+1]], 'r--', label='threshold')
#
# for n, sel in enumerate(selection):
#     # plt.scatter([sel[:-5]]*df.shape[0], df[sel], label=sel[:-5], marker='x')
#     # plt.scatter([n+1]*df.shape[0], np.log(df[sel]), label=sel[:-5], marker='x')
#     plt.scatter([n + 1] * df.shape[0], df[sel],  marker='x')
#
# plt.violinplot(df[selection], showmeans=False, showmedians=True, showextrema=True, )
# # plt.violinplot(np.log(df[selection]), showmeans=False, showmedians=True, showextrema=True, )
# plt.xticks([y + 1 for y in range(len(selection))], labels=[i[:(-5 if 'ppn' in i else -1)]  for i in selection])
# # plt.xticks([sel for sel in selection], labels=[i[:-5] for i in selection])
##### End of violin and scatter plot:



#
#### seaborn swarm plot:
# selection = selection
dfnew = df[selection]
dfnew.columns = [i[:(-4 if 'ppn' in i else -1)]  for i in selection]

# sns.catplot(data=df[selection], kind="swarm", x="day", y="total_bill", hue="smoker")
# sns.catplot(data=df[selection], kind="swarm", x='fragment', y='intensity')
# sns.catplot(data=dfnew, kind="swarm", markersize=0.5)
# sns.catplot(data=dfnew, kind="swarm")
# sns.catplot(data=dfnew, kind="swarm", size=0.5, )

fig, ax = plt.subplots(figsize=(30, 10))
sns.swarmplot(data=dfnew, size=2.5)
# plt.title('correlation matrix for')
# fig, ax = plt.subplots()
fig = plt.gcf()  # Get the current figure object
# # Add padding to the top of the figure
fig.subplots_adjust(top=0.92)
title = f'Swarm scatter plot at threshold {thresh} for ' + fname_short + ' data'
fig.suptitle(title, fontsize=16, )

# sns.stripplot(data=dfnew)
#### End Of - seaborn swarm plot:

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

