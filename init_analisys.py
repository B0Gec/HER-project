# library: only +, i.e. linear
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



### correlation matrix heatmap:
dfcorr = df.corr()
print(df.corr())
print(df.corr().iloc[0, 1])
# 1/0
# corr = df.corr()
# df.columns = [i[:-5] for i in df.columns]
df = df.rename(columns={i: i[:-5] for i in df.columns})

print(df)
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(df.corr())
plt.title('correlation matrix for ' + fname_short + ' data')
# ax.set_title("Harvest of local farmers (in tons/year)")

ax.set_xticks(np.arange(len(df.columns)), labels=df.columns)
ax.set_yticks(np.arange(len(df.columns)), labels=df.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        text = ax.text(j, i, f'{df.corr().iloc[i, j]:.1}',
                       ha="center", va="center", color="w")

fig.tight_layout()
# plt.show()


plt.show()

# sns.heatmap(df.corr(), annot=False)
# plt.title('heatmap')

# plt.show()

1/0


# means = [np.mean(df[col]) for col in cols]
# means = [min(np.mean(df[col], 0.01) for col in cols if np.mean(df[col]) < 0.01 else 0.01]
# means = [min(np.mean(df[col]), 0.01) for col in cols]
# thresh = 100000.002
# thresh = 0.02
# thresh = 0.002
thrs = [10**5, 0.02, 0.002]
thr_idx = 2
thresh = thrs[thr_idx]

means = [min(np.mean(df[col]), thresh) for col in cols]
# vars = [np.var(df[col]) for col in cols]
vars = [np.var(df[col]) if (np.mean(df[col]) < thresh) else 0 for col in cols]


# plt.errorbar(x, means, yerr=vars, label='both limits (default)')
# plt.bar(x, means, yerr=vars, label='both limits (default)')

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

plt.subplots(figsize=(13, 13))
# 14, 7, 12

if thr_idx > 0:
    plt.plot([0+1, len(selection)], [thresh, thresh], 'r--', label='threshold')
if thr_idx < len(thrs)-1:
    plt.plot([0+1, len(selection)], [thrs[thr_idx+1], thrs[thr_idx+1]], 'r--', label='threshold')

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

