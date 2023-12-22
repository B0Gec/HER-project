"""
metrics:
    - correlation matrix [done]
    - max absolute error (uniform) [done]
    - root mean squared error [done]
    - dynamic time warping distance [not done]
    - cross correlation [not done]
    - wasserstein distance [not done]
    - bottleneck distance [not done]

metric -> adj matrix -> heatmap
metric -> chosen clustering -> heatmap
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# # from scipy.spatial.distance import bottleneck
#
# # Create two vectors
# vector1 = np.array([1, 2, 3, 4])
# vector2 = np.array([5, 4, 3, 2])
#
# # Calculate the bottleneck distance between the vectors
# bottleneck_distance = bottleneck(vector1, vector2)
# print(bottleneck_distance)
# 1/0

def max_unif_diff(x, y):
    return np.max(np.abs(x - y))

def root_mean_squared_error(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def bootle_ie_wasser_inf_dist(x,y):




print('max_unif_diff', max_unif_diff(np.array([1, 2, 3]), np.array([1, 2, 3])))
a, b = np.array([10, 3, -2]), np.array([1, 2, 3])
print('max_unif_diff', max_unif_diff(a, b))
print('diff', a - b)
print('abs', np.abs(a - b))

# root_mean_squared_error of a and b by numpy:




from loadit import loaded, fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii

fnames = [fnameiv, fnamevi, fnameviii, fnamex, fnamexi, fnamexii]
fname = fnameiv
fname_short = fname.split('NEG_Ni ')[-1].split('_AE')[0]
df = loaded(fname, ppn_only=True)
cols = list(df.columns)
print(df)

CORRELATION_MATRIX = False
# Create a correlation matrix
# correlation_matrix = df.corr()
# adj_matrix = np.abs(correlation_matrix)

distance_of_choice = max_unif_diff
distname = distance_of_choice.__name__

# adj_matrix = pd.DataFrame(np.array([[max_unif_diff(df[i], df[j]) for j in df.columns]
#                                     for i in df.columns]), columns=df.columns, index=df.columns)
adj_matrix = pd.DataFrame(np.array([[distance_of_choice(df[i], df[j]) for j in df.columns]
                                    for i in df.columns]), columns=df.columns, index=df.columns)
print(adj_matrix)
print(type(adj_matrix), adj_matrix.shape)




fig, ax = plt.subplots()
# if CORRELATION_MATRIX:
#     cmap = sns.diverging_palette(220, 10, as_cmap=True)
#     sns.clustermap(correlation_matrix, cmap=cmap)
# else:
# sns.heatmap(adj_matrix)
sns.clustermap(adj_matrix)

fig = plt.gcf()  # Get the current figure object
fig.subplots_adjust(top=0.92, bottom=0.19, left=0.15)
title = f'Distance (adj) matrix by {distname} for ' + fname_short + ' data'
fig.suptitle(title, fontsize=16, )
plt.show()


