from scipy.stats._stats_py import _cdf_distance as _cdf
import numpy as np
from scipy.stats import wasserstein_distance, energy_distance

a, b = [1,2,3,54], [1,2,3,4]
"""
1 2 3 4
1 2 3 54
0 0 0 50
50
"""
# print(_cdf(2, a, b))
# print(energy_distance( a,b)/np.sqrt(2))
# 1/0

# for i in [_cdf(10**i, a,b) for i in range(1, 10)][500:538]
for i in [_cdf(10**i, a,b) for i in range(1, 120)][0:538]:
    print(' in')
    print(i)
# [0.25195974860982445, 0.2519558294971846, 0.2519519260280944, 0.2519480381090765, 0.2519441656473971, 0.25194030855105787, 0.25193646672878944, 0.2519326400900438, 0.2519288285449871, 0.25192503200449284, 0.2519212503801349, 0.25191748358418053, 0.2519137315295838, 0.2519099941299785, 0.2519062712996721, 0.25190256295363855, 0.2518988690075123, 0.25189518937758176, 0.2518915239807828, 0.2518878727346927, 0.25188423555752404, 0.2518806123681182, 0.2518770030859398, 0.2518734076310705, 0.25186982592420304, 0.25186625788663564, 0.25186270344026584, 0.25185916250758533, 0.2518556350116739, 0.25185212087619396, 0.2518486200253852, 0.2518451323840589, 0.25184165787759283, 0.25183819643192557, 0.25183474797355165, 0.251831312429516, 0.25182788972740905, 0.0]
print('here')

u_values, u_weights = np.array([1,  2,3,54]), None
v_values, v_weights = np.array([1,-12,4,23]), None
"""
-12 1 4 23
  1 2 3 54
13 1 1 31
"""
print()
# print('inspect 1', (1/4)*np.sum(np.array([13, 1, 1, 31])))
#
# print(_cdf(1, u_values, v_values))
print('inspect square', np.sqrt((1/4)*np.sum(np.square(np.array([13, 3, 1, 31])))))
print('inspect square', np.square(np.array([13, 3, 1, 31])))
# print(31/4)

print(_cdf(2, u_values, v_values))
# from scipy.stats import wasserstein_distance, energy_distance
# print('vaser', wasserstein_distance(u_values, v_values))
# print('emd', energy_distance(u_values, v_values))
# print(wd(u_values, v_values))
print('inspect inf', 31/4)
1/0


# u_values, u_weights = _validate_distribution(u_values, u_weights)
# v_values, v_weights = _validate_distribution(v_values, v_weights)

u_sorter = np.argsort(u_values)
v_sorter = np.argsort(v_values)
print(u_values, v_values)
print(u_sorter, v_sorter)
print(v_values[v_sorter], 'sorted')
print('202')

all_values = np.concatenate((u_values, v_values))
print(all_values)
all_values.sort(kind='mergesort')
print(all_values)

# Compute the differences between pairs of successive values of u and v.
deltas = np.diff(all_values)
print(deltas, 'deltas')
print(u_values.size, 'values size')

# Get the respective positions of the values of u and v among the values of
# both distributions.
print(all_values[:-1])
print(u_values[u_sorter], 'sorted')
u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')
print(u_cdf_indices, v_cdf_indices)
print(np.searchsorted(u_values[u_sorter], all_values[:-1], 'right'))




def _cdf_distance2(p, u_values, v_values, u_weights=None, v_weights=None):
    """Compute the bottleneck distance between two weighted distributions.
    see scipy.stats.wasserstein_distance or scipy.stats._cdf_distance
    """

    from scipy.stats._stats_py import _validate_distribution

    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        print(np.multiply(np.abs(u_cdf - v_cdf), deltas))
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        print(np.multiply(np.abs(u_cdf - v_cdf), deltas))
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    if p == np.inf:
        u_cdf, v_cdf = u_cdf * u_values.size, v_cdf * v_values.size
        print(np.multiply(np.abs(u_cdf - v_cdf), deltas))
        return np.max(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1 / p)

def bottleneck_distance(u_values, v_values, u_weights=None, v_weights=None):
    return _cdf_distance2(np.inf, u_values, v_values, u_weights, v_weights)


# u_values, v_values = a, b
print(_cdf(1, u_values, v_values))
print(_cdf_distance2(1, u_values, v_values))
print(_cdf(2, u_values, v_values))
print(_cdf_distance2(2, u_values, v_values))
print(bottleneck_distance(u_values, v_values))
print(_cdf_distance2(np.inf, u_values, v_values))
print(_cdf(500, u_values, v_values))
print(_cdf(1500, u_values, v_values))
