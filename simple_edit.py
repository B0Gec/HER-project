import datetime
import os

import pysindy as ps
import numpy as np
# print(pysindy.__version__, sys.version)
# print(warnings.filters)

from loadit import loaded, fnamei, fnameii, fnameiii

MAKEFILE = True

fname = fnamei
df = loaded(fname)
print(df)
1/0

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
threshold = 100
# # threshold = 1e-1
# threshold = 5e-2
# threshold = 4e-2
# threshold = 3e-2
# threshold = 2.5e-2
# threshold = 2e-2
# threshold = 1e-2
# threshold = 0.001
# threshold = 0.0012005   # good
# threshold = 0.0013005   # good
#
# # threshold = 0.0014
# # threshold = 0.00146
# # threshold = 0.00146104  # good
# # threshold = 0.00146504  # good
# # threshold = 0.00146604  # good
# # threshold = 0.0014665  # bad
# # threshold = 0.00146704  # bad
# # threshold = 0.00146804  # bad
# # threshold = 0.00147
# # threshold = 0.0015005   # zero
#
# threshold = 1e-3  # interesting output
threshold = 1e-4
# threshold = 2e-5
# threshold = 1.52e-5
threshold = 1e-5
# threshold = 1e-6
# # threshold = 1e-7
# threshold = 1e-8
# threshold = 1e-9

# # threshold = 1.5e-10
# # threshold = 2e-10
# # threshold = 3e-10
# # threshold = 5e-10
# threshold = 1e-10
# threshold = 1e-12
# threshold = 1e-15

degree = 1
# degree = 2
degree = 3
# model = ps.SINDy(feature_names=["x", "y"],
#                 optimizer = ps.STLSQ(threshold=threshold))
model = ps.SINDy(feature_names=df.columns[1+shift:scale],
                 feature_library=ps.PolynomialLibrary(degree=degree),
                 optimizer = ps.STLSQ(threshold=threshold))

model.fit(X, t=t)

# print(model.coefficients())
print(sum(abs(model.coefficients()[0, :])>0))

total = False if shift == 1 else True
meta = f'\nfilename: {fname}, \'total\' included: {total}, degree: {degree}, threshold: {threshold}, date: {str(datetime.datetime.now())} \n'
print(meta)

from io import StringIO
import sys
tmp = sys.stdout
my_result = StringIO()
sys.stdout = my_result
# print('hello world')  # output stored in my_result
model.print(precision=4)
sys.stdout = tmp
eqs = my_result.getvalue()
print(eqs)


# print(sum(abs(model.coefficients()[0, :])>0))

if MAKEFILE:

    vi = fname.split('_')[3]
    outdir = f'results{os.sep}auto-gen{os.sep}'
    outtxt = f'res_{vi}_{"un"*(not total)}tot{int(total)}_deg{degree}_thr{threshold}.txt'
    outpath = outdir + outtxt
    print(f'results written to {outpath}!!')
    f = open(outpath, 'w')
    f.write(meta + '\n')
    f.write(eqs)
    f.close()

    import pickle
    outmodel = f'{outdir}model{outtxt[3:-4]}.pkl'
    with open(outmodel, 'wb') as f:
        pickle.dump(model, f)
    print(f'model_path = \'{outmodel}\'')

# model.print(lhs=["total"])


