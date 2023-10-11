import pandas as pd
import os

dir = 'data' + os.sep
FNAMEI =     'NEG_profil_Ni_VI_AE_typoless.txt'
FNAMEII =   'NEG_profil_Ni_VII_AE_typoless.txt'
FNAMEIII = 'NEG_profil_Ni_VIII_AE_typoless.txt'
FNAMEIV = 'NEG_Ni IV_AE_SM_HMR_Bi1@1_0pA_300um_250eVCs_500um_1.txt'
FNAMEVI = 'NEG_Ni VI_AE_SM_HMR_Bi1@1_0pA_300um_250eVCs_500um_1.txt'

fnames = [FNAMEI, FNAMEII, FNAMEIII, FNAMEIV, FNAMEVI]
fnamei, fnameii, fnameiii, fnameiv, fnamevi = tuple(dir + i for i in fnames)

def loaded(fname, ppn_only=False):
    f = open(fname, 'r')
    nline = 2
    line = [f.readline() for i in range(nline + 1)][nline]
    # line = line[:-1]  # typeless manually edited data.
    # cols = ['t'] + line.split('\t')[1:-1]  # typo
    cols = line.split('\t')
    cols[0] = 't'
    cols = [i if i != '' else cols[n-1]+' ppn' for n, i in enumerate(cols)]

    # cols[2] = 'Ni-!'
    # cols[36] = 'H-!'  # in case of using original files
    f.close()
    # df = pd.read_csv(fname, sep='\t', skiprows=5, header=None)
    df = pd.read_csv(fname, sep='\t', skiprows=5, header=None, names=cols)
    # df = pd.read_csv(fname, sep='\t', skiprows=5, header=0)
    df = df.iloc[:, :-1]
    # df.columns = df.iloc[0, :]
    # df = df.iloc[:, :-1]
    # df.columns = cols
    if ppn_only:
        ppncols = ['t'] + [i for i in cols if 'ppn' in i]
        df = df[ppncols]
    return df

df = loaded(fnameiv)
# print(df)
df = loaded(fnameiv, ppn_only=True)
# print(df)
# print(f'starting loading {fnamei}')
# print(df)
# print('end loading')
#

# ## cols check:
cols = df.columns
# print(len(cols), set(cols), )
# print(len(set(cols)))
counts = [(i, len([j for j in cols if j == i])) for i in cols]
# print(counts)
counts = [(i, len([j for j in cols if j == i])) for i in cols if len([j for j in cols if j == i]) > 1]


