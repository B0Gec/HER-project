import pandas as pd
import os

dir = 'data' + os.sep
FNAMEI =     'NEG_profil_Ni_VI_AE_typoless.txt'
FNAMEII =   'NEG_profil_Ni_VII_AE_typoless.txt'
FNAMEIII = 'NEG_profil_Ni_VIII_AE_typoless.txt'

fnames = [FNAMEI, FNAMEII, FNAMEIII]
fnamei, fnameii, fnameiii = tuple(dir + i for i in fnames)

def loaded(fname):
    f = open(fname, 'r')
    nline = 2
    line = [f.readline() for i in range(nline + 1)][nline]
    line = line[:-1]  # typeless manually edited data.
    # cols = ['t'] + line.split('\t')[1:-1]  # typo
    cols = line.split('\t')
    cols[0] = 't'
    # cols[2] = 'Ni-!'
    # cols[36] = 'H-!'  # in case of using original files
    f.close()
    df = pd.read_csv(fname, sep='\t', skiprows=5, header=None)
    df = df.iloc[:, :-1]
    df.columns = cols
    return df

# df = loaded(fnamei)
# print(f'starting loading {fnamei}')
# print(df)
# print('end loading')
#

# ## cols check:
# cols = df.columns
# print(len(cols), set(cols), )
# print(len(set(cols)))
# counts = [(i, len([j for j in cols if j == i])) for i in cols]
# counts = [(i, len([j for j in cols if j == i])) for i in cols if len([j for j in cols if j == i]) > 1]
#


