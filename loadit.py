import pandas as pd
import os

dir = 'data' + os.sep
FNAMEI = 'NEG_profil_Ni_VI_AE.txt'
FNAMEII = 'NEG_profil_Ni_VII_AE.txt'
FNAMEIII = 'NEG_profil_Ni_VIII_AE.txt'

fnames = [FNAMEI, FNAMEII, FNAMEIII]
fnamei, fnameii, fnameiii = tuple(dir + i for i in fnames)

def loaded(fname):
    f = open(fname, 'r')
    nline = 2
    line = [f.readline() for i in range(nline + 1)][nline]
    cols = ['t'] + line.split('\t')[1:-1]
    f.close()
    df = pd.read_csv(fname, sep='\t', skiprows=5, header=None)
    df.columns = cols + ['\t']
    df = df.iloc[:, :-1]

    return df

# df = loaded(fnamei)
# print(f'starting loading {fnamei}')
# print(df)
# print('end loading')
#
