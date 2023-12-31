# 1.) loading txt as pandas dataframe

# import pandas as pd
# import numpy as np
# import os
#
# dir = 'data' + os.sep
# fnamei = 'NEG_profil_Ni_VI_AE.txt'
# fnameiec = 'NEG_profil_Ni_VI_AE-edited.txt'
# fnameie = 'NEG_profil_Ni_VI_AEeh.txt'
# fnameie = 'NEG_profil_Ni_VI_AE-et.txt'
# fnameie = 'NEG_profil_Ni_VI_AE-ecom.txt'
# fnameii = 'NEG_profil_Ni_VII_AE.txt'
# fnameiii = 'NEG_profil_Ni_VIII_AE.txt'
# import io
#
# # f = open(dir + fnamei, 'r')
# # data = f.read()
# # data = io.StringIO()
# # f.close
# # data = io.StringIO()
#
# print(dir + fnamei)
# f = open(dir + fnamei, 'r')
# nline = 2
# line = [f.readline() for i in range(nline+1)][nline]
# cols = ['t'] + line.split('\t')[1:-1]
# print(cols)
# f.close()
# print(len(cols))
# # 1/0
# # df = pd.read_csv(dir + fnamei, sep='\t', header=0, index_col=0)
# # df = pd.read_csv(dir + fnamei, sep='\t')
# df = pd.read_csv(dir + fnamei, comment='#', sep='\t')
# df = df.iloc[:, :-1]
# print(df)
# # 1/0
# # df = pd.read_csv(dir + fnameie, comment='#')
# df = pd.read_csv(dir + fnameie, comment='#')
# df = pd.read_csv(dir + fnameie, comment='#', sep='\t', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
# df = pd.read_csv(dir + fnameie, comment='#', sep='\t', usecols=[i for i in range(39)])
# print(df)
# print('a')
# # 1/0
#
# df = pd.read_csv(dir + fnameiec)
# df = pd.read_csv(dir + fnameiec, sep='\t')
# df = pd.read_csv(dir + fnameie, comment='#', sep='\t', usecols=[i for i in range(40)])
# print(df)
# # 1/0
# # df = pd.read_csv(dir + fnameie, comment='#', sep='\t', usecols=[i for i in range(39)])
# print(len(df.columns), df.columns)
#
# def loaded(fname):
#     # df_cut = df[1:]
#     f = open(fname, 'r')
#     nline = 2
#     line = [f.readline() for i in range(nline + 1)][nline]
#     cols = ['t'] + line.split('\t')[1:-1]
#     f.close()
#     # df = pd.read_csv(fname, comment='#', sep='\t', header=0)
#     # df = pd.read_csv(fname, sep='\t', skiprows=5, header=None, names=cols)  # not working
#     # df = pd.read_csv(fname, sep='\t', skiprows=5, header=0, names=cols)  # not working
#     df = pd.read_csv(fname, sep='\t', skiprows=5, header=None)
#     df.columns = cols + ['\t']
#     df = df.iloc[:, :-1]
#
#
#     return df, cols
#
#
# print('\n'*5)
# saved_cols = []
# for fname in [fnamei, fnameii, fnameiii]:
#     df, cols = loaded(dir + fnamei)
#     print(f'starting loading {fname}')
#     print(df)
#     print(len(cols), cols)
#     print(saved_cols == cols)
#     saved_cols = cols
#     print('end loading')
# # 1/0
#
# # df = pd.read_csv(dir + fnamei)
# # df = pd.read_csv(dir + fnameie, header=[0,1])
# # df = pd.read_csv(dir + fnameie, sep='\t')
# # df = pd.read_csv(dir + fnameie, sep='\t')
# print(df.columns)
#
# print(df)
#
# df_cut = df[1:]
# print(df_cut)
# a = np.array(df)
#
# print(len(df.index))
# # 1/0
# # for row in df_cut:
# #     print(row)
# print(df[1:2])
# print('df0')
#
# # print(a[1])
#
#
# # for i in df:
# #     print(i)
#
# def check_tot(df):
#     """check tot"""
#
#     print('inside check tot')
#
#     print(len(df.index))
#     count = 0
#     for i in range(0, len(df.index)):
#         # row = df[i:i+1]
#         # print(i, row)
#         row = np.array(df[i:i+1])[0]
#         # row = list(df[i:i+1])
#         # print(i, row)
#
#         # print([i for i in row])
#         # print(row[1], 'data and sum', sum(row[2:]), sum(row[2:])*2, row[1:])
#         print(row[1], 'data and sum', row[1]-sum(row[2:]))
#         if row[1] < sum(row[2:]):
#             count += 1
#             print('\n'*5, 'count', count)
#         # print(row[1], 'sum', sum(row[2:]), sum(row[2:])*2,)
#         # print(row[1], 'explain', [sum(row[2:i]) for i in range(2+1, len(row))])
#         # break
#     print(count, 'count')
#
#     return
#
# check_tot(df_cut)
#


# 2.) loading pickled sindy model

import pickle

# model_path = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/'



model_path = f'results/auto-gen/model_VI_untot0_deg1_thr100.pkl'
model_path = 'results/auto-gen/model_VI_untot0_deg3_thr1.52e-05.pkl'
model_path = 'results/auto-gen/model_VI_untot0_deg3_thr2e-05.pkl'
model_path = 'results/auto-gen/model_VI_untot0_deg3_thr1e-06.pkl'


with open(model_path, 'rb') as f:
    model = pickle.load(f)
model.print(precision=3)
print(2)

