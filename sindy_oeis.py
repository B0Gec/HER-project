import warnings
warnings.simplefilter("ignore")

from typing import Union
from itertools import product

import pandas as pd
import sympy as sp
import numpy as np
import pysindy as ps

warnings.filterwarnings('ignore', module='pysindy')
# from exact_ed import grid_sympy, dataset, unpack_seq, truth2coeffs, solution_vs_truth, instant_solution_vs_truth, solution2str, check_eq_man

import warnings
warnings.filterwarnings("ignore")


def sindy(seq: Union[list, sp.Matrix], max_order: int, seq_len: int, threshold: float = 0.1,
          ensemble: bool = False, library_ensemble: bool = False):
    """Perform SINDy."""

    # Generate training data
    # seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    #        1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    #        121393, 196418, 317811, 514229, 832040, 1346269,
    #        2178309, 3524578, 5702887, 9227465, 14930352, 24157817,
    #        39088169, 63245986, 102334155]

    # print(len(ongrid))
    seq = [int(i) for i in seq][:seq_len]
    # print(type(seq), type(seq[-1]))
    # seq = seq[:90]
    # seq = seq[:70]
    # seq = seq[:30]
    # print(f"{int(seq[-1]):.4e}")
    # 1/0

    b, A = dataset(seq, max_order, linear=True)
    # b, A = dataset(seq, 19, linear=True)
    # b, A = dataset(sp.Matrix(seq), 2, linear=True)
    # 2-7, 9-14, 16-19 finds, 8,15 not
    b, A = np.array(b, dtype=int), np.array(A, dtype=int)
    # print(b, A)
    # 1/0
    # data = grid_sympy(sp.Matrix(seq), max_order)
    # data = sp.Matrix.hstack(b, A)
    data = np.hstack((b, A))

    # print(data.shape, type(data))
    head = data[:6, :6]
    # for i in range(head.rows):
    # for i in range(head.shape[0]):
    #     print(data[i, :6])

    # for i in range(data[:6, :].rows):
    #     print(data[:i, :])
    #
    # print(data)


    # poly_order = 8
    poly_order = 1
    # threshold = 0.1

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
        feature_names=[f"a(n-{i+1})" for i in range(max_order-1)],
        discrete_time=True,
    )


    # # model.fit(x_train, t=dt)
    # model.fit(x_train, t=dt, x_dot=dot_x)
    # model.fit(A, x_dot=b)
    # model.fit(A, x_dot=b, ensemble=True)
    # model.fit(A, x_dot=b, library_ensemble=True)
    model.fit(A, x_dot=b, ensemble=ensemble, library_ensemble=library_ensemble)
    # model.print()
    model.coefficients()
    # print(model.coefficients())
    x = sp.Matrix([round(i) for i in model.coefficients()[0][1:]])
    x = sp.Matrix.vstack(sp.Matrix([0]), x)

    # print(x)
    return x
