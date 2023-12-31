import datetime

import os
from math import log
from io import StringIO
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pysindy as ps
import numpy as np
# print(pysindy.__version__, sys.version)
# print(warnings.filters)

from loadit import loaded, fnamei, fnameii, fnameiii, fnameiv, fnamevi

MAKEFILE = True

fname = fnamei
fname = fnameiv
vi = fname.split('_')[1][3:]
# df = loaded(fname)
df = loaded(fname, ppn_only=True)
# print(df)
# 1/0

# t = np.linspace(0, 1, 100)
t = np.array(df['t'])
# x = np.array(df['total'])
# y = np.array(df['Ni-'])
# simple:
# x = 3 * np.exp(-2 * t)
# y = 0.5 * np.exp(t)

# X = np.stack((x, y), axis=-1)  # First column is x, second is y


scale = 4
# scale = 14
# scale = 20
# scale = 25
# scale = 30
# # # scale = 32
# # # scale = 33
# # # scale = 35
# # # scale = 35  # fail
# scale = 36
scale = 336

shift = 0  # standard
shift = 1   # ignored 'total' column
# shift = 14
# shift = 6
X = np.array(df[df.columns[1+shift:scale]])

thr_ord = 10
# 4 * 10^4 lstmse at 0.923
# 1 * 10^4 lstmse at 0
threshold = 0  # mse: 9.9612e+03, mse_sim: inf
# threshold = 0.923  # mse: 4.7354e+04, mse_sim: 2.0143e+08
threshold = 0.1
# threshold = 1
# # # threshold = 2
# threshold = 2.24
# threshold = 3
# # threshold = 3.5

# threshold = 4
# threshold = 5
# threshold = 6
# threshold = 8
# # threshold = 9
# threshold = 10
# threshold = 100

# # # # threshold = 1e-1
# # # threshold = 5e-2
# # # threshold = 4e-2
# # # threshold = 3e-2
# # # threshold = 2.5e-2
# # # threshold = 2e-2
threshold = 1e-2
threshold = 0.001
# # # threshold = 0.0012005   # good
# # # threshold = 0.0013005   # good
# # #
# # # # threshold = 0.0014
# # # # threshold = 0.00146
# # # # threshold = 0.00146104  # good
# # # # threshold = 0.00146504  # good
# # # # threshold = 0.00146604  # good
# # # # threshold = 0.0014665  # bad
# # # # threshold = 0.00146704  # bad
# # # # threshold = 0.00146804  # bad
# # # # threshold = 0.00147
# # # # threshold = 0.0015005   # zero
# # #
# # # threshold = 1e-3  # interesting output
# # threshold = 1e-4
# # # threshold = 2e-5
# # # threshold = 1.52e-5
# threshold = 1e-5
# # # # threshold = 1e-6
# # # # # threshold = 1e-7
# # # # threshold = 1e-8
# # # # threshold = 1e-9
#
# # # threshold = 1.5e-10
# # # threshold = 2e-10
# # # threshold = 3e-10
# # # threshold = 5e-10
# # threshold = 1e-10
# # threshold = 1e-12
# # threshold = 1e-15

precision = 3
precision = max(round(-1*log(threshold, 10))+3, 0)
precision = 16

degree = 1
# degree = 2
# degree = 3
# model = ps.SINDy(feature_names=["x", "y"],
#                 optimizer = ps.STLSQ(threshold=threshold))
model = ps.SINDy(feature_names=df.columns[1+shift:scale],
                 feature_library=ps.PolynomialLibrary(degree=degree),
                 # optimizer = ps.STLSQ(threshold=threshold),
                 )

print('scale:', scale)

main = True
main = False
hyperparametertuning = False
if not main:
    hyperparametertuning = True
if main:
    print('in main')
    model.fit(X, t=t)

    # print(model.coefficients())
    print(sum(abs(model.coefficients()[0, :])>0))

    total = False if shift == 1 else True
    meta = f'\n% filename: {fname}, \'total\' included: {total}, degree: {degree}, threshold: {threshold}, date: {str(datetime.datetime.now())} \n'
    print(meta)

    latex1 = f"""
    \subsection{{Scenario with degree {degree}, threshold {threshold} and precision {precision}}}

    Ordinary differential equations (ODEs) outputed by SINDy:

    \\begin{{verbatim}}"""
    print(latex1)

    tmp = sys.stdout
    my_result = StringIO()
    sys.stdout = my_result
    # print('hello world')  # output stored in my_result
    model.print(precision=precision)
    sys.stdout = tmp
    eqs = my_result.getvalue()
    print(eqs)

    latex2 = """\end{verbatim}

    """

    print(latex2)


    x_train = X
    # x_test = x_train[:100, :]
    x_test = x_train
    t_test = t
    dt = np.mean(t[1:] - t[:-1])
    # rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
    mse = model.score(x_test, t=dt, metric=mean_squared_error)
    rmse = mse**0.5


    print('mse lstsq', mse, 'rmse (lin reg):', rmse, 'done')
    sim_success = True
    try:
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
    except:
        sim_success = False
        print('simulation failed!!!')
        mse_sim = np.inf
    if sim_success:
        # x_test_sim = np.array([1e19, 1])
        if np.any(x_test_sim > 1e14):
            x_test_sim = 1e14
        rmse_sim = np.mean((x_test - x_test_sim) ** 2)**0.5

    printmses = f'rmse: {rmse:.4e}, rmse_sim: {rmse_sim:.4e}'

    # printmses = f"""
    # with: \\
    #     \\verb| mse: {mse:.4e}, mse_sim: {mse_sim:.4e} |\n\n
    # """
    print(printmses)

    # print(sum(abs(model.coefficients()[0, :])>0))


outdir = f'results_new{os.sep}auto-gen{os.sep}'

MAKEFILE = False
if MAKEFILE:

    # vi = fname.split('_')[3]
    outtxt = f'res_{vi}_{"un"*(not total)}tot{int(total)}_deg{degree}_thr{threshold}.txt'
    outpath = outdir + outtxt
    print(f'results written to {outpath}!!')
    f = open(outpath, 'w')
    f.write(meta + '\n')
    # print(printmses)
    f.write(eqs + '\n')
    f.write(printmses + '\n')
    f.close()

    import pickle
    outmodel = f'{outdir}model{outtxt[3:-4]}.pkl'
    with open(outmodel, 'wb') as f:
        pickle.dump(model, f)
    print(f'model_path = \'{outmodel}\'')

# model.print(lhs=["total"])

    # # model.simulate(X, t=t)
    # traj = model.simulate(X[0, :], t=t)
    # print(traj.shape)
    # print(sum(traj - X)**2)
    # print(np.sqrt(sum(sum((traj - X)**2))/(X.shape[0]*X.shape[1])))
    # print(len(t))
    #


# hyperparametertuning = True
# hyperparametertuning = False
if hyperparametertuning:

    print('in hypper tunn')
    # lorenz = True
    # lorenz = False
    # if lorenz:
    #     from scipy.integrate import solve_ivp
    #     from pysindy.utils import lorenz, lorenz_control, enzyme
    #
    #     np.random.seed(100)
    #
    #     # Initialize integrator keywords for solve_ivp to replicate the odeint defaults
    #     integrator_keywords = {}
    #     integrator_keywords['rtol'] = 1e-12
    #     integrator_keywords['method'] = 'LSODA'
    #     integrator_keywords['atol'] = 1e-12
    #
    #     dt = 0.002
    #
    #     t_train = np.arange(0, 10, dt)
    #     x0_train = [-8, 8, 27]
    #     t_train_span = (t_train[0], t_train[-1])
    #     x_train = solve_ivp(
    #         lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    #     ).y.T
    #
    #     t_test = np.arange(0, 15, dt)
    #     t_test_span = (t_test[0], t_test[-1])
    #     x0_test = np.array([8, 7, 15])
    #     x_test = solve_ivp(
    #         lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
    #     ).y.T
    #
    #     # print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
    #     # 1/0
    #     # Instantiate and fit the SINDy model
    #     feature_names = ['x', 'y', 'z']
    #     sparse_regression_optimizer = ps.STLSQ(threshold=0)  # default is lambda = 0.1
    #     model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
    #     model.fit(x_train, t=dt)
    #     model.print()
    #
    #     # Make coefficient plot for threshold scan

    MAKEFILE = True
    MAKEFILE = False
    def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test, degree):
        print('inside plot_pareto')
        dt = t_test[1] - t_test[0]
        mse = np.zeros(len(threshold_scan))
        mse_sim = np.zeros(len(threshold_scan))
        skipsim = True
        skipsim = False
        title = 'degree | threshold | stslq rmse | sim rmse\n'
        tuning = title
        # if MAKEFILE:
        #     # vi = fname.split('_')[3]
        #     vi = fname.split('_')[1][3:]
        #     # outdir = f'results_new{os.sep}auto-gen{os.sep}'
        #     outtxt = f'tuning_{vi}_deg{degree}.txt'
        #     outpath = outdir + outtxt
        #     print(f'tuning table written to {outpath}!!')
        #     print('\n\ntitle', title)
        #     with open(outpath, 'w') as f:
        #         f.write(title)

        for i, threshold in enumerate(threshold_scan):
            upper = 1e30
            opt.coef_ = coefs[i]
            try:
                mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)**0.5
            except:
                print(f'score UNSUCCESSFUL!!     ... for threshold: {threshold} and degree: {degree}')
                # print(x_test, x_test.shape)
                mse[i] = upper
                # mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
                # mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)**0.5
            print(f'rmse lstsq {i} done, rmse[{i}]: {mse[i]:.4e}')
            sim_success = True
            if not skipsim:
                # upper = 1e4
                mse_sim[i] = upper
                try:
                    x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
                except:
                    print('sim UNSUCCESSFUL!!')
                    sim_success = False
                    print('sim_success:', sim_success, 'rmse_sim[i]:', mse_sim[i])
                    # mse_sim[i] = np.inf
                    # mse_sim[i] = 1e50
                if sim_success:
                    print('sim SUCCESSFUL!!')
                        # not np.any(x_test_sim > upper)):
                    # if np.any(x_test_sim > upper):
                    #     print('x_test_sim > 1e4!!')
                    #     print(x_test_sim)
                    #     x_test_sim = 1e4
                    #     print('after:', x_test_sim)
                    # mse_sim[i] = min(np.mean((x_test - x_test_sim) ** 2), upper)
                    mse_sim[i] = min(np.mean((x_test - x_test_sim) ** 2)**0.5, upper)  # actually rmse

                    # if mse_sim[i] > upper:
                    #     mse_sim[i] = upper

                    # # print('after:', mse_sim[i])
                print(f'mse sim {i} done, rmse_sim[{i}]: {mse_sim[i]:.4e}')

            tuning += f'% {degree} & {threshold:.4e} & {mse[i]:.4e}  & {mse_sim[i]:.4e}  \\\\ \n'


        plt.figure()
        plt.semilogy(threshold_scan, mse, "bo")
        plt.semilogy(threshold_scan, mse, "b")
        time = str(datetime.datetime.now()) + f'rmse lstsq deg {degree}'
        plt.title(time, fontsize=20)
        plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
        plt.xlabel(r"$\lambda$ 0", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        # MAKEFILE = False
        if MAKEFILE:
            outpng = f'tuning_{vi}_deg{degree}_rmse.png'
            outpngpath = outdir + outpng
            plt.savefig(outpngpath, dpi=300)
            print(f'tunning pic written to {outpngpath}!!')

        if not skipsim:
            plt.figure()
            plt.semilogy(threshold_scan, mse_sim, "bo")
            plt.semilogy(threshold_scan, mse_sim, "b")
            time = str(datetime.datetime.now()) + f'rmse sim deg {degree}'
            plt.title(time, fontsize=20)
            plt.ylabel(r"$\dot{X}$ RMSE sim", fontsize=20)
            plt.xlabel(r"$\lambda$ sim", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True)
            if MAKEFILE:
                outpngpath = outdir + outpng[:-4] + '_sim.png'
                plt.savefig(outpngpath, dpi=300)
                print(f'tunning sim pic written to {outpngpath}!!')
        return tuning

    print('after trys')
    plt.show()
    # plot_pareto()


    threshold_scan = np.linspace(0, 1.0, 10)[:2002]
    # threshold_scan = np.linspace(0.9, 1, 10)[:2002]
    threshold_scan = np.linspace(0.0, 0.1, 10)[:2002]
    threshold_scan = [0, 1.52e-05, 1e-05, 1e-15  ]
    threshold_scan = np.linspace(0.0, 0.000000001, 4)[:2002]
    threshold_scan_std = np.linspace(0, 1.0, 11)[:2002]
    thrs_std = list(threshold_scan_std)
    std_exp = [10**(-i) for i in range(1, 10)]
    scaled_std = [10**(-2) * i for i in thrs_std]
    # thrs_std = std_exp
    # thrs_std = thrs_std + std_exp + scaled_std
    thrs_std = thrs_std[1:] + std_exp[:3] + scaled_std

    # deg3 = [10**(i) for i in range(-9, -2)]
    deg1 = list(threshold_scan_std) + [0, 0.923, 2.24, 3, 4, 10]
    deg1 = thrs_std
    deg2 = list(threshold_scan_std) + [0.025, 0.03, 0.04, 0.05, 0.1, 1.0]
    # deg2 = [0.04]
    deg2 = thrs_std
    deg3 = list(threshold_scan_std) + [1e-05, 1.52e-05, 1e-15, ]
    # deg2 = deg2[:3]
    # deg3 = deg3[:3]
    deg3 = thrs_std
    # thr_std = [(3, thr) for thr in threshold_scan_std] + [(2, thr) for thr in threshold_scan_std]
    # degthr = deg3 + deg2
    # degthr = deg2
    # degthr = deg3

    # threshold_scan = np.linspace(0.922, 0.924, 50)[:2002]
    coefs = []
    x_train = X
    # x_test = x_train[:100, :]
    x_test = x_train
    t_test = t
    dt = np.mean(t[1:] - t[:-1])
    rmse = mean_squared_error(x_train, np.zeros(x_train.shape), squared=False)
    # x_train_added_noise = x_train + np.random.normal(0, rmse / 10.0,  x_train.shape)

    x_train_added_noise = x_train
    eqss = ''
    tuning = ''
    for degree, threshold_scan in [(1, deg1), (2, deg2), (3, deg3)][0:1]:
        print(f' degree: {degree}')
        # degree = deg + 2
        for i, threshold in enumerate(threshold_scan):
            print(f'   threshold: {threshold}')
            # print('inside loop')
            # print(config)
            # Instantiate and fit the SINDy model
            feature_names = ['x', 'y', 'z']
            # sparse_regression_optimizer = ps.STLSQ(threshold=0)  # default is lambda = 0.1
            # model.fit(x_train, t=dt)

            # degree, threshold = config
            sparse_regression_optimizer = ps.STLSQ(threshold=threshold)
            # model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
            model = ps.SINDy(feature_names=df.columns[1 + shift:scale],
                             feature_library=ps.PolynomialLibrary(degree=degree),
                             optimizer=sparse_regression_optimizer)
            # model = ps.SINDy(feature_names=feature_names,
            #                  optimizer=sparse_regression_optimizer)
            model.fit(x_train_added_noise, t=dt, quiet=True)
            print(f'   fitted: {i}, with thr: {degree, threshold}')
            coefs.append(model.coefficients())

            # save equation(system) to file):
            tmp = sys.stdout
            my_result = StringIO()
            sys.stdout = my_result
            # print('hello world')  # output stored in my_result
            print(threshold)
            # 1/0
            # a = log(threshold, 13)
            precision = max(round(-1 * log(max(threshold, 1e-16), 10)) + 3, 0)
            precision = 16
            model.print(precision=precision)
            sys.stdout = tmp
            eqss += f'system of odes for degree: {degree} and threshold: {threshold} \n' + my_result.getvalue() + '\n'


        # print('after loop')
        # threshold_scan = [thr for deg, thr in degthr]
        tuning += plot_pareto(coefs, sparse_regression_optimizer, model,
                    threshold_scan, x_test, t_test, degree) + '\n'
        # plot_pareto(coefs, sparse_regression_optimizer, model,
        #             threshold_scan, x_test, t_test, degree)

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open(outdir + f'eqs_deg1_{vi}_{now}.txt', 'w') as f:
        f.write(eqss)
    if MAKEFILE:
        outpath = outdir + f'tuning_{vi}.txt'
        print(f'tuning table written to {outpath}!!')
        # with open(outpath, 'a') as f:
        with open(outpath, 'w') as f:
            f.write(tuning + '\n')

    # plt.plot(t, t, 'k--')
    # 4 * 10^4 lstmse at 0.923
    # 1 * 10^4 lstmse at 0

print('scale:', scale)
