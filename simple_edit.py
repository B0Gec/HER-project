import datetime
import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pysindy as ps
import numpy as np
# print(pysindy.__version__, sys.version)
# print(warnings.filters)

from loadit import loaded, fnamei, fnameii, fnameiii

MAKEFILE = True

fname = fnamei
df = loaded(fname)
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

# 4 * 10^4 lstmse at 0.923
# 1 * 10^4 lstmse at 0
threshold = 0  # mse: 9.9612e+03, mse_sim: inf
# threshold = 0.923  # mse: 4.7354e+04, mse_sim: 2.0143e+08
threshold = 1
# # threshold = 2
threshold = 2.24
threshold = 3
# threshold = 3.5

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
# # # threshold = 1e-2
# # # threshold = 0.001
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

degree = 1
# degree = 2
degree = 3
# model = ps.SINDy(feature_names=["x", "y"],
#                 optimizer = ps.STLSQ(threshold=threshold))
model = ps.SINDy(feature_names=df.columns[1+shift:scale],
                 feature_library=ps.PolynomialLibrary(degree=degree),
                 optimizer = ps.STLSQ(threshold=threshold))

main = True
main = False
if main:
    model.fit(X, t=t)

    # print(model.coefficients())
    print(sum(abs(model.coefficients()[0, :])>0))

    total = False if shift == 1 else True
    precision = 3
    meta = f'\n% filename: {fname}, \'total\' included: {total}, degree: {degree}, threshold: {threshold}, date: {str(datetime.datetime.now())} \n'
    print(meta)

    latex1 = f"""
    \subsection{{Scenario with degree {degree}, threshold {threshold} and precision {precision}}}

    Ordinary differential equations (ODEs) outputed by SINDy:

    \\begin{{verbatim}}"""
    print(latex1)

    from io import StringIO
    import sys
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


    # print('mse lstsq', i, 'done')
    sim_success = True
    try:
        x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
    except:
        sim_success = False
        mse_sim = np.inf
    if sim_success:
        if np.any(x_test_sim > 1e4):
            x_test_sim = 1e4
        mse_sim = np.mean((x_test - x_test_sim) ** 2)

    printmses = f'mse: {mse:.4e}, mse_sim: {mse_sim:.4e}'
    printmses = f"""
    with: \\
        \\verb| mse: {mse:.4e}, mse_sim: {mse_sim:.4e} |\n\n
    """
    print(printmses)

    # print(sum(abs(model.coefficients()[0, :])>0))



MAKEFILE = False
if MAKEFILE:

    vi = fname.split('_')[3]
    outdir = f'results{os.sep}auto-gen{os.sep}'
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


hyperparametertuning = True
# hyperparametertuning = False
if hyperparametertuning:

    lorenz = True
    lorenz = False
    if lorenz:
        from scipy.integrate import solve_ivp
        from pysindy.utils import lorenz, lorenz_control, enzyme

        np.random.seed(100)

        # Initialize integrator keywords for solve_ivp to replicate the odeint defaults
        integrator_keywords = {}
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['method'] = 'LSODA'
        integrator_keywords['atol'] = 1e-12

        dt = 0.002

        t_train = np.arange(0, 10, dt)
        x0_train = [-8, 8, 27]
        t_train_span = (t_train[0], t_train[-1])
        x_train = solve_ivp(
            lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
        ).y.T

        t_test = np.arange(0, 15, dt)
        t_test_span = (t_test[0], t_test[-1])
        x0_test = np.array([8, 7, 15])
        x_test = solve_ivp(
            lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
        ).y.T

        # print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
        # 1/0
        # Instantiate and fit the SINDy model
        feature_names = ['x', 'y', 'z']
        sparse_regression_optimizer = ps.STLSQ(threshold=0)  # default is lambda = 0.1
        model = ps.SINDy(feature_names=feature_names, optimizer=sparse_regression_optimizer)
        model.fit(x_train, t=dt)
        model.print()

        # Make coefficient plot for threshold scan

    MAKEFILE = True
    def plot_pareto(coefs, opt, model, threshold_scan, x_test, t_test, degree):
        dt = t_test[1] - t_test[0]
        mse = np.zeros(len(threshold_scan))
        mse_sim = np.zeros(len(threshold_scan))
        skipsim = True
        skipsim = False
        tuning = ""
        title = 'degree | threshold | stslq mse | sim mse\n'
        if MAKEFILE:
            vi = fname.split('_')[3]
            outdir = f'results{os.sep}auto-gen{os.sep}'
            outtxt = f'tuning_{vi}_deg{degree}.txt'
            outpath = outdir + outtxt
            print(f'tuning table written to {outpath}!!')
            print('\n\ntitle', title)
            with open(outpath, 'w') as f:
                f.write(title)

        for i, threshold in enumerate(threshold_scan):
            opt.coef_ = coefs[i]
            mse[i] = model.score(x_test, t=dt, metric=mean_squared_error)
            print(f'mse lstsq {i} done, mse[{i}]: {mse[i]}')
            sim_success = True
            if not skipsim:
                try:
                    x_test_sim = model.simulate(x_test[0, :], t_test, integrator="odeint")
                except:
                    sim_success = False
                    mse_sim[i] = np.inf
                if sim_success:
                    if np.any(x_test_sim > 1e4):
                        x_test_sim = 1e4
                    mse_sim[i] = np.sum((x_test - x_test_sim) ** 2)
                else:
                    mse_sim[i] = np.inf
                print(f'mse sim {i} done, mse_sim[{i}]: {mse_sim[i]}')

            tuning = f'% {degree} & {threshold:.4e} & {mse[i]:.4e}  & {mse_sim[i]:.4e}  \\\\'
            if MAKEFILE:
                print(f'tuning table written to {outpath}!!')
                with open(outpath, 'a') as f:
                    f.write(tuning + '\n')


        plt.figure()
        plt.semilogy(threshold_scan, mse, "bo")
        plt.semilogy(threshold_scan, mse, "b")
        time = str(datetime.datetime.now()) + f'mse lstsq deg {degree}'
        plt.title(time, fontsize=20)
        plt.ylabel(r"$\dot{X}$ RMSE", fontsize=20)
        plt.xlabel(r"$\lambda$ 0", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        # MAKEFILE = False
        if MAKEFILE:
            outpng = f'tuning_{vi}_deg{degree}_mse.png'
            outpngpath = outdir + outpng
            plt.savefig(outpngpath, dpi=300)
            print(f'tunning pic written to {outpngpath}!!')

        if not skipsim:
            plt.figure()
            plt.semilogy(threshold_scan, mse_sim, "bo")
            plt.semilogy(threshold_scan, mse_sim, "b")
            time = str(datetime.datetime.now()) + f'mse sim deg {degree}'
            plt.title(time, fontsize=20)
            plt.ylabel(r"$\dot{X}$ RMSE sim", fontsize=20)
            plt.xlabel(r"$\lambda$ sim", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True)
            outpngpath = outdir + outpng[:-4] + '_sim.png'
            plt.savefig(outpngpath, dpi=300)
            print(f'tunning sim pic written to {outpngpath}!!')

    plt.show()
    # plot_pareto()


    threshold_scan = np.linspace(0, 1.0, 10)[:2002]
    # threshold_scan = np.linspace(0.9, 1, 10)[:2002]
    threshold_scan = np.linspace(0.0, 0.1, 10)[:2002]
    threshold_scan = [0, 1.52e-05, 1e-05, 1e-15  ]
    threshold_scan = np.linspace(0.0, 0.000000001, 4)[:2002]
    threshold_scan_std = np.linspace(0, 1.0, 10)[:2002]
    # deg3 = [10**(i) for i in range(-9, -2)]
    deg2 = list(threshold_scan_std) + [0.1, 0.03, 0.05, 0.025, ]
    deg3 = list(threshold_scan_std)[3:] + [1e-05, 1.52e-05, 1e-15, ]
    # deg2 = deg2[:3]
    # deg3 = deg3[:3]
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
    for degree, threshold_scan in [(2, deg2), (3, deg3)][1:2]:
        # degree = deg + 2
        for i, threshold in enumerate(threshold_scan):
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
            print(f'fitted: {i}, with thr: {degree, threshold}')


            coefs.append(model.coefficients())

        # print('after loop')
        # threshold_scan = [thr for deg, thr in degthr]
        plot_pareto(coefs, sparse_regression_optimizer, model,
                    threshold_scan, x_test, t_test, degree)
        plot_pareto(coefs, sparse_regression_optimizer, model,
                    threshold_scan, x_test, t_test, degree)


    # plt.plot(t, t, 'k--')
    # 4 * 10^4 lstmse at 0.923
    # 1 * 10^4 lstmse at 0

