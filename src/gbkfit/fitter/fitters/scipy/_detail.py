
FITTER_KWARGS_REQ = []

FITTER_KWARGS_OPT = dict(
    jac='2-point', method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08,
    x_scale=1.0, loss='linear', f_scale=1.0, diff_step=np.finfo(float).eps,
    tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0)

PARAMS_KWARGS_REQ = ['init']

PARAMS_KWARGS_OPT = dict(x_scale=1.0, diff_step=np.finfo(float).eps)
