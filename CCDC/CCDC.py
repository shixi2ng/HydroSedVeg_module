import concurrent.futures
import traceback
import copy
from scipy.stats import chi2
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from osgeo import gdal
import basic_function as bf
import ast
from tqdm import tqdm
from itertools import repeat, product
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils import check_random_state
import json
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
import os
import re
import xgboost as xgb
# CCDC 12.30 version - Zhe Zhu, EROS USGS
# It is based on 7 bands fitting for Iterative Seasonal, Linear, and Break Models
# This function works for analyzing one line of time series pixel

global w
w = 2 * np.pi / 365.25


def flatten(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):   # 如果元素还是 list，就递归展开
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def glmnet_lasso_single(X, y, lambda_val=1.0, fit_intercept=True, standardize=True, max_iter=10000, tol=1e-4):
    """
    Simplified glmnet-style Lasso for a single lambda value.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
    - y: ndarray, shape (n_samples,)
    - lambda_val: float, lambda (alpha) value for Lasso
    - fit_intercept: bool, whether to fit and recover intercept
    - standardize: bool, whether to standardize X
    - max_iter: int, maximum iterations
    - tol: float, tolerance for convergence

    Returns:
    - dict with keys: 'lambda', 'beta', 'a0', 'df', 'rss', 'r2'
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).flatten()

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    model = Lasso(alpha=lambda_val, fit_intercept=fit_intercept,
                  max_iter=max_iter, tol=tol)
    model.fit(X_scaled, y)

    beta = model.coef_
    intercept = model.intercept_ if fit_intercept else 0.0

    y_pred = model.predict(X_scaled)
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - rss / tss if tss > 1e-12 else 0.0

    return {
        'lambda': lambda_val,
        'beta': beta,         # shape (n_features,)
        'a0': intercept,      # scalar
        'df': np.sum(np.abs(beta) > 1e-8),
        'rss': rss,
        'r2': r2
    }

# def lassofit(x, y, optimum_lambda):
#
#     nobs, nvars = x.shape
#
#     # Set default options.
#     options = {
#         'weights': [],
#         'alpha': 1.0,
#         'lambda': optimum_lambda,  # optimum lambda for CCDC
#         'standardize': True,
#         'thresh': 1E-4,
#         'HessianExact': False,
#     }
#
#     # ======= 参数准备 =======
#     weights = options.get('weights', np.ones(nobs))
#     alpha = options.get('alpha', 1.0)
#     lambda_seq = options.get('lambda', None)
#     standardize = options.get('standardize', True)
#     thresh = options.get('thresh', 1e-4)
#
#     if len(options['weights']) == 0:
#         options['weights'] = np.ones((nobs, 1))
#
#     # ======= 中心化 + 权重下的 null deviance =======
#     ybar = np.dot(y.T, weights) / np.sum(weights)
#     nulldev = np.dot((y - ybar)**2, weights) / np.sum(weights)
#
#     # ======= 是否标准化输入 =======
#     if standardize:
#         x_mean = np.average(x, axis=0, weights=weights)
#         x_std = np.std(x, axis=0)
#         x_std[x_std == 0] = 1.0
#         x = (x - x_mean) / x_std
#
#     # convert lambda to ascending order like glmnet.m does internally
#     lambda_list = np.sort(lambda_list)[::-1]
#     nlam = len(lambda_list)
#
#     # storage
#     a0 = np.zeros(nlam)  # intercepts
#     beta = np.zeros((nvars, nlam))  # coefficients
#     dev = np.zeros(nlam)  # R²
#     df = np.zeros(nlam, dtype=int)  # degrees of freedom (nonzero coef)
#
#     # for each lambda value
#     for i in range(nlam):
#         lam = lambda_list[i]
#         model = Lasso(alpha=lam, fit_intercept=True, tol=thresh, max_iter=10000)
#         model.fit(x, y)
#
#         a0[i] = model.intercept_
#         beta[:, i] = model.coef_
#         df[i] = np.sum(model.coef_ != 0)
#
#         y_pred = model.predict(x)
#         rss = np.dot((y - y_pred) ** 2, weights)
#         dev[i] = 1 - rss / np.dot((y - ybar) ** 2, weights)
#
#     # return in a glmnet-style dictionary
#     fit = {
#         'a0': a0,  # intercepts
#         'beta': beta,  # coefficients matrix (nvars x nlam)
#         'df': df,  # non-zero coefficient count
#         'dev': dev,  # R^2 per lambda
#         'nulldev': nulldev,  # total sum of squares
#         'lambda': lambda_list,  # regularization values
#         'dim': (nvars, nlam),  # dimensions of beta
#         'class': 'elnet'  # consistent with glmnet
#     }
#
#     return fit

# def lass2(x, y, optimum_lambda):
#     nobs, nvars = x.shape
#
#     # Set default options.
#     options = {
#         'weights': [],
#         'alpha': 1.0,
#         'lambda': optimum_lambda,  # optimum lambda for CCDC
#         'standardize': True,
#         'thresh': 1E-4,
#         'HessianExact': False,
#     }
#
#     if len(options['weights']) == 0:
#         options['weights'] = np.ones((nobs, 1))
#     ybar = np.dot(y.T, options['weights']) / np.sum(options['weights'])
#     nulldev = np.dot((y - ybar) ** 2, options['weights']) / np.sum(options['weights'])
#
#
#     isd = float(options['standardize'])
#     thresh = options['thresh']
#     lambda_ = options['lambda']
#     flmin = 1.0
#     ulam = -np.sort(-lambda_)
#     nlam = len(lambda_)
#     parm = options['alpha']
#
#     # ======= 权重处理：拟合时 sqrt(w) 调整 ====
#     x_weighted = x * np.sqrt(weights[:, np.newaxis])
#     y_weighted = y * np.sqrt(weights)
#
#     # ======= 模型拟合（路径方式）=======
#     if lambda_seq is None:
#         alphas, coefs, _ = lasso_path(x_weighted, y_weighted, eps=1e-3, fit_intercept=True)
#         intercepts = np.mean(y_weighted) - np.dot(np.mean(x_weighted, axis=0), coefs)
#         rsq = 1 - np.sum((y_weighted[:, None] - x_weighted @ coefs) ** 2, axis=0) / np.var(y_weighted) / len(y)
#     else:
#         alphas = np.sort(lambda_seq)[::-1]
#         coefs = []
#         intercepts = []
#         rsq = []
#         for a in alphas:
#             model = Lasso(alpha=a, fit_intercept=True, max_iter=10000, tol=thresh)
#             model.fit(x_weighted, y_weighted)
#             coefs.append(model.coef_)
#             intercepts.append(model.intercept_)
#             rsq.append(model.score(x_weighted, y_weighted))
#         coefs = np.array(coefs).T
#         intercepts = np.array(intercepts)
#         rsq = np.array(rsq)
#
#     # ======= 构造结果结构体（仿 MATLAB）=======
#     fit = {
#         'a0': intercepts,
#         'beta': coefs,
#         'dev': rsq,
#         'nulldev': nulldev,
#         'df': np.sum(np.abs(coefs) > 1e-8, axis=0),
#         'lambda': alphas,
#         'npasses': 0,
#         'jerr': 0,
#         'dim': coefs.shape,
#         'class': 'elnet'
#     }
#
#     return fit
#
#
# def fix_lam(lam):
#     new_lam = lam.copy()
#     llam = np.log(lam)
#     new_lam[0] = np.exp(2 * llam[1] - llam[2])
#     return new_lam



def autoTSFit(x, y, df):
    """
    Auto Trends and Seasonal Fit between breaks
    Args:
    - x: Julian day (e.g., [1, 2, 3])
    - y: Predicted reflectances (e.g., [0.1, 0.2, 0.3])
    - df: Degree of freedom (num_c)

    Returns:
    - fit_coeff: Fitted coefficients
    - rmse: Root mean square error
    - v_dif: Differences between observed and predicted values
    """

    # Initialize fit coefficients
    fit_coeff = np.zeros(8)

    # Build X matrix
    X = np.zeros((len(x), df - 1))
    X[:, 0] = x

    if df >= 4:
        X[:, 1] = np.cos(w * x)
        X[:, 2] = np.sin(w * x)

    if df >= 6:
        X[:, 3] = np.cos(2 * w * x)
        X[:, 4] = np.sin(2 * w * x)

    if df >= 8:
        X[:, 5] = np.cos(3 * w * x)
        X[:, 6] = np.sin(3 * w * x)

    # Lasso fit with lambda = 20
    lasso = Lasso(alpha=20/10000, fit_intercept=True, max_iter=25000)
    lasso.fit(X, y)

    # Set fit coefficients
    fit_coeff[0] = lasso.intercept_
    fit_coeff[1: df] = lasso.coef_

    # Predicted values
    yhat = autoTSPred(x, fit_coeff)
    fit_diff = y - yhat
    fit_rmse = np.linalg.norm(fit_diff) / np.sqrt(len(x) - df)

    return fit_coeff, fit_rmse, fit_diff


def autoTSPred(outfitx, fit_coeff):
    """
    Auto Trends and Seasonal Predict
    Args:
    - outfitx: Julian day (e.g., [1, 2, 3])
    - fit_coeff: Fitted coefficients

    Returns:
    - outfity: Predicted reflectances
    """
      # annual cycle

    # Construct the design matrix
    X = np.column_stack([
        np.ones_like(outfitx),  # overall ref
        outfitx,                # trending
        np.cos(w * outfitx),    # seasonality
        np.sin(w * outfitx),
        np.cos(2 * w * outfitx),  # bimodal seasonality
        np.sin(2 * w * outfitx),
        np.cos(3 * w * outfitx),  # trimodal seasonality
        np.sin(3 * w * outfitx)
    ])

    outfity = X @ fit_coeff  # matrix multiplication

    return outfity


def update_cft(i_span, n_times, min_num_c, mid_num_c, max_num_c, num_c):
    """
    Determine the time series model based on the span of observations.

    Args:
    - i_span: Span of observations
    - n_times: Multiplier for the number of coefficients
    - min_num_c: Minimum number of coefficients
    - mid_num_c: Middle number of coefficients
    - max_num_c: Maximum number of coefficients
    - num_c: Current number of coefficients

    Returns:
    - update_num_c: Updated number of coefficients
    """
    if i_span < mid_num_c * n_times:
        # Start with 4 coefficients model
        update_num_c = min(min_num_c, num_c)
    elif i_span < max_num_c * n_times:
        # Start with 6 coefficients model
        update_num_c = min(mid_num_c, num_c)
    else:
        # Start with 8 coefficients model
        update_num_c = min(max_num_c, num_c)

    return update_num_c


def TrendSeasonalFit_v12_30Line(doy_arr, inform_arr, num_c: int = 8, Thr_change_detect=0.99, min_obs4stable: int=6, ):

    if doy_arr.shape[0] != inform_arr.shape[0]:
        raise Exception('The doy and inform is not consistent in size！')

    # Define constants %
    # maximum number of coefficient required
    # 2 for tri-modal; 2 for bi-modal; 2 for seasonality; 2 for linear;
    min_num_c = 4 # Minimum number of coefficients
    mid_num_c = 6 # Middle number of coefficients
    max_num_c = 8 # Maximum number of coefficients
    n_times = 3 # Multiplier for the number of coefficients
    number_fitcurve = 0
    num_yrs = 365.25
    num_byte = 2
    yrs_thr = 4

    # Result list
    CCDC_result = []

    # Define var
    num_time, num_band = inform_arr.shape[0], inform_arr.shape[1]
    x_arr = doy_arr
    y_arr = inform_arr
    deltay_arr = y_arr[1:, :] - y_arr[:-1, :]
    adj_rmse = np.median(np.abs(deltay_arr), axis=0)
    i_end = n_times * min_num_c - 1
    i_start = 0

    # Fitcurve indicator
    # record the start of the model initialization (0=>initial;1=>done)
    BL_train = 0 # No break found at the beggning
    number_fitcurve += 1
    record_fitcurve = copy.deepcopy(number_fitcurve)
    Thr_change_detect = chi2.ppf(Thr_change_detect, num_band)
    Thrmax_change_detect = chi2.ppf(1 - 0.01, num_band)

    try:
        # CCDC procedure
        while i_end < len(x_arr) - min_obs4stable:
            i_span = i_end - i_start + 1
            time_span = (x_arr[i_end] - x_arr[i_start]) / num_yrs

            if i_span >= n_times * min_num_c and time_span >= yrs_thr:

                # initializing model
                if BL_train == 0:

                    fit_coeff = np.zeros((max_num_c, num_band))
                    bands_fit_rmse = np.zeros(num_band)
                    v_dif = np.zeros(num_band)
                    bands_fit_diff = np.zeros((i_end - i_start + 1, num_band))

                    # fitting
                    for band_index in range(num_band):
                        fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[i_start:i_end + 1], y_arr[i_start:i_end + 1, band_index], min_num_c)

                        # normalized to z - score
                        # minimum rmse
                        mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                        # compare the first clear obs
                        v_start = bands_fit_diff[0, band_index] / mini_rmse
                        # compare the last clear observation
                        v_end = bands_fit_diff[-1, band_index] / mini_rmse
                        # normalized slope values
                        v_slope = fit_coeff[1, band_index] * (x_arr[i_end] - x_arr[i_start]) / mini_rmse
                        # difference in model initialization
                        v_dif[band_index] = abs(v_slope) + abs(v_start) + abs(v_end)

                    v_dif = np.linalg.norm(v_dif) ** 2

                    # find stable start for each curve
                    if v_dif > Thr_change_detect:
                        # MOVE FORWARD
                        i_start += 1
                        i_end += 1
                        continue

                    else:
                        # model ready! Count difference of i for each itr
                        BL_train = 1
                        i_count = 0

                        # find the previous break point
                        if number_fitcurve == record_fitcurve:
                            i_break = 0
                        else:
                            tmp = np.where(x_arr >= CCDC_result[number_fitcurve - 2]['t_break'])[0]
                            i_break = int(tmp[0]) if tmp.size > 0 else 0

                        if i_start > i_break:
                            # model fit at the beginning of the time series
                            for i_ini in range(i_start - 1, i_break - 1, -1):

                                ini_conse = min(i_start - i_break, min_obs4stable)

                                # change vector magnitude
                                v_dif = np.zeros((ini_conse, num_band))
                                v_dif_mag = np.zeros((ini_conse, num_band))
                                vec_mag = np.zeros(ini_conse)

                                for i_conse in range(ini_conse):
                                    for band_index in range(num_band):
                                        v_dif_mag[i_conse, band_index] = y_arr[i_ini - i_conse, band_index] - autoTSPred(x_arr[i_ini - i_conse], fit_coeff[:, band_index])
                                        mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                                        v_dif[i_conse, band_index] = v_dif_mag[i_conse, band_index] / mini_rmse
                                    vec_mag[i_conse] = np.linalg.norm(v_dif[i_conse, :]) ** 2

                                if min(vec_mag) > Thr_change_detect:
                                    break
                                elif vec_mag[0] > Thrmax_change_detect:
                                    x_arr = np.delete(x_arr, i_ini)
                                    y_arr = np.delete(y_arr, i_ini, axis=0)
                                    i_end -= 1

                                # update new_i_start if i_ini is not a confirmed break
                                i_start = i_ini

                        if number_fitcurve == record_fitcurve and i_start - i_break >= min_obs4stable:
                            fit_coeff = np.zeros((max_num_c, num_band))
                            bands_fit_rmse = np.zeros(num_band)
                            bands_fit_diff = np.zeros((i_start - i_break, num_band))
                            qa = 10

                            for band_index in range(num_band):
                                fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index]  = autoTSFit(x_arr[i_break: i_start], y_arr[i_break: i_start, band_index], min_num_c)

                            CCDC_result.append({
                                't_start': x_arr[0],
                                't_end': x_arr[i_start -1],
                                't_break': x_arr[i_start],
                                'pos': num_time - 1,
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                'change_prob': 1,
                                'num_obs': i_start-i_break + 1,
                                'category': qa + min_num_c,
                                'magnitude': -np.median(v_dif_mag, axis=0)
                            })
                            number_fitcurve += 1

                # continuous monitoring started!!!
                if BL_train == 1:

                    IDs = np.arange(i_start, i_end + 1)
                    i_span = i_end - i_start + 1

                    # determine the time series model
                    update_num_c = update_cft(i_span, n_times, min_num_c, mid_num_c, max_num_c, num_c)

                    # initial model fit when there are not many obs
                    if i_count == 0 or i_span <= max_num_c * n_times:

                        # update i_count at each interation
                        i_count = x_arr[i_end] - x_arr[i_start]

                        # defining computed variables
                        fit_coeff = np.zeros((max_num_c, num_band))
                        bands_fit_rmse = np.zeros(num_band)
                        bands_fit_diff = np.zeros((len(IDs), num_band))
                        qa = 0

                        for band_index in range(num_band):
                            fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[IDs], y_arr[IDs, band_index], update_num_c)

                        if number_fitcurve > len(CCDC_result):
                            CCDC_result.append({
                                't_start': x_arr[i_start],
                                't_end': x_arr[i_end],
                                't_break': 0,
                                'pos': num_time - 1,
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                'change_prob': 0,
                                'num_obs': i_end - i_start + 1,
                                'category': qa + update_num_c,
                                'magnitude': np.zeros(num_band)
                            })
                        else:
                            CCDC_result[number_fitcurve - 1].update({
                                't_start': x_arr[i_start],
                                't_end': x_arr[i_end],
                                't_break': 0,
                                'pos': num_time - 1,
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                'change_prob': 0,
                                'num_obs': i_end - i_start + 1,
                                'category': qa + update_num_c,
                                'magnitude': np.zeros(num_band)
                            })

                        v_dif = np.zeros((min_obs4stable, num_band))
                        v_dif_mag = np.zeros((min_obs4stable, num_band))
                        vec_mag = np.zeros(min_obs4stable)

                        for i_conse in range(min_obs4stable):
                            for band_index in range(num_band):
                                v_dif_mag[i_conse, band_index] = y_arr[i_end + i_conse + 1, band_index] - autoTSPred(x_arr[i_end + i_conse + 1], fit_coeff[:, band_index])[0]
                                mini_rmse = max(adj_rmse[band_index], bands_fit_rmse[band_index])
                                v_dif[i_conse, band_index] = v_dif_mag[i_conse, band_index] / mini_rmse
                            vec_mag[i_conse] = np.linalg.norm(v_dif[i_conse, :]) ** 2
                        IDsOld = copy.deepcopy(IDs)

                    else:

                        if x_arr[i_end] - x_arr[i_start] >= 1.33 * i_count:
                            i_count = x_arr[i_end] - x_arr[i_start]
                            fit_coeff = np.zeros((max_num_c, num_band))
                            bands_fit_rmse = np.zeros(num_band)
                            bands_fit_diff = np.zeros((len(IDs), num_band))
                            qa = 0

                            for band_index in range(num_band):
                                fit_coeff[:, band_index], bands_fit_rmse[band_index], bands_fit_diff[:, band_index] = autoTSFit(x_arr[IDs], y_arr[IDs, band_index], update_num_c)

                            CCDC_result[number_fitcurve - 1].update({
                                'coeffs': fit_coeff,
                                'bands_fit_rmse': bands_fit_rmse,
                                'num_obs': i_end - i_start + 1,
                                'category': qa + update_num_c
                            })
                            IDsOld = IDs.copy()

                        # record time of curve end
                        CCDC_result[number_fitcurve - 1]['t_end'] = x_arr[i_end]
                        num4rmse = int(n_times * CCDC_result[number_fitcurve - 1]['category'])
                        d_rt = x_arr[IDsOld] - x_arr[i_end + min_obs4stable]
                        d_yr = np.abs(np.round(d_rt / num_yrs) * num_yrs - d_rt)
                        sorted_index = np.argsort(d_yr)[: num4rmse]

                        temp_change_rmse = np.zeros(num_band)
                        for band_index in range(num_band):
                            rows_t =  IDsOld[sorted_index] - IDsOld[0]
                            temp_change_rmse[band_index] = np.linalg.norm(bands_fit_diff[rows_t, band_index]) / np.sqrt(num4rmse - CCDC_result[number_fitcurve - 1]['category'])

                        v_dif[:-1, :] = v_dif[1:, :]
                        v_dif[-1, :] = 0
                        v_dif_mag[:-1, :] = v_dif_mag[1:, :]
                        v_dif_mag[-1, :] = 0
                        vec_mag[:-1] = vec_mag[1:]
                        vec_mag[-1] = 0

                        for band_index in range(num_band):
                            v_dif_mag[-1, band_index] = y_arr[i_end + min_obs4stable, band_index] - autoTSPred(x_arr[i_end + min_obs4stable], fit_coeff[:, band_index])[0]
                            mini_rmse = max(adj_rmse[band_index], temp_change_rmse[band_index])
                            v_dif[-1, band_index] = v_dif_mag[-1, band_index] / mini_rmse
                        vec_mag[-1] = np.linalg.norm(v_dif[-1, :]) ** 2

                    if min(vec_mag) > Thr_change_detect:
                        CCDC_result[number_fitcurve - 1]['t_break'] = x_arr[i_end + 1]
                        CCDC_result[number_fitcurve - 1]['change_prob'] = 1
                        CCDC_result[number_fitcurve - 1]['magnitude'] = np.median(v_dif_mag, axis=0)

                        number_fitcurve += 1
                        i_start = i_end + 1
                        BL_train = 0

                    elif vec_mag[0] > Thrmax_change_detect:
                        x_arr = np.delete(x_arr, i_end + 1)
                        y_arr = np.delete(y_arr, i_end + 1, axis=0)
                        i_end -= 1
            i_end += 1

        # Two ways for processing the end of the time series
        if BL_train == 1:
            #  if no break find at the end of the time series
            #  define probability of change based on conse
            id_last = 0
            for i_conse in range(min_obs4stable - 1, -1, -1):
                if vec_mag[i_conse] <= Thr_change_detect:
                    id_last = i_conse
                    break

            CCDC_result[number_fitcurve - 1]['change_prob'] = (min_obs4stable - id_last) / min_obs4stable
            CCDC_result[number_fitcurve - 1]['t_end'] = x_arr[-min_obs4stable + id_last]

            if min_obs4stable > id_last:
                CCDC_result[number_fitcurve - 1]['t_break'] = x_arr[-min_obs4stable + id_last + 1]
                CCDC_result[number_fitcurve - 1]['magnitude'] = np.median(v_dif_mag[id_last: min_obs4stable, :], axis=0)

        elif BL_train == 0:

            #  2) if break find close to the end of the time series
            #  Use [min_obs4stable,min_num_c*n_times+min_obs4stable) to fit curve

            if number_fitcurve == record_fitcurve:
                i_start = 0
            else:
                tmp = np.where(x_arr >= CCDC_result[number_fitcurve - 2]['t_break'])[0]
                i_start = int(tmp[0]) if tmp.size > 0 else 0

            # Check if there is enough data
            if len(x_arr) - i_start >= min_obs4stable:
                # Define computed variables
                fit_cft = np.zeros((max_num_c, num_band))
                rmse = np.zeros(num_band)
                fit_diff = np.zeros((len(x_arr[i_start:]), num_band))
                qa = 20

                for band_index in range(num_band):
                    fit_cft[:, band_index], rmse[band_index], fit_diff[:, band_index] = autoTSFit(x_arr[i_start:], y_arr[i_start:, band_index], min_num_c)

                # Record information
                CCDC_result.append({
                    't_start': x_arr[i_start],
                    't_end': x_arr[-1],
                    't_break': 0,
                    'pos': num_time - 1,
                    'coeffs': fit_cft,
                    'rmse': rmse,
                    'change_prob': 0,
                    'num_obs': len(x_arr[i_start:]),
                    'category': qa + min_num_c,
                    'magnitude': np.zeros(num_band)
                })
    except:
        print(traceback.format_exc())
        print('Error')
        return []

    return CCDC_result


def plot_ccdc_segments(doy_arr, index_arr, CCDC_result, output_figpath, min_year, title="CCDC Fitting", dpi=100):
    """
    多波段绘图（每行一个波段）。
    - doy_arr: (T,)
    - index_arr: (T, n_bands) 或 (T,)
    - CCDC_result: list of dict，且每个 dict 至少包含：
        {
          't_start': int/float,
          't_end':   int/float,
          't_break': 可选，int/float（<=0 或缺失则不画竖线），
          'coeffs':  list，长度 n_bands；每个元素是该波段的系数向量(1D)
        }
    - autoTSPred(t_range, coeff_vec) 已存在
    """
    try:
        # 规范化输入
        x = np.asarray(doy_arr, dtype=float).ravel()
        Y = np.asarray(index_arr)
        if Y.ndim == 1:
            Y = Y[:, None]
        T, n_bands = Y.shape

        # 公共年份刻度（x 仍用“天”）
        min_doy = int(np.floor(np.nanmin(x)))
        max_doy = int(np.ceil (np.nanmax(x)))
        start_year = int(np.floor(min_doy / 365.25 + min_year))
        end_year   = int(np.ceil (max_doy / 365.25 + min_year))
        year_tick_positions = [int((yr - min_year) * 365.25 + 182.75) for yr in range(start_year, end_year)]
        yline_positions     = [int((yr - min_year + 1) * 365.25)      for yr in range(start_year, end_year)]
        year_labels         = [str(yr) for yr in range(start_year, end_year)]

        # 画布
        fig, axes = plt.subplots(n_bands, 1, figsize=(10, 3 * n_bands), sharex=True)
        if n_bands == 1:
            axes = [axes]

        # 段颜色
        n_seg = len(CCDC_result)
        seg_colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_seg)))

        # 每个波段一行
        for b in range(n_bands):
            ax = axes[b]
            # 观测散点
            ax.scatter(x, Y[:, b], color='gray', s=10, alpha=0.5, label='Observed', rasterized=True)

            # 逐段绘制
            idx = 0
            for seg in CCDC_result:
                try:
                    t_start = int(np.floor(seg['t_start']))
                    t_end   = int(np.ceil (seg['t_end']))

                    # coeffs: list 长度 n_bands；取第 b 个波段的 1D 系数
                    coeffs_per_band = seg['coeffs'][:, b]

                    # 逐日预测
                    t_range = np.arange(t_start, t_end + 1, 1)
                    y_pred = autoTSPred(t_range, coeffs_per_band)
                    ax.plot(t_range, y_pred, color=seg_colors[idx % len(seg_colors)], linewidth=2, label=f"Seg {idx+1}")

                    # 断点（可选）
                    tb = seg.get('t_break', 0)
                    try:
                        tb = float(tb)
                        if tb > 0:
                            ax.axvline(tb, color=seg_colors[idx % len(seg_colors)], linestyle='--', alpha=0.6)
                    except Exception:
                        print(traceback.format_exc())
                        pass
                except Exception:
                    # 单段失败不影响其它段
                    print(traceback.format_exc())
                    continue

            idx += 1
            # 年份竖线/轴
            for xv in yline_positions:
                ax.axvline(xv, color='lightgray', linestyle=':', linewidth=1)
            ax.axvline(0, color='lightgray', linestyle=':', linewidth=1)

            ax.set_ylabel(f"Band {b+1}")
            ax.legend(loc='best', fontsize=9)

            if b == n_bands - 1:
                ax.set_xticks(year_tick_positions)
                ax.set_xticklabels(year_labels)
                ax.set_xlabel("Year")
            else:
                ax.tick_params(labelbottom=False)

        fig.suptitle(title, y=0.995, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(output_figpath, dpi=dpi)
        plt.close(fig)

    except Exception:
        print(traceback.format_exc())
        pass

def ccdc_indi_res(filename, doy, arr_):
    l_ = filename.split('.csv')[0].split('_')

    # Get the xy coordinate
    try:
        x_coord = [int(__[1:]) for __ in l_ if __.startswith('x')][0]
        y_coord = [int(__[1:]) for __ in l_ if __.startswith('y')][0]
    except:
        x_coord = np.nan
        y_coord = np.nan

    def spaced_str_to_array(s):
        # 若是 (idx, '[[...]]') 这种，取第二个元素
        if isinstance(s, tuple) and len(s) == 2:
            s = s[1]

        # 拿到第一行，数出本行有几个数作为列数
        lines = [ln for ln in str(s).splitlines() if ln.strip()]
        n_cols = np.fromstring(lines[0].replace('[', ' ').replace(']', ' '), sep=' ').size

        # 去掉方括号，整体按空格解析为一维，再按列数 reshape
        flat = np.fromstring(str(s).replace('[', ' ').replace(']', ' '), sep=' ')
        return flat.tolist()

    # Get the feature lista
    try:
        df_ = pd.read_csv(filename)
        coeff = df_.loc[(df_['t_start'] < doy) & (df_['t_end'] > doy), 'coeffs'].to_numpy()
        if coeff.shape[0] > 0:
            coeff = coeff[0]
            coeff_list = spaced_str_to_array(coeff)
        else:
            coeff_list = []
    except:
        print(traceback.format_exc())
        coeff_list = []

    # Get the arr
    if arr_ is not None:
        label = arr_[y_coord, x_coord]
    else:
        label = np.nan
    return x_coord, y_coord, coeff_list, label

def ccdc_pro_res(ccdc_csv_folder, doy, output_folder, label_tif=None):

    if not os.path.exists(output_folder):
        bf.create_folder(output_folder)

    if label_tif is None:
        arr_ = None
    else:
        try:
            ds_ = gdal.Open(label_tif)
            arr_ = ds_.GetRasterBand(1).ReadAsArray()
        except:
            arr_ = None
            print('No label tif was adopted')

    ccdc_csv_file = bf.file_filter(ccdc_csv_folder, ['CCDC_result'])
    xcoord_list, ycoord_list, label_ls, feature_ls = [], [], [], []

    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as exe:
        results = list(tqdm(exe.map(ccdc_indi_res, ccdc_csv_file, repeat(doy), repeat(arr_), chunksize=100), total=len(ccdc_csv_file)))
    results = list(results)

    for _ in results:
        xcoord_list.append(_[0])
        ycoord_list.append(_[1])
        label_ls.append(_[3])
        feature_ls.append(_[2])

    feature_num = max([len(_) for _ in feature_ls])
    feature_ls_normalised = [x if len(x) == feature_num else [np.nan] * feature_num for x in feature_ls]
    df = pd.DataFrame({'x_cord': xcoord_list,  'y_cord': ycoord_list, 'label': label_ls})
    value_df = pd.DataFrame(feature_ls_normalised, columns=[f'feature_{i + 1}' for i in range(feature_num)])
    df = pd.concat([df, value_df], axis=1)
    df.to_csv(f'{output_folder}\\ccdc_list_yr{str(np.floor(doy/365.25)+1986)}.csv')


def train_xgbrf_gpu_single(
    csv_path=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res\ccdc_list1.csv",
    output_dir=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res\xgbrf_gpu_out",
    feature_cols = None,
    label_col: str = "label",
    x_coord_col: str = "x_cord",
    n_bins: int = 50,
    train_ratio_per_bin: float = 0.8,
    seed: int = 31,
    grid_max_depth=[6, ],
    grid_subsample=[0.7, ],
    grid_gamma= [4,],
    grid_alpha= [ 1],
    grid_lambda = [ 2], # RF-style
    n_estimators: int = 8000,
    learning_rate: float = 0.5,
    early_stopping_rounds: int = 100
):
    rng = np.random.RandomState(seed)
    if feature_cols is None:
        feature_cols = [f"feature_{i}" for i in range(1, 80)]

    if not os.path.exists(csv_path) and os.path.exists(csv_path + ".csv"):
        csv_path = csv_path + ".csv"
    df = pd.read_csv(csv_path)

    needed = set(feature_cols + [label_col, x_coord_col])
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺失列: {miss}")

    # 映射标签到 0..6
    uniq_labels = sorted(df[label_col].dropna().unique().tolist())
    if uniq_labels != [1,2,3,4,5,6,7]:
        print(f"[WARN] 唯一标签集合为 {uniq_labels}，将按实际类别训练。")
    lab2id = {lab:i for i, lab in enumerate(uniq_labels)}
    id2lab = {i:lab for lab,i in lab2id.items()}
    num_class = len(uniq_labels)
    df["_y"] = df[label_col].map(lab2id).astype(int)

    # 50 个等量分箱；不够则退化为等宽
    try:
        df["_bin"] = pd.qcut(df[x_coord_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["_bin"] = pd.cut(df[x_coord_col], bins=n_bins, labels=False, include_lowest=True)
    df["_bin"] = df["_bin"].astype(int)

    train_idx, valid_idx = [], []
    for b in sorted(df["_bin"].unique()):
        idx = np.where(df["_bin"].values == b)[0]
        rng.shuffle(idx)
        k = int(len(idx) * train_ratio_per_bin)
        if len(idx) > 1:
            k = max(1, min(len(idx)-1, k))
        train_idx.extend(idx[:k])
        valid_idx.extend(idx[k:])

    X = df[feature_cols].values
    y = df["_y"].values
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for md, sub, gm, al, lm in product(grid_max_depth, grid_subsample, grid_gamma, grid_alpha, grid_lambda):

        tag = f"md{md}_sub{str(sub).replace('.','p')}_gm{gm}_al{str(al).replace('.','p')}_lm{str(lm).replace('.','p')}"
        save_dir = os.path.join(output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)

        clf = XGBClassifier(
            objective="multi:softprob",
            device = "cuda",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=md,
            subsample=sub,
            colsample_bytree=0.7,
            colsample_bynode=0.7,
            colsample_bylevel=0.7, # RF 风格列抽样（节点级）
            sampling_method="uniform",
            gamma=gm,
            reg_alpha=al,
            reg_lambda=lm,
            min_child_weight=5,
            random_state=seed,
            eval_metric=[ "mlogloss", "merror", 'auc'],
            verbosity=1,
            early_stopping_rounds = 10,
        )

        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        clf.fit(
            X_train, y_train,
            eval_set=eval_set,
        )

        booster = clf.get_booster()
        booster.set_attr(class_labels=json.dumps(['bareland', 'low', 'high', 'forest', 'farm', 'water', 'impre'], ensure_ascii=False))

        # 评估
        y_pred = clf.predict(X_valid)
        rep = classification_report(
            y_valid, y_pred,
            target_names=[str(id2lab[i]) for i in range(num_class)],
            digits=4, output_dict=True
        )
        cm = confusion_matrix(y_valid, y_pred, labels=list(range(num_class)))

        y_pred_train = clf.predict(X_train)
        rep_train = classification_report(
            y_train,  y_pred_train ,
            target_names=[str(id2lab[i]) for i in range(num_class)],
            digits=4, output_dict=True
        )
        cm2 = confusion_matrix( y_train,  y_pred_train , labels=list(range(num_class)))

        # 1) report -> DataFrame（转置成常见行式）
        df_rep = pd.DataFrame(rep).T  # 列一般为 ['precision','recall','f1-score','support']
        df_rep_train = pd.DataFrame(rep_train).T
        # 2) 取特征名
        if hasattr(X_valid, "values"):
            n_feat = X_valid.shape[1]
            feat_names = list(X_valid.columns)
        else:
            n_feat = X_valid.shape[1]
            feat_names = [f"{i}" for i in range(1, n_feat + 1)]

        # 3) 用 XGBoost 的 gain 重要性
        booster = clf.get_booster()
        gain_dict = booster.get_score(importance_type="gain")  # {"f0": val, "f1": val, ...}

        # 对齐到特征顺序（f0 对应第 1 列，以此类推；缺失的补 0）
        gain_vals = [float(gain_dict.get(f"f{j}", 0.0)) for j in range(n_feat)]

        # 做成“接在下面”的块：先分隔行，再是特征重要性表
        sep_row = pd.DataFrame(
            [[np.nan] * len(df_rep.columns)],
            index=["feature_importance_gain"],
            columns=df_rep.columns
        )
        df_gain_block = (
            pd.DataFrame({"gain_importance": gain_vals}, index=feat_names)
            .sort_values("gain_importance", ascending=False)
        )

        # 4) 竖向拼接并一次性保存
        df_out = pd.concat([df_rep, sep_row, df_gain_block], axis=0)
        report_path = os.path.join(save_dir, "classification_report_valid.csv")
        df_out.to_csv(report_path, encoding="utf-8-sig")
        df_train_path = os.path.join(save_dir, "classification_report_train.csv")
        df_rep_train.to_csv(df_train_path, encoding="utf-8-sig")

        # 混淆矩阵照旧单独存
        pd.DataFrame(
            cm,
            index=[f"true_{id2lab[i]}" for i in range(num_class)],
            columns=[f"pred_{id2lab[i]}" for i in range(num_class)]
        ).to_csv(os.path.join(save_dir, "confusion_matrix_valid.csv"), encoding="utf-8-sig")

        # 学习曲线
        ev = clf.evals_result()
        tr_mlog = ev["validation_0"]["mlogloss"]; va_mlog = ev["validation_1"]["mlogloss"]
        tr_merr = ev["validation_0"]["merror"];   va_merr = ev["validation_1"]["merror"]

        plt.figure(); plt.plot(tr_mlog,label="train-mlogloss"); plt.plot(va_mlog,label="valid-mlogloss")
        if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
            plt.axvline(clf.best_iteration, ls="--", label=f"best_iter={clf.best_iteration}")
        plt.title(f"Learning Curve (mlogloss) - {tag}")
        plt.xlabel("Boosting Rounds"); plt.ylabel("mlogloss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curve_mlogloss.png"), dpi=150); plt.close()

        plt.figure(); plt.plot(tr_merr,label="train-merror"); plt.plot(va_merr,label="valid-merror")
        if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
            plt.axvline(clf.best_iteration, ls="--", label=f"best_iter={clf.best_iteration}")
        plt.title(f"Learning Curve (merror) - {tag}")
        plt.xlabel("Boosting Rounds"); plt.ylabel("merror"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curve_merror.png"), dpi=150); plt.close()

        # 保存模型 + 结果
        clf.save_model(os.path.join(save_dir, "model.json"))
        summary.append({
            "combo": tag,
            "max_depth": md, "subsample": sub, "gamma": gm, "alpha":al,"lambda":lm,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": getattr(clf, "best_iteration", None),
            "best_score_valid": getattr(clf, "best_score", None),
            "train_accuracy": rep_train["accuracy"],
            "train_macro_f1": rep_train["macro avg"]["f1-score"],
            "train_weighted_f1": rep_train["weighted avg"]["f1-score"],
            "valid_accuracy": rep["accuracy"],
            "valid_macro_f1": rep["macro avg"]["f1-score"],
            "valid_weighted_f1": rep["weighted avg"]["f1-score"],
            "model_path": os.path.join(save_dir, "model.json")
        })

        with open(os.path.join(save_dir, "evals_result.json"), "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)

        print(f"[DONE] {tag} | acc={rep['accuracy']:.4f} | macro_f1={rep['macro avg']['f1-score']:.4f} | "
              f"weighted_f1={rep['weighted avg']['f1-score']:.4f}")

    if summary:
        df_sum = pd.DataFrame(summary).sort_values(
            by=["valid_weighted_f1","valid_macro_f1","valid_accuracy"], ascending=False
        )
        out_csv = os.path.join(output_dir, "grid_summary.csv")
        df_sum.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[SUMMARY] {out_csv}")
        print("[BEST]", df_sum.iloc[0].to_dict())



def predict_xgbrf_gpu_single(
        csv_path,
        model_path,
        outpath,
        feature_cols=None,
        x_coord_col: str = "x_cord",
        yr=0,
):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    if feature_cols is None:
        feature_cols = [f"feature_{i}" for i in range(1, 80)]

    # 检查缺失列
    needed = set(feature_cols + [x_coord_col])
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺失列: {miss}")

    X = df[feature_cols].values

    # 加载模型
    booster = xgb.Booster()
    booster.load_model(model_path)

    # 预测
    dmatrix = xgb.DMatrix(X)
    y_pred_prob = booster.predict(dmatrix)
    y_pred_class = np.argmax(y_pred_prob, axis=1)

    # ===== 新增：读取模型中保存的标签名 =====
    class_labels = None
    if booster.attr("class_labels") is not None:
        class_labels = json.loads(booster.attr("class_labels"))

    df["predicted_class"] = y_pred_class
    if class_labels is not None:
        df["predicted_label"] = [class_labels[int(i)] for i in y_pred_class]
    else:
        df["predicted_label"] = df["predicted_class"]

    # 保存结果
    os.makedirs(outpath, exist_ok=True)
    result_path = os.path.join(outpath, f"predicted_results_yr{str(yr)}.csv")
    df.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 预测完成 → {result_path}")



# 示例（按需取消注释）：
if __name__ == "__main__":
    # _ = 13687
    # ccdc_pro_res('G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\CCDC\output\pixel_csv\\', _, output_folder='G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_new\\', label_tif='G:\A_GEDI_Floodplain_vegh\Veg_map\\veg_mapV2_30m.tif')
    # _ = 13687 - 730
    # for __ in range(3):
    #     ccdc_pro_res('G:\A_Landsat_Floodplain_veg\Landsat_floodplain_2020_datacube\CCDC\output\pixel_csv\\', _, output_folder='G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_csv\\')
    #     _ += 365
    # train_xgbrf_gpu_single(csv_path=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_new\ccdc_list_new30m.csv", output_dir=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_new\xgbrf_gpu_out")
    for _ in range(1986, 2024):
        predict_xgbrf_gpu_single(f'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_csv\\ccdc_list_yr{str(_)}.0.csv', 'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_new\\xgbrf_gpu_out\md8_sub0p5_gm4_al1_lm2\\model.json', 'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_pre_new', yr=_)
    #
    # # train_xgbrf_gpu_single(csv_path=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_old\ccdc_list_old30m.csv", output_dir=r"G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_old\xgbrf_gpu_out")
    # for _ in range(1986, 2024):
    #     predict_xgbrf_gpu_single(f'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_csv\\ccdc_list_yr{str(_)}.0.csv', 'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_res_old\\xgbrf_gpu_out\md8_sub0p5_gm4_al1_lm2\\model.json', 'G:\A_GEDI_Floodplain_vegh\Veg_map\CCDC_pre_old',yr=_)

#     doy_arr = [320,336,352,512,704,720,768,800,832,848,1040,1136,1536,1568,1840,2144,2160,2176,2208,2512,2528,2624,2688,2720,2944,3232,3344,3600,4160,4320,4736,4800,4880,5088,5096,5104,5200,5264,5688,5808,5848,5864,5880,5912,5944,6104,6152,6240,6288,6296,6312,6592,6616,6640,6648,6672,6880,6896,6912,7016,7032,7272,7288,7376,7384,7400,7536,7544,7616,7816,8072,8080,8096,8120,8176,8368,8384,8400,8424,8432,8496,8504,8696,8728,8816,8840,9088,9104,9120,9136,9248,9288,9408,9424,9432,9480,9560,9800,10144,10160,10208,10224,10240,10248,10320,10352,10368,10536,10544,10568,10576,10872,10896,10992,11216,11232,11272,11296,11320,11344,11408,11424,11440,11632,11680,11696,11744,11824,11848,11944,11992,12016,12120,12160,12384,12392,12416,12448,12456,12464,12496,12520,12576,12784,12792,12816,12832,12856,12864,12904,12912,13080,13120,13136,13144,13216,13240,13248,13272,13360,13368,13520,13544,13576,13628,13632,13650,13664,13728,13760,13770,13797,13808,13816,13824,13840,13856,13872]
#     trend_arr = [[32987.], [33094.], [33105.], [33492.], [33038.], [32995.], [33250.], [33225.], [33193.], [33364.], [32733.], [33312.], [33297.], [33458.], [33334.], [33574.], [32979.], [32982.], [33104.], [33270.], [32805.], [33218.], [33706.], [33905.], [32962.], [33108.], [33306.], [32948.], [33566.], [33100.], [33231.], [33607.], [33857.], [33382.], [33188.], [33372.], [33389.], [33502.], [33260.], [33156.], [33185.], [33059.], [33000.], [33136.], [33426.], [33153.], [33019.], [32776.], [33229.], [33308.], [33356.], [33146.], [32866.], [33456.], [33232.], [33587.], [33283.], [33320.], [33079.], [33379.], [33286.], [33224.], [32945.], [33311.], [33103.], [33504.], [34076.], [33333.], [33251.], [34055.], [32795.], [33245.], [33257.], [33514.], [33829.], [33324.], [32849.], [33475.], [33227.], [33079.], [33870.], [33848.], [33646.], [33360.], [34019.], [33944.], [33562.], [33754.], [33789.], [33262.], [34432.], [34888.], [34249.], [34251.], [33622.], [33824.], [34151.], [34037.], [34711.], [34604.], [34962.], [34788.], [34555.], [34428.], [35407.], [36021.], [36340.], [35207.], [35273.], [34657.], [34820.], [34528.], [35744.], [34739.], [34969.], [34520.], [34469.], [35456.], [33766.], [34860.], [36151.], [36538.], [36364.], [34924.], [34621.], [34849.], [34853.], [35284.], [35494.], [35069.], [35708.], [35134.], [34455.], [36033.], [35324.], [34713.], [34509.], [35218.], [34727.], [35595.], [35380.], [36213.], [35310.], [34156.], [33948.], [34312.], [34657.], [35076.], [35018.], [35634.], [35017.], [34422.], [34059.], [34147.], [34417.], [34905.], [35228.], [34956.], [34657.], [35229.], [34605.], [34469.], [34357.], [34599.], [35854.], [35047.], [35781.], [35986.], [34528.], [34738.], [34715.], [34821.], [34387.], [34770.], [35460.], [35046.], [34982.], [34923.]]
#     trend_arr = (np.array(trend_arr) - 32768) / 10000
#     ccdc_res = TrendSeasonalFit_v12_30Line(np.array(doy_arr), np.array(trend_arr))