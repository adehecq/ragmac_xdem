"""All tools and functions needed for the temporal fit of DEM stacks"""

import multiprocessing as mp

import matplotlib
import numpy as np
import pandas as pd
import psutil
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    PairwiseKernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm import tqdm

import xarray as xr
import dask

from ragmac_xdem import utils

###########
"""
Modified from https://github.com/dshean/pygeotools/blob/master/pygeotools/lib/malib.py#L999

@author: dshean
"""


def calcperc(b, perc=(2.0, 98.0)):
    """Calculate values at specified percentiles"""
    b = checkma(b)
    if b.count() > 0:
        low = np.percentile(b.compressed(), perc[0])
        high = np.percentile(b.compressed(), perc[1])
    else:
        low = 0
        high = 0
    return low, high


def calcperc_sym(b, perc=(2.0, 98.0)):
    """
    Get symmetrical percentile values
    Useful for determining clim centered on 0 for difference maps
    """
    clim = np.max(np.abs(calcperc(b, perc)))
    return -clim, clim


def checkma(a, fix=False):
    # isinstance(a, np.ma.MaskedArray)
    if np.ma.is_masked(a):
        out = a
    else:
        out = np.ma.array(a)
    # Fix invalid values
    # Note: this is not necessarily desirable for writing
    if fix:
        # Note: this fails for datetime arrays! Treated as objects.
        # Note: datetime ma returns '?' for fill value
        from datetime import datetime

        if isinstance(a[0], datetime):
            print("Input array appears to be datetime.  Skipping fix")
        else:
            out = np.ma.fix_invalid(out, copy=False)
    return out


def fast_median(a):
    """Fast median operation for masked array using 50th-percentile"""
    a = checkma(a)
    # return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out


def mad(a, axis=None, c=1.4826, return_med=False):
    """Compute normalized median absolute difference

    Can also return median array, as this can be expensive, and often we want both med and nmad
    Note: 1.4826 = 1/0.6745
    """
    a = checkma(a)
    # return np.ma.median(np.fabs(a - np.ma.median(a))) / c
    if a.count() > 0:
        if axis is None:
            med = fast_median(a)
            out = fast_median(np.fabs(a - med)) * c
        else:
            med = np.ma.median(a, axis=axis)
            # The expand_dims is necessary for broadcasting
            out = np.ma.median(np.ma.fabs(a - np.expand_dims(med, axis=axis)), axis=axis) * c
    else:
        out = np.ma.masked
    if return_med:
        out = (out, med)
    return out


def do_robust_linreg(arg):
    date_list_o, y, model = arg
    y_idx = ~(np.ma.getmaskarray(y))
    # newaxis is necessary b/c ransac expects 2D input array
    x = date_list_o[y_idx].data[:, np.newaxis]
    y = y[y_idx].data
    return robust_linreg(x, y, model)


def robust_linreg(x, y, model="theilsen"):
    if model == "linear":
        m = linear_model.LinearRegression()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    elif model == "ransac":
        m = linear_model.RANSACRegressor()
        m.fit(x, y)
        slope = m.estimator_.coef_
        intercept = m.estimator_.intercept_
    elif model == "theilsen":
        m = linear_model.TheilSenRegressor()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    return (slope[0], intercept)


def ma_linreg(
    ma_stack,
    dt_list,
    n_thresh=2,
    model="linear",
    dt_stack_ptp=None,
    min_dt_ptp=None,
    smooth=False,
    rsq=False,
    conf_test=False,
    parallel=True,
    n_cpu=None,
    remove_outliers=False,
):
    """Compute per-pixel linear regression for stack object"""
    # Need to check type of input dt_list
    # For now assume it is Python datetime objects
    date_list_o = np.ma.array(matplotlib.dates.date2num(dt_list))
    date_list_o.set_fill_value(0.0)

    # Only compute trend where we have n_thresh unmasked values in time
    # Create valid pixel count
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16).data
    print("Excluding pixels with count < %i" % n_thresh)
    valid_mask_2D = count >= n_thresh

    # Only compute trend where the time spread (ptp is max - min) is large
    if dt_stack_ptp is not None:
        if min_dt_ptp is None:
            # Calculate from datestack ptp
            max_dt_ptp = calcperc(dt_stack_ptp, (4, 96))[1]
            # Calculate from list of available dates
            min_dt_ptp = 0.10 * max_dt_ptp
        print("Excluding pixels with dt range < %0.2f days" % min_dt_ptp)
        valid_mask_2D = valid_mask_2D & (dt_stack_ptp >= min_dt_ptp).filled(False)

    # Extract 1D time series for all valid pixel locations
    y_orig = ma_stack[:, valid_mask_2D]
    # Extract mask and invert: True where data is available
    valid_mask = ~(np.ma.getmaskarray(y_orig))
    valid_sample_count = np.inf

    if y_orig.count() == 0:
        print("No valid samples remain after count and min_dt_ptp filters")
        slope = None
        intercept = None
        detrended_std = None
    else:
        # Create empty (masked) output grids with original dimensions
        slope = np.ma.masked_all_like(ma_stack[0])
        intercept = np.ma.masked_all_like(ma_stack[0])
        detrended_std = np.ma.masked_all_like(ma_stack[0])

        # While loop here is to iteratively remove outliers, if desired
        # Maximum number of iterations
        max_n = 3
        n = 1
        while y_orig.count() < valid_sample_count and n <= max_n:
            print(f"Iteration #{n}")
            valid_pixel_count = np.sum(valid_mask_2D)
            valid_sample_count = y_orig.count()
            print(
                f"{valid_pixel_count} valid pixels with up to {ma_stack.shape[0]} timestamps: {valid_sample_count} total valid samples"
            )
            if model in ["theilsen", "ransac"]:
                # Create empty arrays for slope and intercept results
                m = np.ma.masked_all(y_orig.shape[1])
                b = np.ma.masked_all(y_orig.shape[1])
                if parallel:
                    if n_cpu is None:
                        n_cpu = mp.cpu_count() - 1
                    n_cpu = int(n_cpu)
                    print("Running in parallel with %i processes" % n_cpu)
                    pbar_kwargs = {"total": y_orig.shape[1], "desc": "Fitting temporal trend", "smoothing": 0}
                    with mp.Pool(processes=n_cpu) as pool:
                        results = list(
                            tqdm(
                                pool.map(
                                    do_robust_linreg,
                                    [(date_list_o, y_orig[:, n], model) for n in range(y_orig.shape[1])],
                                    chunksize=1,
                                ),
                                **pbar_kwargs,
                            )
                        )
                    results = np.array(results)
                    m = results[:, 0]
                    b = results[:, 1]
                else:
                    for n in range(y_orig.shape[1]):
                        # print('%i of %i px' % (n, y_orig.shape[1]))
                        y = y_orig[:, n]
                        m[n], b[n] = do_robust_linreg([date_list_o, y, model])
            else:
                # if model == 'linear':
                # Remove masks, fills with fill_value
                y = y_orig.data
                # Independent variable is time ordinal
                x = date_list_o
                x_mean = x.mean()
                x = x.data
                # Prepare matrices
                X = np.c_[x, np.ones_like(x)]
                a = np.swapaxes(np.dot(X.T, (X[None, :, :] * valid_mask.T[:, :, None])), 0, 1)
                b = np.dot(X.T, (valid_mask * y))

                # Solve for slope/intercept
                print("Solving for trend")
                r = np.linalg.solve(a, b.T)
                # Reshape to original dimensions
                m = r[:, 0]
                b = r[:, 1]

            print("Computing residuals")
            # Compute model fit values for each valid timestamp
            y_fit = m * np.ma.array(date_list_o.data[:, None] * valid_mask, mask=y_orig.mask) + b
            # Compute residuals
            resid = y_orig - y_fit
            # Compute detrended std
            # resid_std = resid.std(axis=0)
            resid_std = mad(resid, axis=0)

            if remove_outliers and n < max_n:
                print("Removing residual outliers > 3-sigma")
                outlier_sigma = 3.0
                # Mask any residuals outside valid range
                valid_mask = valid_mask & (np.abs(resid) < (resid_std * outlier_sigma)).filled(False)
                # Extract new valid samples
                y_orig = np.ma.array(y_orig, mask=~valid_mask)
                # Update valid mask
                valid_count = y_orig.count(axis=0) >= n_thresh
                y_orig = y_orig[:, valid_count]
                valid_mask_2D[valid_mask_2D] = valid_count
                # Extract 1D time series for all valid pixel locations
                # Extract mask and invert: True where data is available
                valid_mask = ~(np.ma.getmaskarray(y_orig))
                # remove_outliers = False
            else:
                break
            n += 1

        # Fill in the valid indices
        slope[valid_mask_2D] = m
        intercept[valid_mask_2D] = b
        detrended_std[valid_mask_2D] = resid_std

        # Smooth the result
        if smooth:
            raise NotImplementedError()
            # size = 5
            # print("Smoothing output with %i px gaussian filter" % size)
            # from pygeotools.lib import filtlib
            # #Gaussian filter
            # #slope = filtlib.gauss_fltr_astropy(slope, size=size)
            # #intercept = filtlib.gauss_fltr_astropy(intercept, size=size)
            # #Median filter
            # slope = filtlib.rolling_fltr(slope, size=size, circular=False)
            # intercept = filtlib.rolling_fltr(intercept, size=size, circular=False)

        if rsq:
            rsquared = np.ma.masked_all_like(ma_stack[0])
            SStot = np.sum((y_orig - y_orig.mean(axis=0)) ** 2, axis=0).data
            SSres = np.sum(resid ** 2, axis=0).data
            r2 = 1 - (SSres / SStot)
            rsquared[valid_mask_2D] = r2

        if conf_test:
            count = y_orig.count(axis=0)
            SE = np.sqrt(SSres / (count - 2) / np.sum((x - x_mean) ** 2, axis=0))
            T0 = r[:, 0] / SE
            alpha = 0.05
            ta = np.zeros_like(r2)
            from scipy.stats import t

            for c in np.unique(count):
                t1 = abs(t.ppf(alpha / 2.0, c - 2))
                ta[(count == c)] = t1
            sig = np.logical_and((T0 > -ta), (T0 < ta))
            sigmask = np.zeros_like(valid_mask_2D, dtype=bool)
            sigmask[valid_mask_2D] = ~sig
            # SSerr = SStot - SSres
            # F0 = SSres/(SSerr/(count - 2))
            # from scipy.stats import f
            #    f.cdf(sig, 1, c-2)
            slope = np.ma.array(slope, mask=~sigmask)
            intercept = np.ma.array(intercept, mask=~sigmask)
            detrended_std = np.ma.array(detrended_std, mask=~sigmask)
            rsquared = np.ma.array(rsquared, mask=~sigmask)

        # slope is in units of m/day since x is ordinal date
        slope *= 365.25

    return slope, intercept, detrended_std


"""
@author: friedrichknuth
"""

def remove_nan_from_training_data(X_train, y_train_masked_array):
    array = y_train_masked_array.data
    mask = ~np.ma.getmaskarray(y_train_masked_array)
    X_train = X_train[mask]
    y_train = y_train_masked_array[mask]
    return X_train, y_train


def mask_low_count_pixels(ma_stack, n_thresh=3):
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16).data
    valid_mask_2D = count >= n_thresh
    valid_data = ma_stack[:, valid_mask_2D]
    return valid_data, valid_mask_2D


def create_prediction_timeseries(start_date="2000-01-01", end_date="2023-01-01", dt="M"):
    # M  = monthly frequency
    # 3M = every 3 months
    # 6M = every 6 months
    d = pd.date_range(start_date, end_date, freq=dt)
    X = d.to_series().apply([utils.date_time_to_decyear]).values.squeeze()
    return X


def linreg_fit(X_train, y_train, method="TheilSen"):

    if method == "Linear":
        m = linear_model.LinearRegression()
        m.fit(X_train.squeeze()[:, np.newaxis], y_train.squeeze())
        slope = m.coef_
        intercept = m.intercept_

    if method == "TheilSen":
        m = linear_model.TheilSenRegressor()
        m.fit(X_train.squeeze()[:, np.newaxis], y_train.squeeze())
        slope = m.coef_
        intercept = m.intercept_

    if method == "RANSAC":
        m = linear_model.RANSACRegressor()
        m.fit(X_train.squeeze()[:, np.newaxis], y_train.squeeze())
        slope = m.estimator_.coef_
        intercept = m.estimator_.intercept_

    return slope[0], intercept


def linreg_run(args):
    X_train, y_train_masked_array, method = args

    X_train, y_train = remove_nan_from_training_data(X_train, y_train_masked_array)
    slope, intercept = linreg_fit(X_train, y_train, method=method)

    return slope, intercept


def linreg_predict(args):
    slope, x, intercept = args
    prediction = slope * x + intercept
    return prediction


def linreg_predict_parallel(slope, X, intercept, cpu_count=None):
    if not cpu_count:
        cpu_count = mp.cpu_count() - 1
    pool = mp.Pool(processes=cpu_count)
    results = pool.map(linreg_predict, tqdm([(slope, x, intercept) for x in X]))
    return np.ma.array(results)


def linreg_reshape_parallel_results(results, ma_stack, valid_mask_2D):
    results_stack = []
    for i in range(results.shape[1]):
        m = np.ma.masked_all_like(ma_stack[0])
        m[valid_mask_2D] = results[:, i]
        results_stack.append(m)
    results_stack = np.ma.stack(results_stack)
    return results_stack


def linreg_run_parallel(X_train, ma_stack, cpu_count=None, method="TheilSen"):
    if not cpu_count:
        cpu_count = mp.cpu_count() - 1
    pool = mp.Pool(processes=cpu_count)
    results = pool.map(linreg_run, tqdm([(X_train, ma_stack[:, i], method) for i in range(ma_stack.shape[1])]))
    return np.array(results)


def GPR_glacier_kernel():
    """
    adapted from
    https://github.com/iamdonovan/pyddem/blob/master/pyddem/fit_tools.py#L1054
    """
    #     k1   = PairwiseKernel(1, metric='linear')
    #     k2 = ConstantKernel(30) * ExpSineSquared(length_scale=1, periodicity=1)
    #     kernel = (
    #         k1+k2
    #     )
    ##these values should be pre-computed based on data distribution
    base_var = 1
    nonlin_var = 1
    period_nonlinear = 1

    k3 = (
        ConstantKernel(base_var * 0.6) * RBF(0.75)
        + ConstantKernel(base_var * 0.3) * RBF(1.5)
        + ConstantKernel(base_var * 0.1) * RBF(3)
    )

    k4 = PairwiseKernel(1, metric="linear") * ConstantKernel(nonlin_var) * RationalQuadratic(period_nonlinear, 1)

    kernel = k3 + k4

    return kernel


def GPR_snow_kernel():
    """
    adapted from
    https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-co2-py
    """

    linear_kernel = PairwiseKernel(1, metric="linear")

    v = 10.0
    long_term_trend_kernel = v ** 2 * RBF(length_scale=v)

    seasonal_kernel = (
        2.0 ** 2
        * RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )

    irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

    noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5))

    kernel = linear_kernel + long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    return kernel


def GPR_model(X_train, y_train, kernel, alpha=1e-10):
    X_train = X_train.squeeze()[:, np.newaxis]
    y_train = y_train.squeeze()

    gaussian_process_model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=alpha, n_restarts_optimizer=9
    )

    gaussian_process_model = gaussian_process_model.fit(X_train, y_train)
    return gaussian_process_model


def GPR_predict(gaussian_process_model, X):
    X = X.squeeze()[:, np.newaxis]
    mean_prediction, std_prediction = gaussian_process_model.predict(X, return_std=True)

    return mean_prediction, std_prediction


def GPR_run(args):
    X_train, y_train_masked_array, X, glacier_kernel = args
    X_train, y_train = remove_nan_from_training_data(X_train, y_train_masked_array)
    gaussian_process_model = GPR_model(X_train, y_train, glacier_kernel, alpha=1e-10)
    prediction, std_prediction = GPR_predict(gaussian_process_model, X)

    return prediction


def GPR_run_parallel(X_train, ma_stack, X, kernel, cpu_count=None):
    if not cpu_count:
        cpu_count = mp.cpu_count() - 1
    pool = mp.Pool(processes=cpu_count)
    results = pool.map(GPR_run, tqdm([(X_train, ma_stack[:, i], X, kernel) for i in range(ma_stack.shape[1])]))
    return np.array(results)


def GPR_reshape_parallel_results(results, ma_stack, valid_mask_2D):
    results_stack = []
    for i in range(results.shape[1]):
        m = np.ma.masked_all_like(ma_stack[0])
        m[valid_mask_2D] = results[:, i]
        results_stack.append(m)
    results_stack = np.ma.stack(results_stack)
    return results_stack

def dask_linreg(DataArray, times = None, count_thresh = None, time_delta_min = None):
    """
    Apply linear regression to DataArray.
    Returns np.nan if valid pixel count less than count_thresh
    and/or difference between first and last time stamp less than time_delta_min.
    
    Default value for time_delta_min assumes times are provided in days.
    """
    mask = ~np.isnan(DataArray)
    
    if count_thresh:
        if np.sum(mask) < count_thresh:
                return np.nan, np.nan
    
    if time_delta_min:
        time_delta = max(times[mask]) - min(times[mask])
        if time_delta < time_delta_min:
            return np.nan, np.nan

#     m = linear_model.LinearRegression()
    m = linear_model.TheilSenRegressor()
    m.fit(times[mask].reshape(-1,1), DataArray[mask])
    
    return m.coef_[0], m.intercept_

def dask_apply_linreg(DataArray, dim, kwargs=None):
    # TODO check if da.map_blocks is faster / more memory efficient
    # using da.apply_ufunc for now.
    # da.map_blocks can handle chunked time dim blocks. 
    results = xr.apply_ufunc(
        dask_linreg,
        DataArray,
        kwargs=kwargs,
        input_core_dims=[[dim]],
        output_core_dims=[[],[]],
        vectorize=True,
        dask="parallelized",)
    return results

def nmad(DataArray):
    if np.all(np.isnan(DataArray)):
        return np.nan
    else:
        return 1.4826 * np.nanmedian(np.abs(DataArray - np.nanmedian(DataArray)))
    
def count(DataArray):
    return np.nansum(~np.isnan(DataArray))

def apply_nmad(DataArray):
    return np.apply_along_axis(nmad,0,DataArray)

def apply_count(DataArray):
    return np.apply_along_axis(count,0,DataArray)

def dask_apply_func(DataArray, func):
    result = xr.apply_ufunc(
        func,
        DataArray,
        dask="allowed",)
    return result

def xr_dask_count(ds):
    """
    Computes count along time axis in x, y, time in dask.array.core.Array.
    
    Returns xr.DataArray with x, y dims.
    """
    arr_count = dask_apply_func(ds['band1'].data, apply_count).compute()
    arr_count = np.ma.masked_where(arr_count==0,arr_count)
    
    count_da = ds['band1'].isel(time=0).drop('time')
    count_da.values = arr_count
    count_da.name = 'count'
    
    return count_da
    
def xr_dask_nmad(ds):
    """
    Computes NMAD along time axis in x, y, time in dask.array.core.Array.
    
    Returns xr.DataArray with x, y dims.
    """
    arr_nmad = dask_apply_func(ds['band1'].data, apply_nmad).compute()
    nmad_da = ds['band1'].isel(time=0).drop('time')
    nmad_da.values = arr_nmad
    nmad_da.name   = 'nmad'
    
    return nmad_da
    
    