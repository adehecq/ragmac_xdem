import os
import glob
import xdem
import geoutils as gu
import numpy as np
import datetime
import matplotlib
import psutil

###########
"""
From ragmac_xdem/data/raw/convert_dates.py

@author: brunbarf
"""

def convert_date_time_to_decimal_date(date_time):
        """
    This function converts a date and a time to a decimal date value
    Inputs:
    - date_time: datetime object
    
    Outputs:
    - decimal_date_float: float

    """
        hourdec=(date_time.hour + date_time.minute/60. + date_time.second/3600.)/24.
        doy = date_time.timetuple().tm_yday
        decimal_date = date_time.year + (doy+hourdec)/365.25
        decimal_date = float('{:.8f}'.format(decimal_date))
        return decimal_date
    
def convert_decimal_date_to_date_time(decimal_date):
        """
    This function converts a decimal date and a date and time
    Inputs:
    - decimal_date: float
    
    Outputs:
    - date_time: datetime object
    - date_time_string: formated string from the datetime object
    """
        decimal_date = float('{:.8f}'.format(decimal_date))
        year = np.floor(decimal_date)
        decimal_day = (decimal_date - np.floor(decimal_date))*365.25
        doy = np.floor(decimal_day)
        decimal_time = (decimal_day - doy)*24.
        hour = np.floor(decimal_time)
        minute = np.floor((decimal_time - hour)*60.)
        second = (decimal_time - hour - minute/60.)*3660.
        raw_str = str(int(year))+'{0:03d}'.format(int(doy))+'{0:02d}'.format(int(hour))+'{0:02d}'.format(int(minute))+'{0:02d}'.format(int(second))
        date_time = datetime.datetime.strptime(raw_str, '%Y%j%H%M%S')
        date_time_string=date_time.strftime('%Y-%m-%d %H:%M:%S')
        return date_time, date_time_string
    

    

###########  
"""
Modified from https://github.com/dshean/pygeotools/blob/master/pygeotools/lib/malib.py#L999

@author: dshean
"""
def checkma(a, fix=False):
    #isinstance(a, np.ma.MaskedArray)
    if np.ma.is_masked(a):
        out=a
    else:
        out=np.ma.array(a)
    #Fix invalid values
    #Note: this is not necessarily desirable for writing
    if fix:
        #Note: this fails for datetime arrays! Treated as objects.
        #Note: datetime ma returns '?' for fill value
        from datetime import datetime
        if isinstance(a[0], datetime):
            print("Input array appears to be datetime.  Skipping fix")
        else:
            out=np.ma.fix_invalid(out, copy=False)
    return out

def mad(a, axis=None, c=1.4826, return_med=False):
    """Compute normalized median absolute difference
   
    Can also return median array, as this can be expensive, and often we want both med and nmad
    Note: 1.4826 = 1/0.6745
    """
    a = checkma(a)
    #return np.ma.median(np.fabs(a - np.ma.median(a))) / c
    if a.count() > 0:
        if axis is None:
            med = fast_median(a)
            out = fast_median(np.fabs(a - med)) * c
        else:
            med = np.ma.median(a, axis=axis)
            #The expand_dims is necessary for broadcasting
            out = np.ma.median(np.ma.fabs(a - np.expand_dims(med, axis=axis)), axis=axis) * c
    else:
        out = np.ma.masked
    if return_med:
        out = (out, med)
    return out

def do_robust_linreg(arg):
    date_list_o, y, model = arg
    y_idx = ~(np.ma.getmaskarray(y))
    #newaxis is necessary b/c ransac expects 2D input array
    x = date_list_o[y_idx].data[:,np.newaxis]
    y = y[y_idx].data
    return robust_linreg(x, y, model)

def robust_linreg(x, y, model='theilsen'):
    from sklearn import linear_model
    slope = None
    intercept = None
    if model == 'linear':
        m = linear_model.LinearRegression()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    elif model == 'ransac':
        m = linear_model.RANSACRegressor()
        m.fit(x, y)
        slope = m.estimator_.coef_
        intercept = m.estimator_.intercept_
        #inlier_mask = ransac.inlier_mask_
        #outlier_mask = np.logical_not(inlier_mask)
    elif model == 'theilsen':
        m = linear_model.TheilSenRegressor()
        m.fit(x, y)
        slope = m.coef_
        intercept = m.intercept_
    #xi = np.arange(x.min(), x.max())[:,np.newaxis]
    #yi = model.predict(xi) 
    #ax.plot(xi, yi)
    return(slope[0], intercept)

def ma_linreg(ma_stack, dt_list, n_thresh=2, model='linear', dt_stack_ptp=None, min_dt_ptp=None, smooth=False, \
        rsq=False, conf_test=False, parallel=True, n_cpu=None, remove_outliers=False):
    """Compute per-pixel linear regression for stack object
    """
    #Need to check type of input dt_list
    #For now assume it is Python datetime objects 
#     from pygeotools.lib import timelib
    date_list_o = np.ma.array(matplotlib.dates.date2num(dt_list))
    date_list_o.set_fill_value(0.0)

    #ma_stack = ma_stack[:,398:399,372:373]
    #dt_stack_ptp = dt_stack_ptp[398:399,372:373]

    #Only compute trend where we have n_thresh unmasked values in time
    #Create valid pixel count
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16).data
    print("Excluding pixels with count < %i" % n_thresh)
    valid_mask_2D = (count >= n_thresh)

    #Only compute trend where the time spread (ptp is max - min) is large
    if dt_stack_ptp is not None:
        if min_dt_ptp is None:
            #Calculate from datestack ptp
            max_dt_ptp = calcperc(dt_stack_ptp, (4, 96))[1]
            #Calculate from list of available dates
            #max_dt_ptp = np.ptp(calcperc(date_list_o, (4, 96)))
            min_dt_ptp = 0.10 * max_dt_ptp
        print("Excluding pixels with dt range < %0.2f days" % min_dt_ptp) 
        valid_mask_2D = valid_mask_2D & (dt_stack_ptp >= min_dt_ptp).filled(False)

    #Extract 1D time series for all valid pixel locations
    y_orig = ma_stack[:, valid_mask_2D]
    #Extract mask and invert: True where data is available
    valid_mask = ~(np.ma.getmaskarray(y_orig))
    valid_sample_count = np.inf

    if y_orig.count() == 0:
        print("No valid samples remain after count and min_dt_ptp filters")
        slope = None
        intercept = None
        detrended_std = None
    else:
        #Create empty (masked) output grids with original dimensions
        slope = np.ma.masked_all_like(ma_stack[0])
        intercept = np.ma.masked_all_like(ma_stack[0])
        detrended_std = np.ma.masked_all_like(ma_stack[0])

        #While loop here is to iteratively remove outliers, if desired
        #Maximum number of iterations
        max_n = 3
        n = 1
        while(y_orig.count() < valid_sample_count and n <= max_n):
            print(n)
            valid_pixel_count = np.sum(valid_mask_2D)
            valid_sample_count = y_orig.count()
            print("%i valid pixels with up to %i timestamps: %i total valid samples" % \
                    (valid_pixel_count, ma_stack.shape[0], valid_sample_count))
            if model == 'theilsen' or model == 'ransac':
                #Create empty arrays for slope and intercept results 
                m = np.ma.masked_all(y_orig.shape[1])
                b = np.ma.masked_all(y_orig.shape[1])
                if parallel:
                    import multiprocessing as mp
                    if n_cpu is None:
                        n_cpu = psutil.cpu_count(logical=True)
                    n_cpu = int(n_cpu)
                    print("Running in parallel with %i processes" % n_cpu)
                    pool = mp.Pool(processes=n_cpu)
                    results = pool.map(do_robust_linreg, [(date_list_o, y_orig[:,n], model) for n in range(y_orig.shape[1])])

                    results = np.array(results)
                    m = results[:,0]
                    b = results[:,1]
                else:
                    for n in range(y_orig.shape[1]):
                        print('%i of %i px' % (n, y_orig.shape[1]))
                        y = y_orig[:,n]
                        m[n], b[n] = do_robust_linreg([date_list_o, y, model])
            else:
                #if model == 'linear':
                #Remove masks, fills with fill_value
                y = y_orig.data
                #Independent variable is time ordinal
                x = date_list_o
                x_mean = x.mean()
                x = x.data
                #Prepare matrices
                X = np.c_[x, np.ones_like(x)]
                a = np.swapaxes(np.dot(X.T, (X[None, :, :] * valid_mask.T[:, :, None])), 0, 1)
                b = np.dot(X.T, (valid_mask*y))
            
                #Solve for slope/intercept
                print("Solving for trend")
                r = np.linalg.solve(a, b.T)
                #Reshape to original dimensions
                m = r[:,0]
                b = r[:,1]

            print("Computing residuals")
            #Compute model fit values for each valid timestamp
            y_fit = m*np.ma.array(date_list_o.data[:,None]*valid_mask, mask=y_orig.mask) + b 
            #Compute residuals
            resid = y_orig - y_fit
            #Compute detrended std
            #resid_std = resid.std(axis=0)
            resid_std = mad(resid, axis=0)

            if remove_outliers and n < max_n:
                print("Removing residual outliers > 3-sigma")
                outlier_sigma = 3.0
                #Mask any residuals outside valid range
                valid_mask = valid_mask & (np.abs(resid) < (resid_std * outlier_sigma)).filled(False)
                #Extract new valid samples
                y_orig = np.ma.array(y_orig, mask=~valid_mask)
                #Update valid mask
                valid_count = (y_orig.count(axis=0) >= n_thresh)
                y_orig = y_orig[:, valid_count]
                valid_mask_2D[valid_mask_2D] = valid_count
                #Extract 1D time series for all valid pixel locations
                #Extract mask and invert: True where data is available
                valid_mask = ~(np.ma.getmaskarray(y_orig))
                #remove_outliers = False
            else:
                break
            n += 1

        #Fill in the valid indices
        slope[valid_mask_2D] = m 
        intercept[valid_mask_2D] = b
        detrended_std[valid_mask_2D] = resid_std

        #Smooth the result
        if smooth:
            size = 5
            print("Smoothing output with %i px gaussian filter" % size)
            from pygeotools.lib import filtlib
            #Gaussian filter
            #slope = filtlib.gauss_fltr_astropy(slope, size=size)
            #intercept = filtlib.gauss_fltr_astropy(intercept, size=size)
            #Median filter
            slope = filtlib.rolling_fltr(slope, size=size, circular=False)
            intercept = filtlib.rolling_fltr(intercept, size=size, circular=False)

        if rsq:
            rsquared = np.ma.masked_all_like(ma_stack[0])
            SStot = np.sum((y_orig - y_orig.mean(axis=0))**2, axis=0).data
            SSres = np.sum(resid**2, axis=0).data
            r2 = 1 - (SSres/SStot)
            rsquared[valid_mask_2D] = r2

        if conf_test:
            count = y_orig.count(axis=0)
            SE = np.sqrt(SSres/(count - 2)/np.sum((x - x_mean)**2, axis=0))
            T0 = r[:,0]/SE
            alpha = 0.05
            ta = np.zeros_like(r2)
            from scipy.stats import t
            for c in np.unique(count):
                t1 = abs(t.ppf(alpha/2.0,c-2))
                ta[(count == c)] = t1
            sig = np.logical_and((T0 > -ta), (T0 < ta))
            sigmask = np.zeros_like(valid_mask_2D, dtype=bool)
            sigmask[valid_mask_2D] = ~sig
            #SSerr = SStot - SSres
            #F0 = SSres/(SSerr/(count - 2))
            #from scipy.stats import f
            #    f.cdf(sig, 1, c-2)
            slope = np.ma.array(slope, mask=~sigmask)
            intercept = np.ma.array(intercept, mask=~sigmask)
            detrended_std = np.ma.array(detrended_std, mask=~sigmask)
            rsquared = np.ma.array(rsquared, mask=~sigmask)
        
        #slope is in units of m/day since x is ordinal date
        slope *= 365.25

    return slope, intercept, detrended_std