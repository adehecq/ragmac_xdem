"""
Set of tools to postprocess the DEMs: statistics calculations, coregistration, filtering etc
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
import threading
from datetime import datetime
from glob import glob
from typing import Callable

import cv2
import geoutils as gu
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xdem
from skimage.morphology import disk
from tqdm import tqdm
import matplotlib
import xarray as xr

from ragmac_xdem import io, temporal, utils

# Turn off imshow's interpolation to avoid gaps spread in plots
plt.rcParams["image.interpolation"] = "none"


def calculate_stats(ddem, roi_mask, stable_mask):
    """
    Returns main statistics of ddem:
    fraction of coverage over glacier, number of obs, NMAD and median dh over stable terrain
    Warning: ddems may contain NaNs on top of nodata values.
    """
    # Get array of valid data (no nodata, no nan) and mask
    data, mask = gu.spatial_tools.get_array_and_mask(ddem)

    # Make sure input masks are 2D
    roi_mask = roi_mask.squeeze()
    stable_mask = stable_mask.squeeze()

    # Calculate coverage over glaciers
    nobs = np.sum(~mask[roi_mask])
    ntot = np.sum(roi_mask)
    roi_coverage = nobs / ntot

    # Calculate statistics in stable terrain
    nstable = np.sum(~mask[stable_mask])
    nmad_stable = xdem.spatialstats.nmad(data[stable_mask])
    med_stable = np.nanmedian(data[stable_mask])

    return roi_coverage, nstable, med_stable, nmad_stable


def calculate_init_stats_single(
    dem_path: str,
    ref_dem_path: str,
    roi_outlines: gu.Vector,
    all_outlines: gu.Vector,
):
    """
    Calculate initial statistics of DEM in `dem_path` differenced with ref_dem.

    Reads input DEM, reproject onto ref DEM grid, mask content of outlines, and calculate statistics.

    :param dem_path: Path to the input DEM to be coregistered
    :param ref_dem: the reference DEM
    :param roi_outlines: The outlines of the glacier to study
    :param all_outlines: The outlines of all glaciers in the study area

    :returns: a tuple containing - basename of DEM, path to DEM, count of obs, median and NMAD over stable terrain, coverage over roi
    """
    # Load DEM and reproject to ref grid
    ref_dem = xdem.DEM(ref_dem_path)
    dem = xdem.DEM(dem_path)
    ref_dem = ref_dem.reproject(dem,resampling='bilinear')
    

    # Create masks
    roi_mask = roi_outlines.create_mask(dem)
    stable_mask = ~all_outlines.create_mask(dem)

    # Calculate dDEM
    ddem = dem - ref_dem

    # Filter gross outliers in stable terrain
    inlier_mask = nmad_filter(ddem.data, stable_mask, verbose=False)
    outlier_mask = ~inlier_mask & stable_mask
    ddem.data.mask[outlier_mask] = True
    del inlier_mask

    # Calculate coverage on and off ice
    roi_coverage_orig, nstable_orig, med_orig, nmad_orig = calculate_stats(ddem, roi_mask, stable_mask)

    # Calculate DEM date
    dem_date = utils.get_dems_date(
        [
            dem_path,
        ]
    )

    return (
        os.path.basename(dem_path),
        dem_path,
        dem_date[0].isoformat(),
        nstable_orig,
        med_orig,
        nmad_orig,
        roi_coverage_orig,
    )


def calculate_init_stats_parallel(
    dem_path_list,
    ref_dem,
    roi_outlines,
    all_outlines,
    outfile,
    overwrite: bool = False,
    nthreads: int = 1,
):
    """
    Calculate DEM statistics of all files in dem_path_list by running calculate_init_stats_single in parallel
    """
    # List the filenames of the DEMs to be processed
    if os.path.exists(outfile) & (not overwrite):
        print(f"File {outfile} already exists -> nothing to be done")
        df_stats = pd.read_csv(outfile)
        
        # Change hard coded file paths if generated on another system
        BASE_DIR = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        df_stats['dem_path'] = BASE_DIR+os.sep+df_stats['dem_path'].str.split('ragmac_xdem/').str[-1]
        
        return df_stats

    global _stats_wrapper

    def _stats_wrapper(dem_path):
        """Calculate stats of a DEM in one thread."""
        # this try-except block catches error that arise when the source DEM is at the edge of aoi
        # an example error is plotted in PR#59 
        # https://github.com/adehecq/ragmac_xdem/pull/59
        try:
            outputs = calculate_init_stats_single(dem_path, ref_dem, roi_outlines, all_outlines)
        except ValueError as e:
            outputs = [None]*7
        return outputs

    # Arguments to be used for the progress bar
    pbar_kwargs = {"total": len(dem_path_list), "desc": "Calculate initial stats of DEMs", "smoothing": 0}

    # Run with either 1 or several threads
    if nthreads == 1:
        results = []
        for dem_path in tqdm(dem_path_list, **pbar_kwargs):
            output = _stats_wrapper(dem_path)
            results.append(output)

    elif nthreads > 1:

        # Needed for MacOS with Python >= 3.8
        # See https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
        cx = mp.get_context("fork")
        with cx.Pool(nthreads) as pool:
            results = list(tqdm(pool.imap(_stats_wrapper, dem_path_list, chunksize=1), **pbar_kwargs))
            pool.close()
            pool.join()

    else:
        raise ValueError("nthreads must be >= 1")

    # -- Save stats to file -- #

    # Convert output data to DataFrame
    df_stats = pd.DataFrame(
        results,
        columns=[
            "ID",
            "dem_path",
            "dem_date",
            "nstable_orig",
            "med_orig",
            "nmad_orig",
            "roi_cover_orig",
        ],
    )

    # Save output to file
    df_stats.dropna(inplace=True)
    df_stats.to_csv(outfile, index=False, float_format="%.2f")

    return df_stats


def nmad_filter(
    dh_array: np.ndarray, inlier_mask: np.ndarray, nmad_factor: float = 5, max_iter: int = 20, verbose: bool = False
) -> np.ndarray:
    """
    Iteratively remove pixels where the elevation difference (dh_array) in stable terrain (inlier_mask) is larger \
    than nmad_factor * NMAD.
    Iterations will stop either when the NMAD change is less than 0.1, or after max_iter iterations.

    :params dh_array: 2D array of elevation difference.
    :params inlier_mask: 2D boolean array of areas to include in the analysis (inliers=True).
    :param nmad_factor: The factor by which the stable dh NMAD has to be multiplied to calculate the outlier threshold
    :param max_iter: Maximum number of iterations (normally not reached, just for safety)
    :param verbose: set to True to print some statistics to screen.

    :returns: 2D boolean array with updated inliers set to True
    """
    # Mask unstable terrain
    dh_stable = dh_array.copy()
    dh_stable.mask[~inlier_mask] = True
    nmad_before = xdem.spatialstats.nmad(dh_stable)
    if verbose:
        print(f"NMAD before: {nmad_before:.2f}")
        print("Iteratively remove large outliers")

    # Iteratively remove large outliers
    for i in range(max_iter):
        outlier_threshold = nmad_factor * nmad_before
        dh_stable.mask[np.abs(dh_stable) > outlier_threshold] = True
        nmad_after = xdem.spatialstats.nmad(dh_stable)
        if verbose:
            print(f"Remove pixels where abs(value) > {outlier_threshold:.2f} -> New NMAD: {nmad_after:.2f}")

        # If NMAD change is loweer than a set threshold, stop iterations, otherwise stop after max_iter
        if nmad_before - nmad_after < 0.1:
            break

        nmad_before = nmad_after

    return ~dh_stable.mask


def spatial_filter_ref(ref_dem: np.ndarray, src_dem: np.ndarray, radius_pix: float, dh_thresh: float) -> np.ndarray:
    """
    Masks all values where src_dem < min_ref - dh_thresh & src_dem > max_ref + dh_thresh.
    where min_ref and max_ref are the min/max elevation of ref_dem within radius.

    :param ref_dem: 2D array containing the reference elevation.
    :param src_dem: 2D array containing the DEM to be filtered, of same size as ref_dem.
    :param radius_pix: the radius of the disk where to calculate min/max ref elevation.
    :param dh_thresh: the second elevation can be this far below/above the min/max ref elevation.

    :returns: a boolean 2D array set to True for pixels to be masked
    """
    # Sanity check
    assert ref_dem.shape == src_dem.shape, "Input arrays have different shape"
    assert np.ndim(ref_dem) == np.ndim(src_dem), "Input arrays must be of dimension 2"

    # Calculate ref min.max elevation in given radius
    max_elev = cv2.dilate(ref_dem, kernel=disk(radius_pix))
    min_elev = cv2.erode(ref_dem, kernel=disk(radius_pix))

    # Pixels to be masked
    mask = np.zeros(ref_dem.shape, dtype="bool")
    mask[src_dem < min_elev - dh_thresh] = True
    mask[src_dem > max_elev + dh_thresh] = True

    return mask


def spatial_filter_ref_iter(
    ref_dem: np.ndarray, src_dem: np.ndarray, res: float, plot: bool = False, vmax: float = 50
) -> np.ndarray:
    """
    Apply spatial_filter_ref with iterative hard-coded thresholds, as in Hugonnet et al. (2021) - S1:
    radius, dh_cutoff = (200, 700), (500, 500), (1000, 300).

    :param ref_dem: 2D array containing the reference elevation.
    :param src_dem: 2D array containing the DEM to be filtered, of same size as ref_dem.
    :param res: DEm resolution in meters.
    :param plot: set to True to display intermediate results.
    :param vmax: maximum color scale value, if plot = True.

    :returns: a boolean 2D array set to True for pixels to be masked.
    """
    mask = np.zeros(ref_dem.shape, dtype="bool")

    for radius, dh_cutoff in zip((200, 500, 1000), (700, 500, 300)):

        radius_pix = int(np.ceil(radius / res))
        mask += spatial_filter_ref(ref_dem, src_dem, radius_pix, dh_cutoff)

        if plot:
            dh = ref_dem - src_dem
            dh_filtered = np.ma.where(mask, np.nan, dh)

            plt.figure(figsize=(16, 8))

            ax1 = plt.subplot(121)
            plt.imshow(dh, cmap="coolwarm_r", vmin=-vmax, vmax=vmax)
            cb = plt.colorbar()
            cb.set_label("Elevation change (m)")
            plt.title("Initial dh")

            plt.subplot(122, sharex=ax1, sharey=ax1)
            plt.imshow(dh_filtered, cmap="coolwarm_r", vmin=-vmax, vmax=vmax)
            cb = plt.colorbar()
            cb.set_label("Elevation change (m)")
            plt.title(f"Filtered dh - radius = {radius}, dh_cutoff = {dh_cutoff}")

            plt.tight_layout()
            plt.show()

    return mask


def postprocessing_single(
    dem_path: str,
    ref_dem_path: str,
    roi_outlines: gu.Vector,
    all_outlines: gu.Vector,
    out_dem_path: str,
    coreg_method: xdem.coreg | None = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian),
    filtering: bool = True,
    plot: bool = False,
    out_fig: str = None,
    verbose: bool = False,
):
    """
    Coregister a selected DEM to a reference DEM.

    Reads both DEMs, reproject DEM onto ref DEM grid, mask content of outlines, run the coregistration and save the coregistered DEM as well as some optional figures and returns some statistics.

    :param dem_path: Path to the input DEM to be coregistered
    :param ref_dem: the reference DEM
    :param roi_outlines: The outlines of the glacier to study
    :param all_outlines: The outlines of all glaciers in the study area
    :param out_dem_path: Path where to save the coregistered DEM
    :param coreg_method: The xdem coregistration method, or pipeline. If set to None, DEMs will be resampled to ref grid and optionally filtered, but not coregistered.
    :param filtering: if set to True, final DEM will be filtered based on distance to ref, following the method of Hugonnet et al. (2021), equation S1.
    :param plot: Set to True to plot a figure of elevation diff before/after coregistration
    :param out_fig: Path to the output figure. If None will display to screen.
    :param verbose: set to True to print details on screen during coregistration.

    :returns: a tuple containing - basename of coregistered DEM, [count of obs, median and NMAD over stable terrain, coverage over roi] before coreg, [same stats] after coreg
    """
    # Load DEM and reproject to ref grid
    
    ref_dem = xdem.DEM(ref_dem_path)
    dem = xdem.DEM(dem_path)
    ref_dem = ref_dem.reproject(dem,resampling='bilinear')

    

    # Create masks
    roi_mask = roi_outlines.create_mask(dem)
    stable_mask = ~all_outlines.create_mask(dem)

    # Calculate dDEM
    ddem = dem - ref_dem

    # Filter gross outliers in stable terrain
    inlier_mask = nmad_filter(ddem.data, stable_mask, verbose=False)
    outlier_mask = ~inlier_mask & stable_mask
    ddem.data.mask[outlier_mask] = True
    del inlier_mask

    # Calculate coverage on and off ice
    roi_coverage_orig, nstable_orig, med_orig, nmad_orig = calculate_stats(ddem, roi_mask, stable_mask)

    # Coregister to reference - Note: this will spread NaN
    # Better strategy: calculate shift, update transform, resample
    if isinstance(coreg_method, xdem.coreg.Coreg):
        # coreg_method = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian)
        coreg_method.fit(ref_dem, dem, stable_mask, verbose=verbose)
        dem_coreg = coreg_method.apply(dem, dilate_mask=False)
    elif coreg_method is None:
        dem_coreg = dem
    ddem_coreg = dem_coreg - ref_dem

    # Calculate new stats
    roi_coverage_coreg, nstable_coreg, med_coreg, nmad_coreg = calculate_stats(ddem_coreg, roi_mask, stable_mask)

    # Filter outliers based on reference DEM
    if filtering:
        outlier_mask = spatial_filter_ref_iter(
            ref_dem.data.squeeze(), dem_coreg.data.squeeze(), res=ref_dem.res[0], plot=False
        )
        nfiltered = len(np.where(outlier_mask & ~dem_coreg.data.mask)[0])
        dem_coreg.data.mask[0, outlier_mask] = True
    else:
        nfiltered = 0

    # Save plots
    if plot:
        plt.figure(figsize=(11, 5))

        ax1 = plt.subplot(121)
        plt.imshow(ddem.data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
        cb = plt.colorbar()
        cb.set_label("Elevation change (m)")
        plt.title(f"Before coreg - med = {med_orig:.2f} m - NMAD = {nmad_orig:.2f} m")

        ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
        plt.imshow(ddem_coreg.data.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
        cb = plt.colorbar()
        cb.set_label("Elevation change (m)")
        plt.title(f"After coreg - med = {med_coreg:.2f} m - NMAD = {nmad_coreg:.2f} m")

        # ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
        # plt.imshow(diff_filtered.squeeze(), cmap="coolwarm_r", vmin=-50, vmax=50)
        # plt.colorbar()
        # plt.title(f"After filtering - NMAD = {nmad_filtered:.2f} m")

        plt.tight_layout()
        if out_fig is None:
            plt.show()
        else:
            plt.savefig(out_fig, dpi=200)
            plt.close()

    # Save coregistered DEM
    # project to reference DEM grid for stacking and hole filling later
    dem_coreg.reproject(xdem.DEM(ref_dem_path,load_data=False),resampling='bilinear').save(out_dem_path,tiled=True)
    #dem_coreg.save(out_dem_path, tiled=True)

    return (
        os.path.basename(dem_path),
        out_dem_path,
        nstable_orig,
        med_orig,
        nmad_orig,
        roi_coverage_orig,
        nstable_coreg,
        med_coreg,
        nmad_coreg,
        roi_coverage_coreg,
        nfiltered
    )


def postprocessing_all(
    dem_path_list,
    ref_dem,
    roi_outlines,
    all_outlines,
    outdir,
    coreg_method: xdem.coreg | None = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian),
    filtering: bool = True,
    overwrite: bool = False,
    plot: bool = False,
    nthreads: int = 1,
    mp_method: str = "mp",
):
    """
    Run the postprocessing for all DEMs in dem_path_list.
    dem_path_list can be either a list of paths, or a list of list of paths (to allow grouping files).

    Return: pd.Series containing output stats, list of output files paths (same shape as dem_path_list)
    """
    # Check that chosen mp_method is correct
    if not mp_method in ["mp", "concurrent"]:
        raise ValueError(f"mp_method must be either 'mp' or 'concurrent', currently set to '{mp_method}'")

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # List the filenames of the DEMs to be processed
    if not overwrite:
        existing = glob(outdir + "/*_coreg.tif")
        existing = [os.path.basename(fp).replace("_coreg", "") for fp in existing]
    else:
        existing = []

    # If inputs is a list of list
    if isinstance(dem_path_list[0], (list, tuple, np.ndarray)):
        dem_to_process = [fp for group in dem_path_list for fp in group if os.path.basename(fp) not in existing]
    # If inputs is a list of strings
    elif isinstance(dem_path_list[0], str):
        dem_to_process = [fp for fp in dem_path_list if os.path.basename(fp) not in existing]
    else:
        raise ValueError("Input `dem_path_list` not understood, must be a list of strings, or list of list of strings")

    # Remove possible duplicates
    dem_to_process = np.unique(dem_to_process)

    # Needed to avoid errors when plotting on MacOS
    old_backend = mpl.get_backend()
    mpl.use("Agg")

    global _postproc_wrapper

    def _postproc_wrapper(dem_path):
        """Postprocess the DEMs in one thread."""
        # Path to outputs
        out_dem_path = os.path.join(outdir, os.path.basename(dem_path).replace(".tif", "_coreg.tif"))
        out_fig = out_dem_path.replace(".tif", "_diff.png")
        try:
            outputs = postprocessing_single(
                dem_path, ref_dem, roi_outlines, all_outlines, out_dem_path, coreg_method=coreg_method, filtering=filtering, plot=plot, out_fig=out_fig
            )
        # In case there are too few points, coregistration fail -> skip
        # TODO: need to catch that error in a more rigorous way.
        except (ValueError, AssertionError) as e:
            outputs = tuple([None]*11)

        return outputs

    # Arguments to be used for the progress bar
    pbar_kwargs = {"total": len(dem_to_process), "desc": "Postprocessing DEMs", "smoothing": 0}

    # Run with either 1 or several threads
    if nthreads == 1:
        results = []
        for dem_path in tqdm(dem_to_process, **pbar_kwargs):
            output = _postproc_wrapper(dem_path)
            results.append(output)

    elif nthreads > 1:

        if mp_method == "mp":
            # Needed for MacOS with Python >= 3.8
            # See https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
            cx = mp.get_context("fork")
            with cx.Pool(nthreads) as pool:
                results = list(tqdm(pool.imap(_postproc_wrapper, dem_to_process, chunksize=1), **pbar_kwargs))
                pool.close()
                pool.join()

        elif mp_method == "concurrent":
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = list(tqdm(executor.map(_postproc_wrapper, dem_to_process), **pbar_kwargs))

    else:
        raise ValueError("nthreads must be >= 1")

    # Revert to original backend
    mpl.use(old_backend)

    # -- Save generic stats to file -- #

    # Convert output data to DataFrame
    df = pd.DataFrame(
        results,
        columns=[
            "ID",
            "coreg_path",
            "nstable_orig",
            "med_orig",
            "nmad_orig",
            "roi_cover_orig",
            "nstable_coreg",
            "med_coreg",
            "nmad_coreg",
            "roi_cover_coreg",
            "count_filtered_pixels",
        ],
    )
    
    df.dropna(inplace=True)
    # Read stats from previous run
    stats_file = outdir + "/coreg_stats.txt"
    if os.path.exists(stats_file) & (not overwrite):
        df_previous = pd.read_csv(stats_file)
        out_df = (
            pd.concat((df_previous, df), ignore_index=True)
            .drop_duplicates(keep="last", subset=["ID"])
            .sort_values(by=["ID"])
        )
        
        # Change hard coded file paths if generated on another system
        BASE_DIR = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
        out_df['coreg_path'] = BASE_DIR+os.sep+out_df['coreg_path'].str.split('ragmac_xdem/').str[-1]

    else:
        out_df = df

    # Save concatenated output to file
    out_df.to_csv(stats_file, index=False, float_format="%.2f")

    # Get DEM Ids of files to be returned
    out_paths = []
    if isinstance(dem_path_list[0], (list, tuple, np.ndarray)):
        for group in dem_path_list:
            dem_IDs = np.asarray([os.path.basename(dem_path) for dem_path in group])
            out_paths.append(out_df[out_df["ID"].isin(dem_IDs)]["coreg_path"].values)
    elif isinstance(dem_path_list[0], str):
        dem_IDs = np.asarray([os.path.basename(dem_path) for dem_path in dem_path_list])
        out_paths.extend(out_df[out_df["ID"].isin(dem_IDs)]["coreg_path"].values)

    return out_df, out_paths


def merge_and_calculate_ddems(groups, validation_dates, ref_dem, mode, outdir, overwrite=False, nproc=None):
    """
    A function that takes all DEMs grouped by validation date, and merge and/or interpolate them in time to provide an elevation change for each validation periods.
    """
    # Figure out all possible ddem pairs and IDs
    pair_indexes, pair_ids = utils.list_pairs(validation_dates)

    # Pattern for output file names
    def fname_func(pair_id):
        return os.path.join(outdir, f"ddem_{pair_id}_mode_{mode}.tif")

    # Output dictionary, to contain all pairwise ddem arrays
    ddems = {}

    # Check if output files exist
    outfile_exist = []
    for key in pair_ids:
        fname = fname_func(key)
        outfile_exist.append(os.path.exists(fname))

    # If files exist, simply load them
    if np.all(outfile_exist) and (not overwrite):
        print("Loading existing files")
        for key in pair_ids:
            fname = fname_func(key)
            ddems[key] = gu.Raster(fname)
        return ddems

    if mode == "median":

        # First stack all DEMs and reproject to reference if needed
        print("Loading and stacking DEMs")
        mosaics = {}
        for date, dems_list in zip(validation_dates, groups):
            print(date)
            dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in dems_list]
            dem_stack = gu.spatial_tools.stack_rasters(dem_objs, reference=ref_dem, use_ref_bounds=True)
            mosaics[date] = np.ma.median(dem_stack.data, axis=0)

        # Then calculate elevation changes for each subperiod
        print("Calculating elevation changes")
        for k1, k2 in pair_indexes:
            date1 = validation_dates[k1]
            date2 = validation_dates[k2]
            if (date1 in mosaics.keys()) & (date2 in mosaics.keys()):
                pair_id = f"{date1[:4]}_{date2[:4]}"  # year1_year2
                ddems[pair_id] = mosaics[date2] - mosaics[date1]

    elif mode == "shean":
        
        for count, pair in enumerate(pair_indexes):
            k1, k2 = pair
            dems_list = groups[count]
            pair_id = pair_ids[count]
            print(f"\nProcessing pair {pair_id}")

            # First stack all DEMs and reproject to reference if needed
            dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in dems_list]
            dem_stack = gu.spatial_tools.stack_rasters(dem_objs, reference=ref_dem, use_ref_bounds=True)

            # Then calculate linear fit
            dem_dates = utils.get_dems_date(dems_list)
            results = temporal.ma_linreg(
                dem_stack.data,
                dem_dates,
                n_thresh=5,
                model="theilsen",
                parallel=True,
                n_cpu=nproc,
                dt_stack_ptp=None,
                min_dt_ptp=5,
                smooth=False,
                rsq=False,
                conf_test=False,
                remove_outliers=False,
            )

            slope, intercept, detrended_std = results

            # Finally, calculate total elevation change
            date1 = validation_dates[k1]
            date2 = validation_dates[k2]
            date1_dt = datetime.strptime(date1, "%Y-%m-%d")
            date2_dt = datetime.strptime(date2, "%Y-%m-%d")
            dyear = (date2_dt - date1_dt).total_seconds() / (3600 * 24 * 365.25)
            ddems[pair_id] = dyear * slope

    elif mode == "TimeSeries2":
        
        for count, pair in enumerate(pair_indexes):
            k1, k2 = pair
            dems_list = groups[count]
            pair_id = pair_ids[count]
            print(f"\nProcessing pair {pair_id}")

            # First stack all DEMs
            start = datetime.now()
            dem_dates = utils.get_dems_date(dems_list)
            ds = io.xr_stack_geotifs(dems_list, dem_dates, ref_dem.filename)
            ma_stack = np.ma.masked_invalid(ds["band1"].values)

            print("\n### Prepare training data ###")
            ## TODO: Determine which time conversion to use as this slightly changes the result.
#             X_train = np.ma.array([utils.date_time_to_decyear(i) for i in dem_dates]).data
            X_train = np.array(matplotlib.dates.date2num(dem_dates))
    
            n_thresh = 5
            print('Excluding pixels with count <',n_thresh)
            valid_data, valid_mask_2D = temporal.mask_low_count_pixels(ma_stack, n_thresh=n_thresh)

            # Then calculate linear fit
            print("\n### Run linear fit ###")
            results = temporal.linreg_run_parallel(X_train, valid_data, cpu_count=nproc, method="TheilSen")
            
            results_stack = temporal.linreg_reshape_parallel_results(results, ma_stack, valid_mask_2D)

            slope = results_stack[0]
            ## Only do this if X_train was generated with matplotlib.dates.date2num. Same procedure as in ma_linreg
            slope *= 365.25


            now = datetime.now()
            dt = now - start
            print("Elapsed time:", str(dt).split(".")[0])

            # Not needed for now
            # intercept = results_stack[1]
            # print('\n### Compute residuals ###')
            # prediction = temporal.linreg_predict_parallel(slope,X_train,intercept, cpu_count=nproc)
            # residuals  = ma_stack - prediction
            # residuals[:,valid_mask_2D]  = residuals[:,valid_mask_2D]

            # now = datetime.now()
            # dt = now-start
            # print('Elapsed time:', str(dt).split(".")[0])

            # print('\n### Compute MAD ###')
            # detrended_std = temporal.mad(residuals, axis=0)

            # now = datetime.now()
            # dt = now-start
            # print('Elapsed time:', str(dt).split(".")[0])

            # Finally, calculate total elevation change
            date1 = validation_dates[k1]
            date2 = validation_dates[k2]
            date1_dt = datetime.strptime(date1, "%Y-%m-%d")
            date2_dt = datetime.strptime(date2, "%Y-%m-%d")
            dyear = (date2_dt - date1_dt).total_seconds() / (3600 * 24 * 365.25)
            ddems[pair_id] = dyear * slope
            
    elif mode == "TimeSeries3":

        for count, pair in enumerate(pair_indexes):
            k1, k2 = pair
            dems_list = groups[count]
            pair_id = pair_ids[count]
            print(f"\nProcessing pair {pair_id}")
            
            dems_list = groups[count]
            dem_dates = utils.get_dems_date(dems_list)
             
            time_stamps = np.array(matplotlib.dates.date2num(dem_dates))
#             time_stamps = np.array([utils.date_time_to_decyear(i) for i in dem_dates])
            
    
            ds = io.xr_stack_geotifs(dems_list, dem_dates, ref_dem.filename)
            t = len(ds.time)
            x = len(ds.x)
            y = len(ds.y)
            print('\ndata dims: x, y, time')
            print('data shape:',x,y,t)
            
            ## Optimize chunking WIP
            
            # option 1
#             ds['band1'].data = ds['band1'].data.rechunk({0:-1, 1:'auto', 2:'auto'}, 
#                                                         block_size_limit=1e8, 
#                                                         balance=True)
#             arr = ds['band1'].data
#             t,y,x = arr.chunks[0][0], arr.chunks[1][0], arr.chunks[2][0]
#             tasks_count = io.dask_get_mapped_tasks(ds['band1'].data)
            
            # option 2
            mx, my, t = io.optimize_dask_chunks(ds)
            x = mx
            y = my
            ds = ds.chunk({"time":t, "x":mx, "y":my})
            tasks_count = io.dask_get_mapped_tasks(ds['band1'].data)
            while tasks_count > 10000:
                # increase chunk shape to reduce tasks
                x += mx
                y += my
                # task_count increases if ds is not re-initialized
                ds = io.xr_stack_geotifs(dems_list, dem_dates, ref_dem.filename)
                ds = ds.chunk({"time":t, "x":x, "y":y})
                tasks_count = io.dask_get_mapped_tasks(ds['band1'].data)
            
            # see what we end up with
            print('chunk shape:', x,y,t)
            chunksize = ds['band1'][:t,:y,:x].nbytes / 1048576
            print('chunk size:',np.round(chunksize,2), 'MiB')
            print('tasks:', tasks_count)
            
            # TODO pass down client object to print address here instead of earlier.
            print('\nCheck dask dashboard to monitor progress. See stdout above for address to dashboard.')
            
            n_thresh = 5
            print('\nExcluding pixels with count <',n_thresh)
            results = temporal.dask_apply_linreg(ds['band1'],
                                                 'time', 
                                                 kwargs={'times':time_stamps,
                                                         'n_thresh':n_thresh})
            
            start = datetime.now()
            results = xr.Dataset({'slope':results[0],
                                  'intercept':results[1]}).compute()
            now = datetime.now()
            dt = now - start
            print("Elapsed dask compute time:", str(dt).split(".")[0])
            
            # need to convert timestamps back to yr-1 if using matplotlib.dates.date2num
            # these same conversions are used in temporal.ma_linreg()
            results['slope'] = results['slope'] * 365.25
            
            slope = np.ma.masked_invalid(results['slope'].values)

            # Finally, calculate total elevation change
            date1 = validation_dates[k1]
            date2 = validation_dates[k2]
            date1_dt = datetime.strptime(date1, "%Y-%m-%d")
            date2_dt = datetime.strptime(date2, "%Y-%m-%d")
            dyear = (date2_dt - date1_dt).total_seconds() / (3600 * 24 * 365.25)
            ddems[pair_id] = dyear * slope

    else:
        raise NotImplementedError("`mode` must be either of 'median', 'shean', 'TimeSeries2, or TimeSeries3'")

    # Convert masked arrays to gu.Raster
    for key in list(ddems.keys()):
        ddems[key] = gu.Raster.from_array(ddems[key], ref_dem.transform, ref_dem.crs, nodata=-9999)

    # Save files
    for key in list(ddems.keys()):
        fname = fname_func(key)
        ddems[key].save(fname)
        print(f"Saved to file {fname}")

    return ddems
