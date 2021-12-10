"""
Set of tools to postprocess the DEMs: statistics calculations, coregistration, filtering etc
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from glob import glob
import threading

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xdem
from tqdm import tqdm

# Needed for MacOS with Python >= 3.8
# See https://stackoverflow.com/questions/60518386/error-with-module-multiprocessing-under-python3-8
mp.set_start_method("fork")


def calculate_stats(ddem, roi_mask, stable_mask):
    """
    Returns main statistics of ddem:
    fraction of coverage over glacier, number of obs, NMAD and median dh over stable terrain
    """
    # Calculate coverage over glaciers
    nobs = np.sum(~ddem.data.mask[roi_mask])
    ntot = np.sum(roi_mask)
    roi_coverage = nobs / ntot

    # Calculate statistics in stable terrain
    nstable = np.sum(~ddem.data.mask[stable_mask])
    nmad_stable = xdem.spatialstats.nmad(ddem.data[stable_mask])
    med_stable = np.ma.median(ddem.data[stable_mask])

    return roi_coverage, nstable, med_stable, nmad_stable


def postprocessing_single(
    dem_path: str,
    ref_dem: xdem.DEM,
    roi_outlines: gu.Vector,
    all_outlines: gu.Vector,
    out_dem_path: str,
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
    :param plot: Set to True to plot a figure of elevation diff before/after coregistration
    :param out_fig: Path to the output figure. If None will display to screen.
    :param verbose: set to True to print details on screen during coregistration.

    :returns: a tuple containing - basename of coregistered DEM, [count of obs, median and NMAD over stable terrain, coverage over roi] before coreg, [same stats] after coreg
    """
    # Load DEM and reproject to ref grid
    dem = xdem.DEM(dem_path)
    dem = dem.reproject(ref_dem, resampling="bilinear")

    # Create masks
    roi_mask = roi_outlines.create_mask(dem)
    stable_mask = ~all_outlines.create_mask(dem)

    # Calculate dDEM
    ddem = dem - ref_dem

    # Calculate coverage on and off ice
    roi_coverage_orig, nstable_orig, med_orig, nmad_orig = calculate_stats(ddem, roi_mask, stable_mask)

    # Coregister to reference - Note: this will spread NaN
    # Better strategy: calculate shift, update transform, resample
    coreg = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian)
    coreg.fit(ref_dem, dem, stable_mask, verbose=verbose)
    dem_coreg = coreg.apply(dem, dilate_mask=False)
    ddem_coreg = dem_coreg - ref_dem

    # Calculate new stats
    roi_coverage_coreg, nstable_coreg, med_coreg, nmad_coreg = calculate_stats(ddem_coreg, roi_mask, stable_mask)

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
    dem_coreg.save(out_dem_path, tiled=True)

    return (
        os.path.basename(dem_path),
        nstable_orig,
        med_orig,
        nmad_orig,
        roi_coverage_orig,
        nstable_coreg,
        med_coreg,
        nmad_coreg,
        roi_coverage_coreg,
    )



def postprocessing_all(
        dem_path_list, ref_dem, roi_outlines, all_outlines, outdir, overwrite: bool = False, nthreads: int = 1, method: str = 'mp'
):
    """
    Run the postprocessing for all DEMs in dem_path_list.
    """
    # Create threading locks for files/variables that will be read in multiple threads.
    if method == "mp":
        progress_bar_lock = mp.Lock()
    elif method == "concurrent":
        progress_bar_lock = threading.Lock()
    else:
        raise ValueError(f"method must be either 'mp' or 'concurrent', set to {method}")

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # List the filenames of the DEMs to be processed
    if not overwrite:
        existing = glob(outdir + "/*_coreg.tif")
        existing = [os.path.basename(fp).replace("_coreg", "") for fp in existing]
        dem_path_list = [fp for fp in dem_path_list if os.path.basename(fp) not in existing]

    # Create progress bar
    progress_bar = tqdm(total=len(dem_path_list), desc="Postprocessing DEMs", smoothing=0)

    global _postproc_wrapper
    def _postproc_wrapper(dem_path):
        """Postprocess the DEMs in one thread."""
        # Path to outputs
        out_dem_path = os.path.join(outdir, os.path.basename(dem_path).replace(".tif", "_coreg.tif"))
        out_fig = out_dem_path.replace(".tif", "_diff.png")
        outputs = postprocessing_single(dem_path, ref_dem, roi_outlines, all_outlines, out_dem_path, plot=False, out_fig=out_fig)
        with progress_bar_lock:
            progress_bar.update()
        return outputs

    # Run with either 1 or several threads
    if nthreads == 1:
        results = []
        for dem_path in dem_path_list:
            output = _postproc_wrapper(dem_path)
            results.append(output)
    elif nthreads > 1:
        if method == 'mp':
            #with mp.Pool(nthreads) as pool:
            pool = mp.Pool(nthreads)
            results = list(pool.map(_postproc_wrapper, dem_path_list, chunksize=1))
            pool.close()
            pool.join()
        elif method == 'concurrent':
            with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
                results = list(executor.map(_postproc_wrapper, dem_path_list))
    else:
        raise ValueError("nthreads must be >= 1")

    # # -- Save generic stats to file -- #

    # # Convert output data to DataFrame
    # df = pd.DataFrame(results, columns=["ID", "nstable_orig", "med_orig", "nmad_orig", "roi_cover_orig", "nstable_coreg", "med_coreg", "nmad_coreg", "roi_cover_coreg"])

    # # Read stats from previous run
    # stats_file = outdir + "/coreg_stats.txt"
    # if os.path.exists(stats_file) & (not overwrite):
    #     df_previous = pd.read_csv(stats_file)
    #     out_df = (
    #         pd.concat((df_previous, df), ignore_index=True)
    #         .drop_duplicates(keep="last", subset=["ID"])
    #         .sort_values(by=["ID"])
    #     )
    # else:
    #     out_df = df

    # # Save concatenated output to file
    # out_df.to_csv(stats_file, index=False, float_format="%.2f")

    return results
