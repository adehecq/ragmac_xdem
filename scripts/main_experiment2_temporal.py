#! /usr/bin/env python

"""
The main script to be run for experiment_2, with temporal fitting of the DEM time series
"""

import argparse
import multiprocessing as mp
import os
import warnings
from datetime import datetime

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import xdem
import rasterio as rio

import ragmac_xdem.dem_postprocessing as pproc

from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils, files, temporal


if __name__ == "__main__":

    # -- Setup script arguments -- #
    parser = argparse.ArgumentParser(description="Process all the data and figures for experiment 2")

    parser.add_argument(
        "-sat",
        dest="sat_type",
        type=str,
        default="ASTER",
        help="str, the satellite data to be used, either 'ASTER', 'TDX' or 'both'",
    )
    parser.add_argument(
        "-overwrite", dest="overwrite", action="store_true", help="If set, will overwrite already processed data"
    )
    parser.add_argument(
        "-nproc",
        dest="nproc",
        type=int,
        default=mp.cpu_count() - 1,
        help="int, number of processes to be run in parallel whenever possible (Default is max CPU - 1)",
    )

    args = parser.parse_args()

    # -- Load input data -- #
    baltoro_paths = files.get_data_paths("PK_Baltoro")
    ref_dem, all_outlines, roi_outlines, roi_mask, stable_mask = utils.load_ref_and_masks(baltoro_paths)
    
    # Get list of all DEMs and set output directory
    if args.sat_type == "ASTER":
        dems_files = baltoro_paths["raw_data"]["aster_dems"]
        outdir = baltoro_paths["processed_data"]["aster_dir"]

    elif args.sat_type == "TDX":
        dems_files = baltoro_paths["raw_data"]["tdx_dems"]
        outdir = baltoro_paths["processed_data"]["tdx_dir"]
    else:
        raise NotImplementedError

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -- Calculate initial DEM statistics -- #
    print("\n### Calculate initial statistics ###")
    stats_file = os.path.join(outdir, 'init_stats.csv')
    init_stats = pproc.calculate_init_stats_parallel(dems_files, ref_dem, roi_outlines, all_outlines, stats_file, nthreads=args.nproc, overwrite=args.overwrite)
    print(f"Statistics file saved to {stats_file}")

    # -- Select DEMs to be processed -- #
    print("\n### DEMs selection ###")
    selection_opts = {"mode": None}
    validation_dates = baltoro_paths["validation_dates"]
    groups = utils.dems_selection(dems_files, validation_dates=validation_dates, **selection_opts)

    # -- Postprocess DEMs i.e. coregister, filter etc -- #
    print("\n### Coregister DEMs ###")
    stats, groups_coreg = pproc.postprocessing_all(
        groups,
        ref_dem,
        roi_outlines,
        all_outlines,
        outdir,
        nthreads=args.nproc,
        overwrite=args.overwrite,
        plot=True,
        method="mp",
    )
    print(f"--> Coregistered DEMs saved in {outdir}")

    # -- Temporally interpolate elevation at validation dates -- #
    print("\n### DEMs temporal fit ###")

    # Output files
    dh_slope_fn, dh_intercept_fn, dh_std_fn = [
        os.path.join(outdir, fn) for fn in ['dh_slope.tif', 'dh_intercept.tif', 'dh_std.tif']
    ]

    # Need to downsample DEM for speeding-up process (~ 2h with 7 cores otherwise)
    ref_dem_lowres = ref_dem.reproject(dst_res=5 * ref_dem.res[0])
    roi_mask = roi_outlines.create_mask(ref_dem_lowres)
    
    args.overwrite = True
    if args.overwrite or np.any([not os.path.exists(fn) for fn in [dh_slope_fn, dh_intercept_fn, dh_std_fn]]):
        dems_list = groups[0]
        dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in dems_list]
        print("Loading and stacking DEMs")
        dem_stack = gu.spatial_tools.stack_rasters(dem_objs, reference=ref_dem_lowres, use_ref_bounds=True)
        dem_dates = utils.get_dems_date(dems_list)
        common_mask = np.ma.getmaskarray(dem_stack.data).all(axis=0)
        print("Temporal fit")
        # stride = 1
        # ma_stack_test = dem_stack.data[:, ::stride, ::stride]
        # transform = rio.transform.from_origin(
        #     ref_dem.bounds.left, ref_dem.bounds.top, ref_dem.res[0] * stride, ref_dem.res[1] * stride
        # )

        results = temporal.ma_linreg(dem_stack.data, dem_dates, n_thresh=3, model='theilsen', parallel=True,
                                     n_cpu=args.nproc, dt_stack_ptp=None, min_dt_ptp=None, smooth=False, rsq=False,
                                     conf_test=False, remove_outliers=False)

        dh_slope = gu.Raster.from_array(results[0], ref_dem_lowres.transform, ref_dem.crs, nodata=-9999)
        dh_intercept = gu.Raster.from_array(results[1], ref_dem_lowres.transform, ref_dem.crs, nodata=-9999)
        dh_detrended_std = gu.Raster.from_array(results[2], ref_dem_lowres.transform, ref_dem.crs, nodata=-9999)

        # Save to file
        dh_slope.save(dh_slope_fn)
        dh_intercept.save(dh_intercept_fn)
        dh_detrended_std.save(dh_std_fn)

    else:
        dh_slope = gu.Raster(dh_slope_fn)
        dh_intercept = gu.Raster(dh_intercept_fn)
        dh_detrended_std = gu.Raster(dh_std_fn)

    # -- Calculate elevation change for all periods -- #

    print("\n### Calculate dDEMs for all subperiods ###")
    ddems = {}
    for k1 in range(len(validation_dates)):
        for k2 in range(k1 + 1, len(validation_dates)):
            date1, date2 = validation_dates[k1], validation_dates[k2]
            date1_dt = datetime.strptime(date1, "%Y-%m-%d")
            date2_dt = datetime.strptime(date2, "%Y-%m-%d")
            dyear = (date2_dt - date1_dt).total_seconds() / (3600 * 24 *365.25)

            pair_id = f"{date1[:4]}_{date2[:4]}"  # year1_year2
            print(pair_id)
            ddem = gu.Raster.from_array(dyear * dh_slope.data, transform=dh_slope.transform, crs=dh_slope.crs, nodata=-9999)
            ddems[pair_id] = ddem

    # -- Plot -- #

    # Number of subplots needed and figsize
    nsub = len(ddems)
    figsize = (8 * nsub, 6)

    plt.figure(figsize=figsize)

    for k, pair_id in enumerate(ddems):

        ax = plt.subplot(1, nsub, k + 1)
        roi_outlines.ds.plot(ax=ax, facecolor="none", edgecolor="k", zorder=2)
        ddems[pair_id].show(ax=ax, cmap="coolwarm_r", vmin=-50, vmax=50, cb_title="Elevation change (m)", zorder=1)
        ax.set_title(pair_id)

    plt.tight_layout()
    fig_fn = os.path.join(outdir, 'ddem_temporal_fig.png')
    plt.savefig(fig_fn)
    #plt.show()

    # -- Calculating MB -- #
    print("\n### Calculating mass balance ###")
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        fig_fn = os.path.join(outdir, f'{pair_id}_temporal_mb_fig.png')
        ddem_bins, bins_area, frac_obs, dV, dh_mean = mb.mass_balance_local_hypso(
            ddems[pair_id], ref_dem_lowres, roi_mask, plot=True, outfig=fig_fn
        )
        print(f"Total volume: {dV:.1f} km3 - mean dh: {dh_mean:.2f} m")
