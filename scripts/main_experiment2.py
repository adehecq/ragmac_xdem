"""
The main script to be run for experiment_2
"""

import argparse
import multiprocessing as mp
import os
import warnings

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import xdem

import ragmac_xdem.dem_postprocessing as pproc

from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils, files


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
    selection_opts = {"mode": "temporal", "dt": 365, "months": [8, 9, 10]}
    validation_dates = baltoro_paths["validation_dates"]
    groups = utils.dems_selection(dems_files, validation_dates=validation_dates, **selection_opts)
    dems_files = [item for sublist in groups for item in sublist]

    # -- Postprocess DEMs i.e. coregister, filter etc -- #
    print("\n### Coregister DEMs ###")
    stats = pproc.postprocessing_all(
        dems_files,
        ref_dem,
        roi_outlines,
        all_outlines,
        outdir,
        nthreads=args.nproc,
        overwrite=args.overwrite,
        plot=True,
        method="mp",
    )
    coreg_dems_files = np.asarray(stats["coreg_path"])
    groups_coreg = utils.dems_selection(coreg_dems_files, validation_dates, dt=365)
    print(f"--> Coregistered DEMs saved in {outdir}")

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")

    mosaics = {}
    for date, group in zip(validation_dates, groups_coreg):
        print(date)
        if len(group) > 0:
            mosaics[date] = {}
            dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in group]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mosaic = gu.spatial_tools.merge_rasters(
                    dem_objs, reference=ref_dem, merge_algorithm=np.nanmedian, use_ref_bounds=True
                )
            ddem = ref_dem - mosaic
            cov, _, med, nmad = pproc.calculate_stats(ddem, roi_mask, stable_mask)
            mosaics[date]["dem"] = mosaic
            mosaics[date]["ddem"] = ddem
            mosaics[date]["cov"] = cov
            mosaics[date]["med"] = med
            mosaics[date]["nmad"] = nmad

    # -- Calculate elevation change for all periods -- #

    print("\n### Calculate dDEMs for all subperiods ###")
    ddems = {}
    for k1 in range(len(validation_dates)):
        for k2 in range(k1 + 1, len(validation_dates)):
            date1 = validation_dates[k1]
            date2 = validation_dates[k2]
            if (date1 in mosaics.keys()) & (date2 in mosaics.keys()):
                pair_id = f"{date1[:4]}_{date2[:4]}"  # year1_year2
                print(pair_id)
                ddems[pair_id] = mosaics[date2]["dem"] - mosaics[date1]["dem"]

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
    plt.show()

    # -- Calculating MB -- #
    print("\n### Calculating mass balance ###")
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        ddem_bins, bins_area, frac_obs, dV, dh_mean = mb.mass_balance_local_hypso(
            ddems[pair_id], ref_dem, roi_mask, plot=True
        )
        print(f"Total volume: {dV:.1f} km3 - mean dh: {dh_mean:.2f} m")
