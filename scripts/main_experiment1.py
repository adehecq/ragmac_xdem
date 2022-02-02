"""
The main script to be run for experiment_1
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
from ragmac_xdem import files
from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils
from ragmac_xdem import uncertainty as err

if __name__ == "__main__":

    # -- Setup script arguments -- #
    parser = argparse.ArgumentParser(description="Process all the data and figures for experiment 1")

    parser.add_argument(
        "region",
        type=str,
        help="str, the region to run 'AT_Hintereisferner' or 'CH_Aletschgletscher'",
    )

    parser.add_argument(
        "-sat",
        dest="sat_type",
        type=str,
        default="ASTER",
        help="str, the satellite data to be used, either 'ASTER', 'TDX' or 'both'",
    )
    parser.add_argument(
        "-mode",
        dest="mode",
        type=str,
        default="median",
        help="str, processing mode, either of 'median', 'shean' or 'knuth'",
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
    exp = files.get_data_paths(args.region)
    ref_dem, all_outlines, roi_outlines, roi_mask, stable_mask = utils.load_ref_and_masks(exp)

    # Get list of all DEMs and set output directory
    if args.sat_type == "ASTER":
        dems_files = exp["raw_data"]["aster_dems"]
        coreg_dir = exp["processed_data"]["aster_dir"]

    elif args.sat_type == "TDX":
        dems_files = exp["raw_data"]["tdx_dems"]
        coreg_dir = exp["processed_data"]["tdx_dir"]
    else:
        raise NotImplementedError

    # -- Select different processing modes -- #
    if args.mode == "median":
        selection_opts = {"mode": "close", "dt": 365, "months": [8, 9, 10]}
        merge_opts = {"mode": "median"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_median")
        downsampling = 1
    elif args.mode == "shean":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "shean"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_shean")
    elif args.mode == "knuth":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "knuth"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_knuth")
    else:
        raise ValueError("`mode` must be either of 'median','shean' or 'knuth'")

    # create output directories
    if not os.path.exists(coreg_dir):
        os.makedirs(coreg_dir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -- Caluclate initial DEM statistics -- #
    print("\n### Calculate initial statistics ###")
    stats_file = os.path.join(coreg_dir, "init_stats.csv")
    init_stats = pproc.calculate_init_stats_parallel(
        dems_files, ref_dem, roi_outlines, all_outlines, stats_file, nthreads=args.nproc, overwrite=args.overwrite
    )
    print(f"Statistics file saved to {stats_file}")

    # -- Select DEMs to be processed -- #
    print("\n### DEMs selection ###")
    validation_dates = exp["validation_dates"]
    groups = utils.dems_selection(dems_files, validation_dates=validation_dates, **selection_opts)

    # -- Postprocess DEMs i.e. coregister, filter etc -- #
    print("\n### Coregister DEMs ###")
    stats, groups_coreg = pproc.postprocessing_all(
        groups,
        ref_dem,
        roi_outlines,
        all_outlines,
        coreg_dir,
        nthreads=args.nproc,
        overwrite=args.overwrite,
        plot=True,
        method="mp",
    )
    print(f"--> Coregistered DEMs saved in {outdir}")

    # Temporarily downsample DEM for speeding-up testing
    if downsampling > 1:
        ref_dem = ref_dem.reproject(dst_res=downsampling * ref_dem.res[0])
        roi_mask = roi_outlines.create_mask(ref_dem)

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")
    ddems = pproc.merge_and_calculate_ddems(
        groups_coreg, validation_dates, ref_dem, outdir=outdir, overwrite=args.overwrite, nproc=args.nproc, **merge_opts
    )

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
    fig_fn = os.path.join(outdir, "ddem_fig.png")
    plt.savefig(fig_fn)
    # plt.show()

    # -- Calculating MB -- #
    print("\n### Calculating mass balance ###")
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        fig_fn = os.path.join(outdir, f"{pair_id}_mb_fig.png")
        ddem_bins, bins_area, frac_obs, dV, dh_mean = mb.mass_balance_local_hypso(
            ddems[pair_id], ref_dem, roi_mask, plot=True, outfig=fig_fn
        )
        dh_mean_err = err.compute_mean_dh_error(ddems[pair_id], ref_dem, stable_mask, roi_mask, nproc=args.nproc)

        print(f"Total volume: {dV:.1f} km3 - mean dh: {dh_mean:.2f} +/- {dh_mean_err:.2f} m")
