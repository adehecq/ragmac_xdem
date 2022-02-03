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
        method = "DEMdiff_median"
    elif args.mode == "best":
        selection_opts = {"mode": "best", "dt": 400, "months": [8, 9, 10]}
        merge_opts = {"mode": "median"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_best")
        downsampling = 1
        method = "DEMdiff_best"
    elif args.mode == "shean":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "shean"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_shean")
        method = "TimeSeries"
    elif args.mode == "knuth":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "knuth"}
        outdir = os.path.join(exp["processed_data"]["directory"], "results_knuth")
        method = "TimeSeries2"
    else:
        raise ValueError("`mode` must be either of 'median', 'best', 'shean' or 'knuth'")

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
    if selection_opts["mode"] == 'best':
        selection_opts["init_stats"] = init_stats
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
    print(f"--> Coregistered DEMs saved in {coreg_dir}")

    # Temporarily downsample DEM for speeding-up testing
    if downsampling > 1:
        ref_dem = ref_dem.reproject(dst_res=downsampling * ref_dem.res[0])
        roi_mask = roi_outlines.create_mask(ref_dem)

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")
    ddems = pproc.merge_and_calculate_ddems(
        groups_coreg, validation_dates, ref_dem, outdir=outdir, overwrite=args.overwrite, nproc=args.nproc, **merge_opts
    )

    # Get first/last dates of the subperiods - needed for final report
    start_date, end_date = utils.get_start_end_dates(groups, merge_opts["mode"], validation_dates)

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

    # -- Generate gap-free mosaics -- #
    print("\n### Interpolate data gaps ###")
    ddems_filled = {}
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        fig_fn = os.path.join(outdir, f"{pair_id}_mb_fig.png")
        ddem_filled, ddem_bins = mb.fill_ddem_local_hypso(
            ddems[pair_id], ref_dem, roi_mask, plot=True, outfig=fig_fn
        )
        ddems_filled[pair_id] = ddem_filled

    # -- Calculating MB -- #
    print("\n### Calculating mass balance ###")
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        output_mb = mb.calculate_mb(ddems_filled[pair_id], roi_outlines, stable_mask)

        # Print to screen the results for largest glacier
        largest = output_mb.sort_values(by="area").iloc[-1]
        print(f"Glacier {largest.RGIId} - Volume change: {largest.dV:.2f} +/- {largest.dV_err:.2f} km3 - mean dh: {largest.dh_mean:.2f} +/- {largest.dh_mean_err:.2f} m")

        # Add other inputs necessary for RAGMAC report
        output_mb["run_code"] = np.array(["CLT"], dtype='U4').repeat(len(output_mb))
        output_mb["method"] = np.array([method,], dtype='U10').repeat(len(output_mb))

        start_date_str = start_date[pair_id].strftime("%Y-%m-%d")
        end_date_str = end_date[pair_id].strftime("%Y-%m-%d")
        output_mb["start_date"] = start_date_str
        output_mb["end_date"] = end_date_str

        # Save to csv
        ragmac_headers = ["glacier_id", "run_code", "S_km2", "start_date_yyyy-mm-dd", "end_date_yyyy-mm-dd", "method", "dh_m", "dh_sigma_m", "dV_km3", "dV_sigma_km3"]  # , "remarks"]
        year1, year2 = pair_id.split("_")
        results_file = os.path.join(outdir, f"xdem_PK_Baltoro_{year1}_{year2}_{method}_results.csv")

        print(f"Saving results to file {results_file}\n")
        output_mb.to_csv(
            results_file,
            columns=["RGIId", "run_code", "area", "start_date", "end_date", "method",
                     "dh_mean", "dh_mean_err", "dV", "dV_err"],
            index=False,
            header=ragmac_headers
        )
