#! /usr/bin/env python

"""
The main function to run the processing for a given experiment, case and run.
"""

import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xdem
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ragmac_xdem.dem_postprocessing as pproc
from ragmac_xdem import files
from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils

# Set parameters for the different runs to be processed
default_coreg = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian)

runs = {
    "CTL": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1"},
    "NO-CO": {"coreg_method": None, "filtering": True, "coreg_dir": "coreg0_filter1"},
    "NO-BIAS": {"coreg_method": xdem.coreg.NuthKaab(), "filtering": True, "coreg_dir": "coreg2_filter1"},
    "NO-GAP": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1"},
    "NO-FILT": {"coreg_method": default_coreg, "filtering": False, "coreg_dir": "coreg1_filter0"},
}


def main(case: dict, mode: str, run_name: str, sat_type: str = "ASTER", nproc: int = None, overwrite: bool = False):
    """ 
    Run the processing for a given experiment case, processing mode and run.

    :param case: the case to run e.g. 'AT_Hintereisferner', 'PK_Baltoro' etc
    :param mode: processing mode, either of 'DEMdiff_median', 'DEMdiff_autoselect', 'TimeSeries' or 'TimeSeries2'"
    :param run_name: name of the run to be processed, e.g. 'CTL', 'NO-CO', 'NO-GAP' etc
    :param sat_type: the satellite data to be used, either 'ASTER', 'TDX' or 'both'
    :param nproc: number of processes to be run in parallel whenever possible (Default is max CPU - 1)
    :param overwrite: If set to True, will overwrite already processed data
    """
    # Record time
    t1 = time()

    # -- Load input data -- #
    exp = files.get_data_paths(case)
    ref_dem, all_outlines, roi_outlines, roi_mask, stable_mask = utils.load_ref_and_masks(exp)

    # Get list of all DEMs and set output directory
    if sat_type == "ASTER":
        dems_files = exp["raw_data"]["aster_dems"]
    elif sat_type == "TDX":
        dems_files = exp["raw_data"]["tdx_dems"]
    else:
        raise NotImplementedError

    # -- Select different processing modes -- #
    if mode == "DEMdiff_median":
        selection_opts = {"mode": "close", "dt": 400, "months": [8, 9, 10]}
        merge_opts = {"mode": "median"}
        downsampling = 1
    elif mode == "DEMdiff_autoselect":
        selection_opts = {"mode": "best", "dt": 400, "months": [8, 9, 10]}
        merge_opts = {"mode": "median"}
        downsampling = 1
    elif mode == "TimeSeries":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "shean"}
    elif mode == "TimeSeries2":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "knuth"}
    else:
        raise ValueError("`mode` must be either of 'median', 'shean' or knuth'")

    # Get run parameters
    run = runs[run_name]

    # Create output directories
    process_dir = exp["processed_data"]["directory"]
    coreg_dir = os.path.join(process_dir, run["coreg_dir"])
    outdir = os.path.join(process_dir, f"results_{run_name}_{mode}")

    if not os.path.exists(coreg_dir):
        os.makedirs(coreg_dir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -- Calculate initial DEM statistics -- #
    print("\n### Calculate initial statistics ###")
    stats_file = os.path.join(process_dir, "init_stats.csv")
    init_stats = pproc.calculate_init_stats_parallel(
        dems_files, ref_dem, roi_outlines, all_outlines, stats_file, nthreads=nproc, overwrite=overwrite
    )
    print(f"Statistics file saved to {stats_file}")

    # -- Select DEMs to be processed -- #
    print("\n### DEMs selection ###")
    validation_dates = exp["validation_dates"]
    if selection_opts["mode"] == "best":
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
        coreg_method=run["coreg_method"],
        filtering=run["filtering"],
        nthreads=nproc,
        overwrite=overwrite,
        plot=True,
        mp_method="mp",
    )
    print(f"--> Coregistered DEMs saved in {coreg_dir}")

    # Temporarily downsample DEM for speeding-up process for testing
    if downsampling > 1:
        ref_dem = ref_dem.reproject(dst_res=downsampling * ref_dem.res[0])
        roi_mask = roi_outlines.create_mask(ref_dem)

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")

    ddems = pproc.merge_and_calculate_ddems(
        groups_coreg, validation_dates, ref_dem, outdir=outdir, overwrite=overwrite, nproc=nproc, **merge_opts
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
        ddems[pair_id].show(ax=ax, cmap="coolwarm_r", vmin=-50, vmax=50, add_cb=False, zorder=1)
        ax.set_title(pair_id)

        # adjust cbar to match plot extent
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cmap = plt.cm.get_cmap("coolwarm_r")
        norm = matplotlib.colors.Normalize(vmin=-50, vmax=50)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.set_label(label="Elevation change (m)")

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
            ddems[pair_id], ref_dem, roi_mask, roi_outlines, plot=True, outfig=fig_fn
        )
        ddems_filled[pair_id] = ddem_filled

    # -- Calculating MB -- #
    print("\n### Calculating mass balance ###")
    for k, pair_id in enumerate(ddems):

        print(pair_id)
        output_mb = mb.calculate_mb(ddems_filled[pair_id], roi_outlines, stable_mask)

        # Print to screen the results for largest glacier
        largest = output_mb.sort_values(by="area").iloc[-1]
        print(
            f"Glacier {largest.RGIId} - Volume change: {largest.dV:.2f} +/- {largest.dV_err:.2f} km3 - mean dh: {largest.dh_mean:.2f} +/- {largest.dh_mean_err:.2f} m"
        )

        # Add other inputs necessary for RAGMAC report
        output_mb["run_code"] = np.array(["CTL"], dtype="U4").repeat(len(output_mb))
        output_mb["method"] = np.array(
            [
                mode,
            ],
            dtype="U10",
        ).repeat(len(output_mb))

        start_date_str = start_date[pair_id].strftime("%Y-%m-%d")
        end_date_str = end_date[pair_id].strftime("%Y-%m-%d")
        output_mb["start_date"] = start_date_str
        output_mb["end_date"] = end_date_str

        # Save to csv
        ragmac_headers = [
            "glacier_id",
            "run_code",
            "S_km2",
            "start_date_yyyy-mm-dd",
            "end_date_yyyy-mm-dd",
            "method",
            "dh_m",
            "dh_sigma_m",
            "dV_km3",
            "dV_sigma_km3",
        ]  # , "remarks"]
        year1, year2 = pair_id.split("_")
        results_file = os.path.join(outdir, f"xdem_{case}_{year1}_{year2}_{mode}_results.csv")

        print(f"Saving results to file {results_file}\n")
        output_mb.to_csv(
            results_file,
            columns=[
                "RGIId",
                "run_code",
                "area",
                "start_date",
                "end_date",
                "method",
                "dh_mean",
                "dh_mean_err",
                "dV",
                "dV_err",
            ],
            index=False,
            header=ragmac_headers,
        )

    # print time
    t2 = time()
    print(f"Took {(t2-t1)/60 min to process on {nproc} nodes")
