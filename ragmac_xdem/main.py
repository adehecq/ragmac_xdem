#! /usr/bin/env python

"""
The main function to run the processing for a given experiment, case and run.
"""

import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import geoutils as gu
import xdem
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ragmac_xdem.dem_postprocessing as pproc
from ragmac_xdem import files
from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils
from ragmac_xdem import plotting

# Set parameters for the different runs to be processed
default_coreg = xdem.coreg.NuthKaab() + xdem.coreg.BiasCorr(bias_func=np.nanmedian)

runs = {
    "CTL": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1", "gap_filling": True},
    "NO-CO": {"coreg_method": None, "filtering": True, "coreg_dir": "coreg0_filter1", "gap_filling": True},
    "NO-BIAS": {"coreg_method": xdem.coreg.NuthKaab(), "filtering": True, "coreg_dir": "coreg2_filter1", "gap_filling": True},
    "NO-GAP": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1", "gap_filling": False},
    "NO-FILT": {"coreg_method": default_coreg, "filtering": False, "coreg_dir": "coreg1_filter0", "gap_filling": True},
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
        merge_opts = {"mode": "TimeSeries2"}
    elif mode == "TimeSeries3":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "TimeSeries3"}
    else:
        raise ValueError("`mode` must be either of 'DEMdiff_autoselect', 'DEMdiff_median', 'TimeSeries', 'TimeSeries2' or 'TimeSeries3'")

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
    # if selection_opts["mode"] == "best":
    #     selection_opts["init_stats"] = init_stats
    groups = utils.dems_selection(dems_files, init_stats, validation_dates=validation_dates, **selection_opts)

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
        ddems[pair_id].show(ax=ax, cmap="coolwarm_r", vmin=-30, vmax=30, add_cb=False, zorder=1)
        ax.set_title(pair_id)

        # adjust cbar to match plot extent
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cmap = plt.cm.get_cmap("coolwarm_r")
        norm = matplotlib.colors.Normalize(vmin=-30, vmax=30)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.set_label(label="Elevation change (m)")
        
        #TODO add stable ground mask to this or seperate qc plot
#         if plot_stable_mask:
#             tmp = ddems[pair_id].copy()
#             tmp.data[0] = stable_mask[0] # replace data array in container
#             tmp.data[tmp.data==0.0]=np.nan
#             tmp.show(ax=ax2, cmap="gray", vmin=1, alpha=1, add_cb=False, zorder=2)

    plt.tight_layout()
    fig_fn = os.path.join(outdir, "ddem_fig.png")
    plt.savefig(fig_fn)
    # plt.show()

    # -- Generate gap-free mosaics -- #
    if run["gap_filling"]:
        print("\n### Interpolate data gaps and calculate mass balance ###")
    else:
        print("\n### Calculate mass balance ###")
    
    ddems_filled = {}
    for k, pair_id in enumerate(ddems):
        print(pair_id)

        # -- Interpolate -- #
        
        if run["gap_filling"]:
            ddem_filled, ddem_bins, ddem_bins_filled = mb.fill_ddem_local_hypso(ddems[pair_id], 
                                                                                  ref_dem, 
                                                                                  roi_mask, 
                                                                                  roi_outlines, 
                                                                                  filtering=run["filtering"])
            ddems_filled[pair_id] = ddem_filled
        else:
            ddems_filled[pair_id] = ddems[pair_id]
        
        # -- Calculating MB -- #
        output_mb, ddems_filled_nmad = mb.calculate_mb(ddems_filled[pair_id], 
                                                       roi_outlines, 
                                                       stable_mask)
        
        # Plot
        if run["gap_filling"]:
            fig_fn = os.path.join(outdir, f"{pair_id}_mb_fig.png")

            bins_area = xdem.volume.calculate_hypsometry_area(ddem_bins, 
                                                              ref_dem.data[roi_mask], 
                                                              pixel_size=ref_dem.res)
            bin_width = ddem_bins.index.left - ddem_bins.index.right
            obs_area = ddem_bins["count"] * ref_dem.res[0] * ref_dem.res[1]
            frac_obs = obs_area / bins_area

            dh_mean = np.nanmean(ddems[pair_id].data[roi_mask])
            data, mask = gu.spatial_tools.get_array_and_mask(ddems[pair_id])
            nobs = np.sum(~mask[roi_mask.squeeze()])
            ntot = np.sum(roi_mask)
            ddems_roi_coverage = nobs / ntot

            ddems_filled_dh_mean = output_mb['dh_mean'].values * \
                                   output_mb['area'].values
            ddems_filled_dh_mean = np.sum(ddems_filled_dh_mean) / \
                                   np.sum(output_mb['area'].values)
            
            ddems_filled_dh_mean_err = output_mb['dh_mean_err'].values * \
                                       output_mb['area'].values
            ddems_filled_dh_mean_err = np.sum(ddems_filled_dh_mean_err) / \
                                       np.sum(output_mb['area'].values)
            
            run_id = ' '.join([case,mode,run_name])

            plotting.plot_mb_fig(# hyps curve params
                                 ddem_bins, 
                                 ddem_bins_filled, 
                                 bins_area,
                                 bin_width,
                                 frac_obs,
                                 # stats to annotate plot with
                                 pair_id,
                                 run_id,
                                 ddems_roi_coverage,
                                 ddems_filled_dh_mean,
                                 ddems_filled_dh_mean_err,
                                 ddems_filled_nmad,
                                 # dems to plot
                                 roi_outlines,
                                 ddems[pair_id],
                                 ddem_filled,
                                 # plotting params
                                 outfig=fig_fn,
                                 bin_alpha=0.3,
                                 line_width=3,
                                 dh_spread_map=30,
                                 dh_spread_curve=40)


        # Print to screen the results for largest glacier
        largest = output_mb.sort_values(by="area").iloc[-1]
        print(
            f"Glacier {largest.RGIId} - Volume change: {largest.dV:.2f} +/- {largest.dV_err:.2f} km3 - mean dh: {largest.dh_mean:.2f} +/- {largest.dh_mean_err:.2f} m"
        )

        # Add other inputs necessary for RAGMAC report
        output_mb["run_code"] = np.array([run_name], dtype="U10").repeat(len(output_mb))
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
            na_rep='nan'
        )

    # print time
    t2 = time()
    print(f"Took {(t2-t1)/60:.2f} min to process on {nproc} nodes")
