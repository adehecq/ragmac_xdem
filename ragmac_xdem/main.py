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
import shutil
from pathlib import Path
import xarray as xr
import zarr

import ragmac_xdem.dem_postprocessing as pproc
from ragmac_xdem import files
from ragmac_xdem import mass_balance as mb
from ragmac_xdem import utils
from ragmac_xdem import plotting
from ragmac_xdem import io
from ragmac_xdem import temporal

# Set parameters for the different runs to be processed
default_coreg = xdem.coreg.NuthKaab() + xdem.coreg.Deramp(degree=1) + xdem.coreg.BiasCorr(bias_func=np.nanmedian)

runs = {
    "CTL": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1", "gap_filling": True},
    "NO-CO": {"coreg_method": None, "filtering": True, "coreg_dir": "coreg0_filter1", "gap_filling": True},
    "NO-BIAS": {"coreg_method": xdem.coreg.NuthKaab(), "filtering": True, "coreg_dir": "coreg2_filter1", "gap_filling": True},
    "NO-GAP": {"coreg_method": default_coreg, "filtering": True, "coreg_dir": "coreg1_filter1", "gap_filling": False},
    "NO-FILT": {"coreg_method": default_coreg, "filtering": False, "coreg_dir": "coreg1_filter0", "gap_filling": True},
}


def main(case: dict, mode: str, run_name: str, sat_type: str = "ASTER", nproc: int = None, overwrite: bool = False, qc: bool = False):
    """ 
    Run the processing for a given experiment case, processing mode and run.

    :param case: the case to run e.g. 'AT_Hintereisferner', 'PK_Baltoro' etc
    :param mode: processing mode, either of 'DEMdiff_median', 'DEMdiff_autoselect', 'TimeSeries' or 'TimeSeries2'"
    :param run_name: name of the run to be processed, e.g. 'CTL', 'NO-CO', 'NO-GAP' etc
    :param sat_type: the satellite data to be used, either 'ASTER', 'TDX' or 'both'
    :param nproc: number of processes to be run in parallel whenever possible (Default is max CPU - 1)
    :param overwrite: If set to True, will overwrite already processed data
    :param qc: "If set, will produce quality control products
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
        merge_opts = {"mode": "TimeSeries"}
    elif mode == "TimeSeries2":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "TimeSeries2"}
    elif mode == "TimeSeries3":
        selection_opts = {"mode": "subperiod", "dt": 365}
        downsampling = 1
        merge_opts = {"mode": "TimeSeries3"}
    elif mode == "TimeSeries_full":
        selection_opts = {"mode": "subperiod", "dt": 10958} #~30 years
        downsampling = 1
        merge_opts = {"mode": "TimeSeries3"}
    else:
        raise ValueError("`mode` must be either of 'DEMdiff_autoselect', 'DEMdiff_median', 'TimeSeries', 'TimeSeries2', 'TimeSeries3' or TimeSeries_full")

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
        dems_files, ref_dem.filename, roi_outlines, all_outlines, stats_file, nthreads=nproc, overwrite=overwrite
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
        ref_dem.filename,
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
    
    
    pair_indexes, pair_ids = utils.list_pairs(validation_dates)
    
    if qc:
        print("\n### Plot coregistration QC figures ###")
        for count, pair in enumerate(pair_indexes):

            start = time()

            k1, k2 = pair
            dems_list = groups[count]
            dem_dates = utils.get_dems_date(dems_list)
            dems_coreg_list = groups_coreg[count]
            dem_coreg_dates = utils.get_dems_date(dems_coreg_list)
            pair_id = pair_ids[count]

            outfig = os.path.join(outdir, pair_id+"_coreg_fig.png")


            if not os.path.exists(outfig) or overwrite:
                #TODO create function to handle both coregistered and raw dems
                print('\nPeriod',pair_id)
                
                ### RAW DEMs ### 
                print('\nStacking raw DEMs',pair_id)
                zarr_stack_fn = Path.joinpath(Path(dems_list[0]).parents[0],'stack.zarr')
                zarr_stack_tmp_fn = Path.joinpath(Path(dems_list[0]).parents[0],'stack_tmp.zarr')
                shutil.rmtree(zarr_stack_fn, ignore_errors=True)
                shutil.rmtree(zarr_stack_tmp_fn, ignore_errors=True)

                print('Creating temporary nc files')
                dems_ds = io.xr_stack_geotifs(dems_list,dem_dates,ref_dem.filename, save_to_nc=True)
                nc_files = list(Path(dems_list[0]).parents[0].glob('*.nc'))

                print('Determining optimal chunk size')
                t = len(dems_ds.time)
                x = len(dems_ds.x)
                y = len(dems_ds.y)
                print('data dims: x, y, time')
                print('data shape:',x,y,t)
                arr = dems_ds['band1'].data.rechunk({0:-1, 1:'auto', 2:'auto'}, 
                                                                      block_size_limit=1e8, 
                                                                      balance=True)
                t,y,x = arr.chunks[0][0], arr.chunks[1][0], arr.chunks[2][0]
                tasks_count = io.dask_get_mapped_tasks(dems_ds['band1'].data)
                chunksize = dems_ds['band1'][:t,:y,:x].nbytes / 1048576
                print('chunk shape:', x,y,t)
                print('chunk size:',np.round(chunksize,2), 'MiB')
                print('tasks:', tasks_count)

                print('Creating temporary zarr stack')
                print(str(zarr_stack_tmp_fn))
                dems_ds = xr.open_mfdataset(nc_files,parallel=True)
                dems_ds = dems_ds.drop(['spatial_ref']) 
                dems_ds.to_zarr(zarr_stack_tmp_fn)
                print('Zarr file info')
                source_group = zarr.open(zarr_stack_tmp_fn)
                source_array = source_group['band1']
                print(source_group.tree())
                print(source_array.info)
                del source_group
                del source_array
                
                print('Removing temporary nc files')
                for f in Path(dems_list[0]).parents[0].glob('*.nc'):
                    f.unlink(missing_ok=True)
                
                print('Creating final zarr stack')
                print(str(zarr_stack_fn))
                dems_ds = xr.open_dataset(zarr_stack_tmp_fn,
                                          chunks={'time': t, 'y': y, 'x':x},engine='zarr')
                dems_ds['band1'].encoding = {'chunks': (t, y, x)}
                dems_ds.to_zarr(zarr_stack_fn)
                print('Zarr file info')
                source_group = zarr.open(zarr_stack_fn)
                source_array = source_group['band1']
                print(source_group.tree())
                print(source_array.info)
                del source_group
                del source_array
                
                step = time()  
                print(f"Took {(step-start)/60:.2f} minutes")
                
                print('Removing temporary zarr stack')
                shutil.rmtree(zarr_stack_tmp_fn, ignore_errors=True)

                print('\nComputing NMAD for raw DEMs stack')
                dems_ds = xr.open_dataset(zarr_stack_fn,
                                          chunks={'time': t, 'y': y, 'x':x},engine='zarr')
                nmad_da_before = temporal.xr_dask_nmad(dems_ds)
                start = step
                step = time() 
                print(f"Took {(step-start)/60:.2f} minutes")
                
                print('Removing final zarr stack')
                shutil.rmtree(zarr_stack_fn, ignore_errors=True)

                
                
                ### COREGISTERED DEMs ### 
                print('\nStacking coregistetered DEMs',pair_id)
                zarr_stack_coreg_fn = Path.joinpath(Path(dems_coreg_list[0]).parents[0],'stack.zarr')
                zarr_stack_coreg_tmp_fn = Path.joinpath(Path(dems_coreg_list[0]).parents[0],'stack_tmp.zarr')
                shutil.rmtree(zarr_stack_coreg_fn, ignore_errors=True)
                shutil.rmtree(zarr_stack_coreg_tmp_fn, ignore_errors=True)
                
                print('Creating temporary nc files')
                dems_coreg_ds = io.xr_stack_geotifs(dems_coreg_list,dem_coreg_dates,ref_dem.filename, save_to_nc=True)
                nc_files = list(Path(dems_coreg_list[0]).parents[0].glob('*.nc'))

                print('Determining optimal chunk size')
                t = len(dems_coreg_ds.time)
                x = len(dems_coreg_ds.x)
                y = len(dems_coreg_ds.y)
                print('data dims: x, y, time')
                print('data shape:',x,y,t)
                arr = dems_coreg_ds['band1'].data.rechunk({0:-1, 1:'auto', 2:'auto'}, 
                                                                      block_size_limit=1e8, 
                                                                      balance=True)
                t,y,x = arr.chunks[0][0], arr.chunks[1][0], arr.chunks[2][0]
                tasks_count = io.dask_get_mapped_tasks(dems_coreg_ds['band1'].data)
                chunksize = dems_coreg_ds['band1'][:t,:y,:x].nbytes / 1048576
                print('chunk shape:', x,y,t)
                print('chunk size:',np.round(chunksize,2), 'MiB')
                print('tasks:', tasks_count)
                
                print('Creating temporary zarr stack')
                print(str(zarr_stack_coreg_tmp_fn))
                dems_coreg_ds = dems_coreg_ds.drop(['spatial_ref']) 
                dems_coreg_ds.to_zarr(zarr_stack_coreg_tmp_fn)
                print('Zarr file info')
                source_group = zarr.open(zarr_stack_coreg_tmp_fn)
                source_array = source_group['band1']
                print(source_group.tree())
                print(source_array.info)
                del source_group
                del source_array
                
                print('Removing temporary nc files')
                for f in Path(dems_coreg_list[0]).parents[0].glob('*.nc'):
                    f.unlink(missing_ok=True)
                
                print('Creating final zarr stack')
                print(str(zarr_stack_coreg_fn))
                dems_coreg_ds = xr.open_dataset(zarr_stack_coreg_tmp_fn,
                                          chunks={'time': t, 'y': y, 'x':x},engine='zarr')
                dems_coreg_ds['band1'].encoding = {'chunks': (t, y, x)}
                dems_coreg_ds.to_zarr(zarr_stack_coreg_fn)
                print('Zarr file info')
                source_group = zarr.open(zarr_stack_coreg_fn)
                source_array = source_group['band1']
                print(source_group.tree())
                print(source_array.info)
                del source_group
                del source_array
                
                print('Removing temporary zarr stack')
                shutil.rmtree(zarr_stack_coreg_tmp_fn, ignore_errors=True)

                start = step
                step = time() 
                print(f"Took {(step-start)/60:.2f} minutes")
                
                print('\nComputing count and NMAD for coregistered DEMs stack')
                dems_coreg_ds = xr.open_dataset(zarr_stack_coreg_fn,
                                          chunks={'time': t, 'y': y, 'x':x},engine='zarr')
                count_da = temporal.xr_dask_count(dems_coreg_ds)
                nmad_da_after = temporal.xr_dask_nmad(dems_coreg_ds)

                start = step
                step = time() 
                print(f"Took {(step-start)/60:.2f} minutes")

                outfig = os.path.join(outdir, pair_id+"_coreg_fig.png")
                print('--> Saving plot to ',outfig)
                plotting.xr_plot_count_nmad_before_after_coreg(count_da,
                                                               nmad_da_before, 
                                                               nmad_da_after,
                                                               outfig=outfig)
                print('Removing final zarr stack')
                shutil.rmtree(zarr_stack_coreg_fn, ignore_errors=True)

            else:
                print('--> Plot already exists at',outfig)

    

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
            ddem_filled, ddem_bins, ddem_bins_filled, interp_residuals, frac_obs = mb.fill_ddem_local_hypso(
                ddems[pair_id],
                ref_dem,
                roi_mask,
                roi_outlines,
                filtering=run["filtering"]
            )
            ddems_filled[pair_id] = ddem_filled
        else:
            ddems_filled[pair_id] = ddems[pair_id]

        # -- Calculating MB -- #
        output_mb, ddems_filled_nmad = mb.calculate_mb(ddems_filled[pair_id], 
                                                       roi_outlines, 
                                                       stable_mask,
                                                       ddems[pair_id])
        
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

        # -- Save results to file -- #

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

        # Save results to csv
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

        # Save errors to csv
        ragmac_headers = [
            "glacier_id",
            "run_code",
            "dV_sigma_km3",
            "dh_sigma_km3",
            "S_sigma_km3",
            "voidfill_sigma_km3",
            "temporal_sigma_km3",
        ]
        errors_file = os.path.join(outdir, f"xdem_{case}_{year1}_{year2}_{mode}_errors.csv")

        print(f"Saving errors to file {errors_file}\n")
        output_mb.to_csv(
            errors_file,
            columns=[
                "RGIId",
                "run_code",
                "dV_err",
                "dV_spat_err",
                "dV_area_err",
                "dV_interp_err",
                "dV_temporal_err"
            ],
            index=False,
            header=ragmac_headers,
            na_rep='nan'
        )

    # print time
    t2 = time()
    print(f"Took {(t2-t1)/60:.2f} min to process on {nproc} nodes")
