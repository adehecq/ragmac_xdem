"""
The main script to be run for experiment_2
"""

import multiprocessing as mp
import os

import geoutils as gu
import numpy as np
import xdem
import matplotlib.pyplot as plt
import argparse

import ragmac_xdem.dem_postprocessing as pproc
from ragmac_xdem import utils
from ragmac_xdem import mass_balance as mb

if __name__ == "__main__":

    # -- Setup script arguments -- #
    parser = argparse.ArgumentParser(description="Process all the data and figures for experiment 2")

    parser.add_argument('-sat', dest='sat_type', type=str, default="ASTER", help="str, the satellite data to be used, either 'ASTER', 'TDX' or 'both'")
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', help="If set, will overwrite already processed data")
    parser.add_argument('-nproc', dest='nproc', type=int, default= mp.cpu_count() - 1, help='int, number of processes to be run in parallel whenever possible (Default is max CPU - 1)')

    args = parser.parse_args()

    # -- Load input data -- #
    from ragmac_xdem import files

    baltoro_exp = files.experiments["experiment_2"]["PK_Baltoro"]

    # Load reference DEM
    ref_dem = xdem.DEM(baltoro_exp["raw_data"]["ref_dem_path"])

    # Load all outlines
    all_outlines = gu.geovector.Vector(baltoro_exp["raw_data"]["rgi_path"])

    # Load selected glacier outline
    roi_outlines = gu.geovector.Vector(baltoro_exp["raw_data"]["selected_path"])

    # Create masks
    roi_mask = roi_outlines.create_mask(ref_dem)
    stable_mask = ~all_outlines.create_mask(ref_dem)

    # Get list of all DEMs and set output directory
    if args.sat_type == "ASTER":
        dems_files = baltoro_exp["raw_data"]["aster_dems"]
        outdir = baltoro_exp["processed_data"]["aster_dir"]

    elif args.sat_type == "TDX":
        dems_files = baltoro_exp["raw_data"]["tdx_dems"]
        outdir = baltoro_exp["processed_data"]["tdx_dir"]
    else:
        raise NotImplementedError

    # Create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -- Select DEMs to be processed -- #
    inds_2000 = utils.select_dems_by_date(dems_files, "2000-01-01", "2000-12-31", sat_type=args.sat_type)
    inds_2012 = utils.select_dems_by_date(dems_files, "2012-01-01", "2012-12-31", sat_type=args.sat_type)
    inds_2019 = utils.select_dems_by_date(dems_files, "2019-01-01", "2019-12-31", sat_type=args.sat_type)
    inds_all = np.asarray([*inds_2000, *inds_2012, *inds_2019])

    # Update list of DEMs and indices
    dems_files = dems_files[inds_all]
    if args.sat_type == "ASTER":
        inds_2000 = inds_2000 - inds_2000[0]
        inds_2012 = inds_2012 - inds_2012[0] + inds_2000[-1] + 1
        inds_2019 = inds_2019 - inds_2019[0] + inds_2012[-1] + 1
    elif args.sat_type == "TDX":
        inds_2000 = []
        inds_2012 = inds_2012 - inds_2012[0]
        inds_2019 = inds_2019 - inds_2019[0] + inds_2012[-1] + 1

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
    print(f"--> Coregistered DEMs saved in {outdir}")

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")

    # 2000
    if args.sat_type == "ASTER":
        dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in coreg_dems_files[inds_2000]]
        mosaic_2000 = gu.spatial_tools.merge_rasters(
            dem_objs, reference=ref_dem, merge_algorithm=np.nanmedian, use_ref_bounds=True
        )
        ddem_2000 = ref_dem - mosaic_2000
        cov_2000, _, med_2000, nmad_2000 = pproc.calculate_stats(ddem_2000, roi_mask, stable_mask)

    # 2012
    dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in coreg_dems_files[inds_2012]]
    mosaic_2012 = gu.spatial_tools.merge_rasters(
        dem_objs, reference=ref_dem, merge_algorithm=np.nanmedian, use_ref_bounds=True
    )
    ddem_2012 = ref_dem - mosaic_2012
    cov_2012, _, med_2012, nmad_2012 = pproc.calculate_stats(ddem_2012, roi_mask, stable_mask)

    # 2019
    dem_objs = [xdem.DEM(dem_path, load_data=False) for dem_path in coreg_dems_files[inds_2019]]
    mosaic_2019 = gu.spatial_tools.merge_rasters(
        dem_objs, reference=ref_dem, merge_algorithm=np.nanmedian, use_ref_bounds=True
    )
    ddem_2019 = ref_dem - mosaic_2019
    cov_2019, _, med_2019, nmad_2019 = pproc.calculate_stats(ddem_2019, roi_mask, stable_mask)

    # -- Calculate elevation change for all periods -- #
    if args.sat_type == "ASTER":
        ddem_2000_2012 = mosaic_2012 - mosaic_2000
    ddem_2012_2019 = mosaic_2019 - mosaic_2012

    plt.figure(figsize=(18, 8))
    ax1 = plt.subplot(121)
    if args.sat_type == "ASTER":
        roi_outlines.ds.plot(ax=ax1, facecolor='none', edgecolor='k', zorder=2)
        ddem_2000_2012.show(ax=ax1, cmap='coolwarm_r', vmin=-50, vmax=50, cb_title="Elevation change (m)", zorder=1)
        ax1.set_title("2000 - 2012")

    ax2 = plt.subplot(122)
    roi_outlines.ds.plot(ax=ax2, facecolor='none', edgecolor='k', zorder=2)
    ddem_2012_2019.show(ax=ax2, cmap='coolwarm_r', vmin=-50, vmax=50, cb_title="Elevation change (m)", zorder=1)
    ax2.set_title("2012 - 2019")

    plt.tight_layout()
    plt.show()

    # Calculating MB
    if args.sat_type == "ASTER":
        print("\n### Mass balance 2000 - 2012 ###")
        ddem_bins, bins_area, frac_obs, dV, dh_mean = mb.mass_balance_local_hypso(ddem_2000_2012, ref_dem, roi_mask, plot=True)
        print(f"Total volume: {dV:.1f} km3 - mean dh: {dh_mean:.2f} m")

    print("\n### Mass balance 2012 - 2019 ###")
    ddem_bins, bins_area, frac_obs, dV, dh_mean = mb.mass_balance_local_hypso(ddem_2012_2019, ref_dem, roi_mask, plot=True)
    print(f"Total volume: {dV:.1f} km3 - mean dh: {dh_mean:.2f} m")
