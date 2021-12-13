"""
The main script to be run for experiment_2
"""

import multiprocessing as mp
import os

import geoutils as gu
import numpy as np
import xdem

import ragmac_xdem.dem_postprocessing as pproc
from ragmac_xdem import utils

if __name__ == "__main__":

    nproc = mp.cpu_count() - 1

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

    # Get list of all DEMs
    tdx_dems_files = baltoro_exp["raw_data"]["tdx_dems"]
    aster_dems_files = baltoro_exp["raw_data"]["aster_dems"]

    # Output directory
    outdir = baltoro_exp["processed_data"]["aster_dir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # -- Select DEMs to be processed -- #
    inds_2000 = utils.select_dems_by_date(aster_dems_files, "2000-01-01", "2000-12-31", sat_type="ASTER")
    inds_2012 = utils.select_dems_by_date(aster_dems_files, "2012-01-01", "2012-12-31", sat_type="ASTER")
    inds_2019 = utils.select_dems_by_date(aster_dems_files, "2019-01-01", "2019-12-31", sat_type="ASTER")
    inds_all = np.asarray([*inds_2000, *inds_2012, *inds_2019])

    # Update list of DEMs and indices
    aster_dems_files = aster_dems_files[inds_all]
    inds_2000 = inds_2000 - inds_2000[0]
    inds_2012 = inds_2012 - inds_2012[0] + inds_2000[-1] + 1
    inds_2019 = inds_2019 - inds_2019[0] + inds_2012[-1] + 1

    # -- Postprocess DEMs i.e. coregister, filter etc -- #
    print("\n### Coregister DEMs ###")
    stats = pproc.postprocessing_all(
        aster_dems_files,
        ref_dem,
        roi_outlines,
        all_outlines,
        outdir,
        nthreads=nproc,
        overwrite=False,
        plot=True,
        method="mp",
    )
    coreg_dems_files = np.asarray(stats["coreg_path"])

    # -- Merge DEMs by period -- #
    print("\n### Merge DEMs ###")

    # 2000
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
