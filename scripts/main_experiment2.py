"""
The main script to be run for experiment_2
"""

import os

import geoutils as gu
import xdem

import ragmac_xdem.dem_postprocessing as pproc

from time import time

if __name__ == "__main__":
    
    # -- Load input data -- #
    from ragmac_xdem import files
    baltoro_exp = files.experiments["experiment_2"]["PK_Baltoro"]

    # Load reference DEM
    ref_dem = xdem.DEM(baltoro_exp["raw_data"]["ref_dem_path"])

    # Load all outlines
    all_outlines = gu.geovector.Vector(baltoro_exp["raw_data"]["rgi_path"])

    # Load selected glacier outline
    roi_outlines = gu.geovector.Vector(baltoro_exp["raw_data"]["selected_path"])

    # Get list of all DEMs
    tdx_dems_files = baltoro_exp["raw_data"]["tdx_dems"]
    aster_dems_files = baltoro_exp["raw_data"]["aster_dems"]

    # Output directory
    outdir = baltoro_exp["processed_data"]["aster_dir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # dem_path = aster_dems_files[1]
    # out_dem_path = os.path.join(outdir, os.path.basename(dem_path).replace(".tif", "_coreg.tif"))
    # out_fig = out_dem_path.replace(".tif", "_diff.png")
    # pproc.postprocessing(dem_path, ref_dem, roi_outlines, all_outlines, out_dem_path, plot=True, out_fig=out_fig)

    t0 = time()
    results = pproc.postprocessing_all(aster_dems_files[:6], ref_dem, roi_outlines, all_outlines, outdir, nthreads=3, overwrite=True, method='mp')
    t1 = time()
    print(f"multiprocessing took {t1-t0}s")

    t0 = time()
    results = pproc.postprocessing_all(aster_dems_files[:6], ref_dem, roi_outlines, all_outlines, outdir, nthreads=3, overwrite=True, method='concurrent')
    t1 = time()
    print(f"concurrent took {t1-t0}s")

    t0 = time()
    results = pproc.postprocessing_all(aster_dems_files[:6], ref_dem, roi_outlines, all_outlines, outdir, nthreads=1, overwrite=True)
    t1 = time()
    print(f"Single thread took {t1-t0}s")
