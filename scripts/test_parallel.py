"""
Test different approaches for multiprocessing and check which one is faster.

Last tested with commit 6d71368
=> mulitpocessing.Pool seems like the best approach. Processing time approximately scales with number of processes.
concurrent on the other hands seems to be faster than single thread only by a few seconds.
See this explanation: https://realpython.com/python-concurrency/

With nproc=3, 6 DEMs
multiprocessing took 68.367840051651 s
concurrent took 166.24858903884888 s
Single thread took 189.5693130493164 s
"""

import os

from time import time

import geoutils as gu
import xdem

import ragmac_xdem.dem_postprocessing as pproc


if __name__ == "__main__":

    nproc = 3

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

    t0 = time()
    print("\nTesting multiprocessing")
    results = pproc.postprocessing_all(
        aster_dems_files[: 2 * nproc],
        ref_dem,
        roi_outlines,
        all_outlines,
        outdir,
        nthreads=nproc,
        overwrite=True,
        method="mp",
    )
    print(f"multiprocessing took {time()-t0:.1f} s")

    t0 = time()
    print("\nTesting concurrent")
    results = pproc.postprocessing_all(
        aster_dems_files[: 2 * nproc],
        ref_dem,
        roi_outlines,
        all_outlines,
        outdir,
        nthreads=nproc,
        overwrite=True,
        method="concurrent",
    )
    print(f"concurrent took {time()-t0:.1f} s")

    t0 = time()
    print("\nTesting single thread")
    results = pproc.postprocessing_all(
        aster_dems_files[: 2 * nproc], ref_dem, roi_outlines, all_outlines, outdir, nthreads=1, overwrite=True
    )
    print(f"Single thread took {time()-t0:.1f} s")
