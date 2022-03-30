#! /usr/bin/env python

"""
The main script to be run for experiment_1.
"""

import argparse
import multiprocessing as mp
import traceback

from ragmac_xdem import main, io

if __name__ == "__main__":

    # -- Setup script arguments -- #
    parser = argparse.ArgumentParser(description="Process all the data and figures for experiment 1.")

    parser.add_argument(
        "-c",
        dest="case",
        type=str,
        default=None,
        help="str, the case to run 'CH_Aletschgletscher', 'AT_Hintereisferner', or 'NO_Vestisen' (default is all)",
    )
    parser.add_argument(
        "-mode",
        dest="mode",
        type=str,
        default=None,
        help="str, processing mode, either of 'DEMdiff_median', 'DEMdiff_autoselect' or 'TimeSeries' (default is all)",
    )
    parser.add_argument(
        "-run",
        dest="run",
        type=str,
        default="CTL",
        help="str, the run to be processed (default is CTL)",
    )
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
    parser.add_argument(
        "-qc", dest="qc", action="store_true", help="If set, will produce quality control products"
    )
    args = parser.parse_args()

    # List all possible cases, modes and runs to be processed
    all_cases = ["CH_Aletschgletscher", "AT_Hintereisferner", "NO_Vestisen"]
    all_modes = ["DEMdiff_median", "DEMdiff_autoselect", "TimeSeries"]
    run = args.run

    # Select either all or user selected
    if args.case is not None:
        all_cases = [args.case]

    if args.mode is not None:
        all_modes = [args.mode]
    
    if args.qc or args.mode == "TimeSeries3":
        # Launch dask cluster for computation on larger than memory arrays.
        client = io.dask_start_cluster(args.nproc)
        
    nruns = len(all_cases) * len(all_modes)
    print(f"## Total of {nruns} runs to be processed ##")

    # Loop through all combinations
    for case in all_cases:

        for mode in all_modes:

            print(f"\n\n##### Running case {case}, mode {mode} and run {run} #####\n\n")
            try:
                main.main(case, mode, run, sat_type=args.sat_type, nproc=args.nproc, overwrite=args.overwrite, qc=args.qc)
            except:
                print("ERROR -> skipping run")
                traceback.print_exc()
