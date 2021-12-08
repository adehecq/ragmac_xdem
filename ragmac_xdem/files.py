"""Important info on all files to be processed"""

import os
from glob import glob

import numpy as np
import toml

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Import data tree
cfg = toml.load(os.path.join(BASE_DIR, "ragmac_xdem", "data_tree.toml"))

# Input/processed data folders
DATA_RAW_DIR = os.path.join(BASE_DIR, "data/raw/")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/")

# -- Find DEMs for all experiments -- #

# Nested dictionary containing expeiment_number -> case -> path to all needed data sets
experiments = {}
for exp in ["experiment_1", "experiment_2"]:
    experiments[exp] = {}

    for case in cfg[exp]["cases"]:
        experiments[exp][case] = {}
        case_dir = os.path.join(DATA_RAW_DIR, cfg[exp]["folder"], case)
        try:
            for key, value in cfg[case].items():
                experiments[exp][case][key] = os.path.join(case_dir, value)

            # Get all TDX DEM files
            tdx_dems = glob(experiments[exp][case]["tdx_path"])
            experiments[exp][case]["tdx_dems"] = np.sort(tdx_dems)

            # Get all ASTER DEM files
            aster_dems = glob(experiments[exp][case]["aster_path"])
            experiments[exp][case]["aster_dems"] = np.sort(aster_dems)

        except KeyError:
            print(f"Case {case} not implemented")


# def check():
#     """
#     Check that all folders exist
#     """
#     print("Base directory:", BASE_DIR)
#     for region in list(params.keys()):
#         print("*** " + region + " ***")

#         count_errors = 0
#         for item in list(params[region].values()):
#             if os.path.exists(item):
#                 pass
#             else:
#                 print(f"Missing file {item}")
#                 count_errors += 1
#         if count_errors == 0:
#             print("=> ok")


if __name__ == "__main__":
    pass
    # check()
