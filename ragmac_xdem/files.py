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

# -- Find raw DEMs for all experiments -- #

# Nested dictionary containing experiment_number -> case -> path to all needed data sets
experiments = {}
for exp in ["experiment_1", "experiment_2"]:
    experiments[exp] = {}

    # Create a dictionary for each case within experiment
    for case in cfg[exp]["cases"]:
        experiments[exp][case] = {}
        experiments[exp][case]["raw_data"] = {}

        # Save data directory in attributes
        case_dir = os.path.join(DATA_RAW_DIR, cfg[exp]["folder"], case)
        experiments[exp][case]["raw_data"]["directory"] = case_dir

        # Save the different data attributes
        try:
            for key, value in cfg[case].items():
                if key.__contains__("path"):
                    experiments[exp][case]["raw_data"][key] = os.path.join(case_dir, value)
                else:
                    experiments[exp][case][key] = value

            # Get all TDX DEM files
            tdx_dems = glob(experiments[exp][case]["raw_data"]["tdx_path"])
            experiments[exp][case]["raw_data"]["tdx_dems"] = np.sort(tdx_dems)

            # Get all ASTER DEM files
            aster_dems = glob(experiments[exp][case]["raw_data"]["aster_path"])
            experiments[exp][case]["raw_data"]["aster_dems"] = np.sort(aster_dems)

        except KeyError:
            print(f"Case {case} not implemented")


# -- Path to all processed DEMs -- #

for exp in ["experiment_1", "experiment_2"]:

    # Create a dictionary for processed_data
    for case in cfg[exp]["cases"]:
        experiments[exp][case]["processed_data"] = {}

        case_dir = os.path.join(DATA_PROCESSED_DIR, cfg[exp]["folder"], case)
        experiments[exp][case]["processed_data"]["directory"] = case_dir

        experiments[exp][case]["processed_data"]["tdx_dir"] = os.path.join(case_dir, "TDX_DEMs")
        experiments[exp][case]["processed_data"]["aster_dir"] = os.path.join(case_dir, "ASTER_DEMs")


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
