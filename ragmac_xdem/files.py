"""Important info on all files to be processed"""

import os

from glob import glob

import numpy as np
import toml
import pandas as pd


# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Import data tree
cfg = toml.load(os.path.join(BASE_DIR, "ragmac_xdem", "data_tree.toml"))

# Input/processed data folders
DATA_RAW_DIR = os.path.join(BASE_DIR, "data/raw/")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/")


def get_data_paths(case: str) -> dict:
    """
    Returns a dictionary containing the path to all data needed for a given experiment/case.
    """
    # Find in which experiment it belongs
    if case in cfg["experiment_1"]["cases"]:
        exp = "experiment_1"
    elif case in cfg["experiment_2"]["cases"]:
        exp = "experiment_2"
    else:
        all_cases = cfg['experiment_1']["cases"] + cfg['experiment_2']["cases"]
        raise ValueError(f"Case {case} not found, nust be in {all_cases}")

    # Nested dictionary containing experiment_number -> case -> path to all needed data sets
    case_paths = {}
    case_paths["raw_data"] = {}

    # Save data directory in attributes
    case_dir = os.path.join(DATA_RAW_DIR, cfg[exp]["folder"], case)
    case_paths["raw_data"]["directory"] = case_dir

    # Save the different data attributes
    try:
        for key, value in cfg[case].items():
            if key.__contains__("path"):
                case_paths["raw_data"][key] = os.path.join(case_dir, value)
            else:
                case_paths[key] = value

        # Get all TDX DEM files
        tdx_dems = glob(case_paths["raw_data"]["tdx_path"])
        case_paths["raw_data"]["tdx_dems"] = np.sort(tdx_dems)

        # Get all ASTER DEM files
        aster_dems = glob(case_paths["raw_data"]["aster_path"])
        case_paths["raw_data"]["aster_dems"] = np.sort(aster_dems)

    except KeyError:
        print(f"Case {case} not implemented")

    # Create a dictionary for processed_data
    case_paths["processed_data"] = {}

    case_dir = os.path.join(DATA_PROCESSED_DIR, cfg[exp]["folder"], case)
    case_paths["processed_data"]["directory"] = case_dir

    case_paths["processed_data"]["tdx_dir"] = os.path.join(case_dir, "TDX_DEMs")
    case_paths["processed_data"]["aster_dir"] = os.path.join(case_dir, "ASTER_DEMs")

    return case_paths


def load_mb_series(region: str) -> pd.Series:
    """
    Load the MB series from WGMS, for a given region.
    """
    mb_file = os.path.join(BASE_DIR, 'data', 'raw', 'regional_mb_series', 'regional_adhoc_estimates_BA_anom_mwe.csv')

    # Read CSV, convert year string to integer and use region as index
    mb_series = pd.read_csv(mb_file,
                            names=["Region", ] + list(np.arange(2000, 2021)),
                            header=0,
                            index_col='Region'
                            )

    if region not in mb_series.index:
        raise ValueError(f"`region` must be in {list(mb_series.index)}")

    return mb_series.loc[region]

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
