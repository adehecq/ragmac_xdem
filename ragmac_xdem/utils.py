"""Various utilites"""

import os
from datetime import datetime, timedelta

import numpy as np


def get_aster_date(fname):
    """Parse the date of an ASTER DEM from the filename"""
    # Extract string containing decimal year
    basename = os.path.basename(fname)
    year_decimal = float(basename[4:17])

    # Get integer year and decimals
    year = int(np.trunc(year_decimal))
    decimals = year_decimal - year

    # Convert to date and time
    base = datetime(year, 1, 1)
    ndays = base.replace(year=base.year + 1) - base
    return base + timedelta(seconds=ndays.total_seconds() * decimals)


def get_tdx_date(fname):
    """Parse the date of a TDX DEM from the filename"""
    # Extract string containing date and time
    basename = os.path.basename(fname)
    datetime_str = basename[:17]

    # Convert to datetime
    return datetime.strptime(datetime_str,'%Y-%m-%d_%H%M%S')


def select_dems_by_date(dem_path_list: list[str], date1: str, date2: str, sat_type: str) -> list:
    """
    Returns the list of files which date falls within date1 and date 2 (included)

    :param dem_path_list: List of path to DEMs
    :param date1: Start date in ISO format YYYY-MM-DD
    :param date1: End date in ISO format YYYY-MM-DD
    :param sat_type: Either 'ASTER' or 'TDX'

    :returns: The list of indexes that match the criteria
    """
    if sat_type == "ASTER":
        dates = np.asarray([get_aster_date(dem_file) for dem_file in dem_path_list])
    elif sat_type == "TDX":
        dates = np.asarray([get_tdx_date(dem_file) for dem_file in dem_path_list])
    else:
        raise ValueError("sat_type must be 'ASTER' or 'TDX'")

    date1 = datetime.fromisoformat(date1)
    date2 = datetime.fromisoformat(date2)
    return np.where((date1 <= dates) & (dates <= date2))[0]
