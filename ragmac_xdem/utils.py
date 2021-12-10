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
