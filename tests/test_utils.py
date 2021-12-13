"""
Test suite for utils.py
"""
from datetime import datetime

from ragmac_xdem import utils


def test_aster_date():
    """
    Test that parsing ASTER date from filename works as intended.
    """
    assert utils.get_aster_date("DEM_2000.00000000.tif") == datetime(2000, 1, 1, 0, 0)
    assert utils.get_aster_date("DEM_2001.50000000.tif") == datetime(2001, 7, 2, 12, 0)
    assert utils.get_aster_date("DEM_2019.99999999.tif") == datetime(2019, 12, 31, 23, 59, 59, 684643)
