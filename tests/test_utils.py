"""
Test suite for utils.py
"""
from datetime import datetime

from ragmac_xdem import utils


def test_decyear_conversions():
    """
    Test that functions `date_time_to_decyear` and `decyear_to_date_time` work reversely as expected,
    for a range of years and months.
    """
    for year in range(2000, 2022):
        for month in range(1, 13):
            for day in range(1, 31):

                try:
                    date_time = datetime(year, month, day)
                except ValueError:  # if days is outside of month's range
                    continue

                # convert to decimal year and back
                decimal_date = utils.date_time_to_decyear(date_time)
                date_time_new = utils.decyear_to_date_time(decimal_date)

                # Due to rounding errors, dates may differ within 4 ms
                ddate = abs(date_time - date_time_new)
                assert (ddate.days == 0) & (ddate.seconds == 0) & (ddate.microseconds <= 4), f"Issue with date {date_time}"


def test_aster_date():
    """
    Test that parsing ASTER date from filename works as intended.
    Works except for dates very close to start/end of the year.
    """
    for decyear in [2000.1, 2002.37, 2012.97]:
        fname = f"DEM_{decyear:.8f}.tif"
        date_time = utils.get_aster_date(fname)
        new_decyear = utils.fannys_convert_date_time_to_decimal_date(date_time)
        assert decyear == new_decyear
