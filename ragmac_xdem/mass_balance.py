"""
Functions to calculate a glacier mass balance.
"""
import warnings

import geoutils as gu
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import xdem
from geoutils.georaster import RasterType
from scipy.interpolate import interp1d
from tqdm import tqdm
from ragmac_xdem import uncertainty as err
from ragmac_xdem import plotting


def ddem_bins_filtering(
    ddem_bins: pd.DataFrame, count_threshold: int = 10, nmad_fact: int = 5, verbose: bool = True
) -> pd.DataFrame:
    """
    Filter altitudinally averaged ddem bins (derived from xdem.volume.hypsometric_binning) based on count and distance to the median.

    :param ddem_bins: The ddem averaged by altitude bands.
    :param count_threshold: Minimum count in bin to be considered inlier. If <= 1, will be ignored.
    :param nmad_fact: Bins distant from the median of all bins by this factor times the NMAD of all bins will be discarded. If < 0, will be ignored.
    :param verbose: set to True to print some results to screen

    :returns: the filtered ddem_bins
    """
    # Copy to new DataFrame
    ddem_bins_filtered = ddem_bins.copy()

    # Exclude bins with low count
    if count_threshold > 1:
        outliers = ddem_bins_filtered["count"] < count_threshold
        ddem_bins_filtered.loc[outliers, "value"] = np.nan
        if verbose:
            print(f"Remove {np.sum(outliers)} outliers by count threshold")

    # Exclude bins larger than nmad_fact * NMAD
    # TODO: add iterations
    if nmad_fact > 0:
        outliers = np.abs(
            ddem_bins_filtered["value"] - np.median(ddem_bins_filtered["value"])
        ) > 5 * xdem.spatialstats.nmad(ddem_bins_filtered["value"])
        ddem_bins_filtered.loc[outliers, "value"] = np.nan
        if verbose:
            print(f"Remove {np.sum(outliers)} outliers by NMAD filter")

    return ddem_bins_filtered


def fill_ddem(
    ddem: RasterType, ddem_bins: pd.DataFrame, ref_dem: RasterType, roi_mask: np.ndarray = None
) -> RasterType:
    """
    Fill gaps in ddem with values interpolated from altitudinal bins in ddem_bins.

    :param ddem: the ddem to be filled
    :param ddem_bins: the altitudinally averaged ddem bins
    :param ref_dem: a reference elevation used for interpolating all pixels
    :param roi_mask: a mask of pixels to be interpolated
    """
    # Create a model for 2D interpolation
    gradient_model = interp1d(ddem_bins.index.mid, ddem_bins["value"].values, fill_value="extrapolate")

    # Convert ref_dem to an unmasked np.ndarray, needed for interp1d.
    ref_dem_array, ref_mask = gu.spatial_tools.get_array_and_mask(ref_dem)

    # Mask of pixels to be interpolated
    if roi_mask is None:
        mask = ddem.data.mask & ~ref_mask
    else:
        mask = roi_mask & ddem.data.mask & ~ref_mask

    # Fill ddem
    filled_ddem = ddem.copy()
    filled_ddem.data[mask] = gradient_model(ref_dem_array[mask.squeeze()])

    return filled_ddem


def fill_ddem_local_hypso(pair_id, ddems, ref_dem, roi_mask, roi_outlines, filtering=True, plot=True, outfig=None):
    """
    Function to fill gaps in ddems using a local hypsometric approach.
    """
    # Calculate mean elevation change within elevation bins
    # TODO: filter pixels within each bins that are outliers
    ddem = ddems[pair_id]
    ddem_bins = xdem.volume.hypsometric_binning(ddem.data[roi_mask], ref_dem.data[roi_mask])

    # Filter outliers in bins
    if filtering:
        ddem_bins_filtered = ddem_bins_filtering(ddem_bins, verbose=True)
    else:
        ddem_bins_filtered = ddem_bins.copy()

    # Interpolate missing bins
    ddem_bins_filled = xdem.volume.interpolate_hypsometric_bins(ddem_bins_filtered, method="linear")

    # Create 2D filled dDEM
    ddem_filled = fill_ddem(ddem, ddem_bins_filled, ref_dem, roi_mask)

    # Calculate glacier area within those bins
    bins_area = xdem.volume.calculate_hypsometry_area(ddem_bins, ref_dem.data[roi_mask], pixel_size=ref_dem.res)
    obs_area = ddem_bins["count"] * ref_dem.res[0] * ref_dem.res[1]
    frac_obs = obs_area / bins_area

    # Plot
    if plot:
        dh_mean = np.nanmean(ddem.data[roi_mask])
        data, mask = gu.spatial_tools.get_array_and_mask(ddem)
        nobs = np.sum(~mask[roi_mask.squeeze()])
        ntot = np.sum(roi_mask)
        roi_coverage = nobs / ntot
        bin_width = ddem_bins.index.left - ddem_bins.index.right
        
        plotting.plot_mb_fig(pair_id,
                             ddem_bins, 
                             ddem_bins_filled, 
                             bins_area,
                             bin_width,
                             frac_obs,
                             roi_coverage,
                             roi_outlines,
                             dh_mean,
                             ddem,
                             ddem_filled,
                             outfig=outfig)

    # Calculate total volume change and mean dh
    # dV = np.sum(ddem_bins_filled["value"].values * bins_area.values) / 1e9  # in km^3
    # dh_mean = dV * 1e9 / bins_area.sum()

    return ddem_filled, ddem_bins


def calculate_mb(ddem_filled, roi_outlines, stable_mask, plot=False):
    """
    Calculate mean elevation change and volume change for all features in roi_outlines, along with uncertainties.

    Return a panda.DataFrame containing RGIId, area, dh_mean, dh_mean_err, dV, dV_err
    """
    # Calculate ddem NMAD in stable terrain, to be used for uncertainty calculation
    nmad = xdem.spatialstats.nmad(ddem_filled.data[stable_mask])

    # Raise warning if dDEM contains gaps in ROI area (is expected for NO-GAP run)
    gl_mask = roi_outlines.create_mask(ddem_filled)
    dh_subset = ddem_filled.data[gl_mask]
    if (np.sum(dh_subset.mask) > 0) or (np.sum(~np.isfinite(dh_subset.data)) > 0):
        warnings.warn("dDEM contains gaps in ROI - mean value will be biased")

    rgi_ids = roi_outlines.ds.RGIId
    dh_means, dh_means_err, volumes, volumes_err, areas = [], [], [], [], []

    for gid in tqdm(rgi_ids, desc="Looping through all glaciers"):

        # Create mask for selected glacier
        gl_outline = gu.Vector(roi_outlines.ds.loc[rgi_ids == gid])
        gl_mask = gl_outline.create_mask(ddem_filled)

        # Temporarily plot
        if plot:
            ddem_gl = ddem_filled.data.copy()
            ddem_gl.mask[~gl_mask] = True

            extent = (ddem_filled.bounds.left, ddem_filled.bounds.right, ddem_filled.bounds.bottom, ddem_filled.bounds.top)
            ax = plt.subplot(111)
            ax.imshow(ddem_gl.squeeze(), extent=extent, cmap="RdBu", vmin=-50, vmax=50)
            gl_outline.ds.plot(ax=ax, facecolor='none', edgecolor='k')
            plt.xlim(gl_outline.bounds.left, gl_outline.bounds.right)
            plt.ylim(gl_outline.bounds.bottom, gl_outline.bounds.top)
            plt.show()

        # Extract all dh values within glacier and remove masked/nan values
        dh_subset = ddem_filled.data[gl_mask]
        if (np.sum(dh_subset.mask) > 0) or (np.sum(~np.isfinite(dh_subset.data)) > 0):
            dh_subset = dh_subset.compressed()
            dh_subset = dh_subset[np.isfinite(dh_subset)]

        # Calculate mean elevation change and volume change, remove numpy warnings if empty array
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            dh_mean = np.mean(dh_subset)
            # area = np.count_nonzero(gl_mask) * ddem_filled.res[0] * ddem_filled.res[1]
            area = float(gl_outline.ds["Area"] * 1e6)
            dV = dh_mean * area

        # Calculate associated errors bars
        dh_mean_err = err.err_500m_vario(nmad, area)  # err.compute_mean_dh_error(gl_mask, dh_err, vgm_params, res=ddem_filled.res[0])
        dV_err = dh_mean_err * area

        # Save to output lists
        dh_means.append(dh_mean)
        dh_means_err.append(dh_mean_err)
        areas.append(area / 1e6)
        volumes.append(dV / 1e9)
        volumes_err.append(dV_err / 1e9)

    out_df = pd.DataFrame(
        data=np.vstack([rgi_ids, areas, dh_means, dh_means_err, volumes, volumes_err]).T,
        columns=["RGIId", "area", "dh_mean", "dh_mean_err", "dV", "dV_err"]
    )

    return out_df
