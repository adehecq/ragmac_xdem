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
        ) > nmad_fact * xdem.spatialstats.nmad(ddem_bins_filtered["value"])
        ddem_bins_filtered.loc[outliers, "value"] = np.nan
        if verbose:
            print(f"Remove {np.sum(outliers)} outliers by NMAD filter")

    return ddem_bins_filtered


def fill_ddem(
        ddem: RasterType, ddem_bins: pd.DataFrame, ref_dem: RasterType, roi_mask: np.ndarray = None
) -> tuple[RasterType, RasterType]:
    """
    Fill gaps in ddem with values interpolated from altitudinal bins in ddem_bins.

    :param ddem: the ddem to be filled
    :param ddem_bins: the altitudinally averaged ddem bins
    :param ref_dem: a reference elevation used for interpolating all pixels
    :param roi_mask: a mask of pixels to be interpolated

    :returns: two Rasters, one containing teh gap-filled DEM within roi_mask, one containing the interpolation residuals, i.e. ddem - interpolated_ddem.
    """
    # Create a model for 2D interpolation
    gradient_model = interp1d(ddem_bins.index.mid, ddem_bins["value"].values, fill_value="extrapolate")

    # Convert ref_dem to an unmasked np.ndarray, needed for interp1d.
    ref_dem_array, ref_mask = gu.spatial_tools.get_array_and_mask(ref_dem)

    # Mask of pixels to be interpolated
    if roi_mask is None:
        roi_mask = np.ones_like(ddem.data.mask)
    mask = roi_mask & ddem.data.mask & ~ref_mask

    # Fill ddem
    interp_ddem = gradient_model(ref_dem_array)
    filled_ddem = ddem.copy()
    filled_ddem.data[mask] = interp_ddem[mask.squeeze()]

    # Calculate residuals
    residuals = ddem - interp_ddem.reshape(ddem.data.shape)
    residuals.data.mask[~roi_mask] = True

    return filled_ddem, residuals


def fill_ddem_local_hypso(ddem, ref_dem, roi_mask, roi_outlines, filtering=True):
    """
    Function to fill gaps in ddems using a local hypsometric approach.

    :param ddem: the ddem to be filled
    :param ref_dem: a reference elevation used for interpolating all pixels
    :param roi_mask: a mask of pixels to be interpolated
    :param roi_outlines: The asscoiated outlines of the ROI
    :param filtering: if set to False, disable filtering

    :returns:
    `ddem_filled` gap-filled DEM raster
    `ddem_bins` the output of hypsometric binning
    `ddem_bins` same as previous with filtered and interpolated elevation bins,
    `interp_residuals` the difference between input and interpolated ddem,
    `frac_obs` the fraction of observation for each feature in roi_outlines
    """
    # Calculate mean elevation change within elevation bins
    # TODO: filter pixels within each bins that are outliers
    ddem_bins = xdem.volume.hypsometric_binning(ddem.data[roi_mask], ref_dem.data[roi_mask])

    # Filter outliers in bins
    if filtering:
        ddem_bins_filtered = ddem_bins_filtering(ddem_bins, verbose=True)
    else:
        ddem_bins_filtered = ddem_bins.copy()

    # Interpolate missing bins
    ddem_bins_filled = xdem.volume.interpolate_hypsometric_bins(ddem_bins_filtered, method="linear")

    # Create 2D filled dDEM
    ddem_filled, interp_residuals = fill_ddem(ddem, ddem_bins_filled, ref_dem, roi_mask)

    # Calculate fraction of observations for each glacier
    data, mask = gu.spatial_tools.get_array_and_mask(ddem)
    frac_obs = np.zeros(len(roi_outlines.ds))
    for k in roi_outlines.ds.index:
        # Create mask for selected glacier
        gl_outline = gu.Vector(roi_outlines.ds.iloc[k: k + 1])
        gl_mask = gl_outline.create_mask(ddem).squeeze()

        # Calculate coverage over glaciers
        nobs = np.sum(~mask[gl_mask])
        ntot = np.sum(gl_mask)
        frac_obs[k] = nobs / ntot

    return ddem_filled, ddem_bins, ddem_bins_filled, interp_residuals, frac_obs


def calculate_mb(ddem_filled, roi_outlines, stable_mask, ddem_raw, plot=False):
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

    # Get mask of valid observations
    _, obs_mask = gu.spatial_tools.get_array_and_mask(ddem_raw)

    # Variables to be extracted during loop
    nfeatures = len(roi_outlines.ds)
    dh_means, dh_spat_errs, areas, frac_obs = np.zeros((4, nfeatures))

    for k in tqdm(roi_outlines.ds.index, desc="Looping through all glaciers"):

        # Create mask for selected glacier
        gl_outline = gu.Vector(roi_outlines.ds.iloc[k: k + 1])
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

        # Calculate spatially correlated error
        dh_spat_err = err.err_500m_vario(nmad, area)  # err.compute_mean_dh_error(gl_mask, dh_err, vgm_params, res=ddem_filled.res[0])

        # Calculate fraction of valid observations
        nobs = np.sum(~obs_mask[gl_mask.squeeze()])
        ntot = np.sum(gl_mask)
        frac_obs[k] = nobs / ntot

        # Save to output lists
        dh_means[k] = dh_mean  # m
        dh_spat_errs[k] = dh_spat_err
        areas[k] = area / 1e6  # km2

    # Calculate volume change
    volumes = dh_means / 1e3 * areas  # km3

    # -- Calculate final uncertainty -- #
    # Relative uncertainty in area with a 30 m buffer
    area_err = err.err_area_buffer(roi_outlines, buffer=30, plot=False)

    # Final error, calculated for volume in km3 rather than dh, according to RAGMAC expected outputs
    dV_spat_err = frac_obs * dh_spat_errs * areas / 1e3
    dV_interp_err = 5 * (1 - frac_obs) * dh_spat_errs * areas / 1e3
    dV_area_err = np.abs(dh_means) * area_err * areas / 1e3
    dV_temporal_err = np.zeros_like(dV_area_err)
    dV_err = np.sqrt(dV_spat_err**2 + dV_area_err**2 + dV_interp_err**2 + dV_temporal_err**2)
    dh_err = dV_err / areas * 1e3

    # Save to data frame
    rgi_ids = roi_outlines.ds.RGIId
    out_df = pd.DataFrame(
        data=np.vstack([rgi_ids, areas, dh_means, dh_err, volumes, dV_err, area_err, dV_spat_err, dV_area_err, dV_interp_err, dV_temporal_err]).T,
        columns=["RGIId", "area", "dh_mean", "dh_mean_err", "dV", "dV_err", "area_err", "dV_spat_err", "dV_area_err", "dV_interp_err", "dV_temporal_err"]
    )

    # Convert relevant columns to numeric type
    for col in out_df.columns:
        if col != 'RGIId':
            out_df[col] = pd.to_numeric(out_df[col])

    return out_df, nmad
