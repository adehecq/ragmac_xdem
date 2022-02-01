"""
Set of tools to estimate the uncertainty in average elevation and volume changes
"""

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import xdem

plt.rcParams["image.interpolation"] = "none"


def compute_mean_dh_error(
    dh: gu.Raster,
    ref_dem: gu.Raster,
    stable_mask: np.ndarray,
    roi_mask: np.ndarray,
    nranges: int = 1,
    plot: bool = False,
    nproc: int = 1,
) -> float:
    """
    Function to calculate the 2-sigma uncertainty in elevation change averaged over the area in roi_mask, using
    error standardization and multi-nested variograms. See notebooks/uncertainty.ipynb for more details.

    :param dh: the elevation change raster
    :param ref_dem: a reference DEM of same shape as dh.
    :param stable_mask: a mask of same shape as dh, set to True over known stable pixels
    :param roi_mask: a mask of same shape as dh, set to True over the area to be averaged
    :param nranges: the number of sperical models/ranges to be fit to the empirical variogram, must be 1 or 2
    :param plot: set to True to display intermediate plots
    :param nproc: number of parallel processes to be used

    :returns: the estimated average elevation change 2-sigma uncertainty
    """
    if nranges not in [1, 2]:
        raise ValueError("`nranges` must be 1 or 2")

    # -- Standardize elevation difference based on slope and max curvature -- #

    # Compute terrain slope and maximum curvature
    slope, planc, profc = xdem.terrain.get_terrain_attribute(
        dem=ref_dem.data, attribute=["slope", "planform_curvature", "profile_curvature"], resolution=ref_dem.res
    )
    maxc = np.maximum(np.abs(planc), np.abs(profc))

    # Plot
    if plot:
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        plt.imshow(slope.squeeze())
        cb = plt.colorbar()
        cb.set_label("Slope (deg)")
        plt.title("Slope")
        ax2 = plt.subplot(122)
        plt.imshow(maxc.squeeze(), vmax=2)
        cb = plt.colorbar()
        cb.set_label("Maximum curvature (100 m$^{-1}$)")
        plt.title("Maximum curvature")
        plt.tight_layout()
        plt.show()

    # Remove values on unstable terrain
    dh_arr = dh.data[stable_mask]
    slope_arr = slope[stable_mask]
    maxc_arr = maxc[stable_mask]

    # Remove large outliers
    dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan

    # Define bins for 2D binning
    custom_bin_slope = np.unique(
        np.concatenate(
            [
                np.nanquantile(slope_arr, np.linspace(0, 0.95, 20)),
                np.nanquantile(slope_arr, np.linspace(0.96, 0.99, 5)),
                np.nanquantile(slope_arr, np.linspace(0.991, 1, 10)),
            ]
        )
    )

    custom_bin_curvature = np.unique(
        np.concatenate(
            [
                np.nanquantile(maxc_arr, np.linspace(0, 0.95, 20)),
                np.nanquantile(maxc_arr, np.linspace(0.96, 0.99, 5)),
                np.nanquantile(maxc_arr, np.linspace(0.991, 1, 10)),
            ]
        )
    )

    # Perform 2D binning to estimate the measurement error with slope and maximum curvature
    df = xdem.spatialstats.nd_binning(
        values=dh_arr,
        list_var=[slope_arr, maxc_arr],
        list_var_names=["slope", "maxc"],
        statistics=["count", np.nanmedian, np.nanstd, xdem.spatialstats.nmad],
        list_var_bins=[custom_bin_slope, custom_bin_curvature],
    )

    if plot:
        # 1D plots
        # xdem.spatialstats.plot_1d_binning(df, "slope", "nmad", "Slope (degrees)", "NMAD of dh (m)")
        # xdem.spatialstats.plot_1d_binning(df, "maxc", "nmad", "Maximum absolute curvature (100 m$^{-1}$)", "NMAD of dh (m)")

        # 2D plot
        xdem.spatialstats.plot_2d_binning(
            df,
            "slope",
            "maxc",
            "nmad",
            "Slope (degrees)",
            "Maximum absolute curvature (100 m$^{-1}$)",
            "NMAD of dh (m)",
            scale_var_2="log",
            vmin=5,
            vmax=15,
        )
        plt.show()

    # Estimate an empirical relationship, to be applied at each pixel
    slope_curv_to_dh_err = xdem.spatialstats.interp_nd_binning(
        df, list_var_names=["slope", "maxc"], statistic="nmad", min_count=30
    )
    dh_err = slope_curv_to_dh_err((slope, maxc))

    # Plot estimated error
    if plot:
        plt.imshow(dh_err.squeeze(), cmap="Reds", vmax=15)
        cb = plt.colorbar()
        cb.set_label("Elevation measurement error (m)")
        plt.show()

    # standardize elevation change
    z_dh = dh.data / dh_err

    # Plot original vs standardized dh
    if plot:
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        plt.imshow(dh.data.squeeze(), cmap="RdBu", vmin=-15, vmax=15)
        cb1 = plt.colorbar()
        cb1.set_label("Raw elevation change (m)")
        ax2 = plt.subplot(122)
        plt.imshow(z_dh.squeeze(), cmap="RdBu", vmin=-1.5, vmax=1.5)
        cb2 = plt.colorbar()
        cb2.set_label("Standardized elevation change")
        plt.tight_layout()
        plt.show()

    # Remove values on unstable terrain and large outliers
    z_dh.data[~stable_mask] = np.nan
    z_dh.data[np.abs(z_dh.data) > 4] = np.nan

    # Ensure that standardized elevation difference has an std of 1
    scale_fac_std = np.nanstd(z_dh.data)
    z_dh = z_dh / scale_fac_std

    # -- Estimate spatial variance of the error, using variogram --#

    # Calculate empirical variogram
    df_vgm = xdem.spatialstats.sample_empirical_variogram(
        values=z_dh.data.squeeze(), gsd=dh.res[0], subsample=100, runs=30, n_variograms=5, n_jobs=nproc
    )

    # Remove pairs distant more than 2/3 the scene size, because they are undersampled
    df_vgm_filtered = df_vgm[df_vgm.bins < np.max(df_vgm.bins) / 1.5]

    # Fit spherical model(s)
    if nranges == 1:
        fun, params = xdem.spatialstats.fit_sum_model_variogram(
            [
                "Sph",
            ],
            empirical_variogram=df_vgm_filtered,
        )
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")

        # Calculate effective samples for our ROI glacier
        area_tot = np.count_nonzero(roi_mask) * dh.res[0] * dh.res[1]
        neff = xdem.spatialstats.neff_circ(area_tot, [(params[0], "Sph", params[1])])

    elif nranges == 2:
        fun, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], empirical_variogram=df_vgm_filtered)
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")
        print(f"Second spherical model - range: {params[2]:.0f} m - sill: {params[3]:.2f}")

        # Calculate effective samples for our ROI glacier
        area_tot = np.count_nonzero(roi_mask) * dh.res[0] * dh.res[1]
        neff = xdem.spatialstats.neff_circ(area_tot, [(params[0], "Sph", params[1]), (params[2], "Sph", params[3])])

    print("Number of effective samples: {:.1f}".format(neff))

    if plot:
        xdem.spatialstats.plot_vgm(
            df_vgm,
            xscale_range_split=[100, 1000, 10000],
            list_fit_fun=[fun],
            list_fit_fun_label=["Standardized double-range variogram"],
        )
        plt.show()

    # Destandardize the uncertainty
    fac_dh_err = scale_fac_std * np.nanmean(dh_err[roi_mask])
    dh_mean_err = fac_dh_err / np.sqrt(neff)

    return 2 * dh_mean_err
