"""
Set of tools to estimate the uncertainty in average elevation and volume changes
"""

import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np
import xdem

plt.rcParams["image.interpolation"] = "none"


def compute_mean_dh_error_1step(
    dh: gu.Raster,
    ref_dem: gu.Raster,
    stable_mask: np.ndarray,
    roi_mask: np.ndarray,
    nranges: int = 1,
    plot: bool = False,
    nproc: int = 1,
) -> float:
    """
    DEPRECATED
    Function to calculate the 2-sigma uncertainty in elevation change averaged over the area in roi_mask, using
    error standardization and multi-nested variograms. See notebooks/uncertainty.ipynb for more details.

    Old function tested for development, but works only for a single estimate, not multiple.
    For several estimates, use compute_standardized_error_and_vario followed by compute_mean_dh_error, to avoid calculating the geospatial statistics mutliple times.

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


def get_uncertainty_function(
    dh: gu.Raster,
    ref_dem: gu.Raster,
    stable_mask: np.ndarray,
    nranges: int = 1,
    plot: bool = False,
    nproc: int = 1,
) -> float:
    """
    Function to calculate a modeled elevation change error (used to standardize the error) and a multi-nested variogram model.
    This is needed to calculate the 2-sigma uncertainty in elevation change averaged over any given area.
    See notebooks/uncertainty.ipynb for more details. This corresponds to steps 1 and 2.

    :param dh: the elevation change raster
    :param ref_dem: a reference DEM of same shape as dh.
    :param stable_mask: a mask of same shape as dh, set to True over known stable pixels
    :param nranges: the number of sperical models/ranges to be fit to the empirical variogram, must be 1 or 2
    :param plot: set to True to display intermediate plots
    :param nproc: number of parallel processes to be used

    :returns: the modeled elevation change error dh_err, and variogram parameters, to be fed to `compute_mean_dh_error`.
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
        fun, params = xdem.spatialstats.fit_sum_model_variogram(["Sph",], empirical_variogram=df_vgm_filtered,)
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")
        vgm_params = [(params[0], "Sph", params[1])]

    elif nranges == 2:
        fun, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], empirical_variogram=df_vgm_filtered)
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")
        print(f"Second spherical model - range: {params[2]:.0f} m - sill: {params[3]:.2f}")
        vgm_params = [(params[0], "Sph", params[1]), (params[2], "Sph", params[3])]

    if plot:
        xdem.spatialstats.plot_vgm(
            df_vgm,
            xscale_range_split=[100, 1000, 10000],
            list_fit_fun=[fun],
            list_fit_fun_label=["Standardized double-range variogram"],
        )
        plt.show()

    def err_func(roi_mask):
        """
        Function to calculate the error in mean dh, for a given raster mask, i.e. glacier.
        """
        # Calculate effective samples for our ROI glacier
        area_tot = np.count_nonzero(roi_mask) * dh.res[0]**2
        neff = xdem.spatialstats.neff_circ(area_tot, vgm_params)

        # Calculate associated error
        dh_mean_err = np.nanmean(dh_err[roi_mask]) / np.sqrt(neff)

        return dh_mean_err

    return err_func


def fit_vgm(
    dh: gu.Raster,
    stable_mask: np.ndarray,
    nranges: int = 1,
    plot: bool = False,
    nproc: int = 1,
) -> list:
    """
    Function to calculate a multi-nested variogram model of elevation error (in stable areas).
    This is needed to calculate the 2-sigma uncertainty in elevation change averaged over any given area.

    :param dh: the elevation change raster
    :param stable_mask: a mask of same shape as dh, set to True over known stable pixels
    :param nranges: the number of sperical models/ranges to be fit to the empirical variogram, must be 1 or 2
    :param plot: set to True to display intermediate plots
    :param nproc: number of parallel processes to be used

    :returns: A list containing, for each model, (range, model name, sill)
    """
    if nranges not in [1, 2]:
        raise ValueError("`nranges` must be 1 or 2")

    # Remove values on unstable terrain and large outliers
    dh_arr, mask = gu.spatial_tools.get_array_and_mask(dh)
    dh_arr[~stable_mask.squeeze()] = np.nan
    dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan

    # -- Estimate spatial variance of the error, using variogram --#

    # Calculate empirical variogram
    df_vgm = xdem.spatialstats.sample_empirical_variogram(
        values=dh_arr.squeeze(), gsd=dh.res[0], subsample=100, runs=30, n_variograms=5, n_jobs=nproc
    )

    # Remove pairs distant more than 2/3 the scene size, because they are undersampled
    df_vgm_filtered = df_vgm[df_vgm.bins < np.max(df_vgm.bins) / 1.5]

    # Fit spherical model(s)
    if nranges == 1:
        fun, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", ], empirical_variogram=df_vgm_filtered,)
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")
        vgm_params = [(params[0], "Sph", params[1])]

    elif nranges == 2:
        fun, params = xdem.spatialstats.fit_sum_model_variogram(["Sph", "Sph"], empirical_variogram=df_vgm_filtered)
        print(f"First spherical model - range: {params[0]:.0f} m - sill: {params[1]:.2f}")
        print(f"Second spherical model - range: {params[2]:.0f} m - sill: {params[3]:.2f}")
        vgm_params = [(params[0], "Sph", params[1]), (params[2], "Sph", params[3])]

    if plot:
        xdem.spatialstats.plot_vgm(
            df_vgm,
            xscale_range_split=[100, 1000, 10000],
            list_fit_fun=[fun],
            list_fit_fun_label=["Standardized double-range variogram"],
        )
        plt.show()

    return vgm_params


def err_500m_vario(nmad, area):
    """
    Scale the dh mean error (NMAD) for an averaging zone of size area (in m2) and assuming spatial correlations 
    of 500 m of errors.
    """
    neff = xdem.spatialstats.neff_circ(area, [[500, "Sph", 1.0], ])
    err = nmad / np.sqrt(neff)
    return err


def err_area_buffer(roi_outlines, buffer=30, plot=False):
    """
    Calculate the fractional area uncertainty, estimated by adding a buffer around known outlines.
    Default is 30 m buffer (2-sigma).
    """
    # Create new GeoDataFrame with buffered polygons
    out_gdf = roi_outlines.ds.copy()
    out_gdf.geometry = roi_outlines.ds.buffer(buffer)

    # Calculate relative error in area
    area_err = (out_gdf.area - roi_outlines.ds.area) / roi_outlines.ds.area

    # Plot
    if plot:
        out_gdf['area_err'] = area_err * 100

        fig = plt.figure(figsize=(8, 6))
        ax1 = plt.subplot(111)
        out_gdf.plot(ax=ax1, column='area_err', legend=True, cmap='Reds', vmax=np.percentile(out_gdf['area_err'], 90))
        ax1.set_title('Area error (%)')

        # ax2 = plt.subplot(122)
        # out_gdf.plot(ax=ax2, column='dh_err', legend=True, cmap='Reds', vmax=np.percentile(out_gdf['dh_err'], 90))
        # ax2.set_title('Mean dh error (m)')

        plt.tight_layout()
        plt.show()

    return area_err
