"""
Functions to calculate a glacier mass balance.
"""
import matplotlib.pyplot as plt
import numpy as np
import xdem


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in list(ax.spines.values()):
        sp.set_visible(False)


def make_spine_invisible(ax, direction):
    if direction in ["right", "left"]:
        ax.yaxis.set_ticks_position(direction)
        ax.yaxis.set_label_position(direction)
    elif direction in ["top", "bottom"]:
        ax.xaxis.set_ticks_position(direction)
        ax.xaxis.set_label_position(direction)
    else:
        raise ValueError("Unknown Direction : %s" % (direction,))
    ax.spines[direction].set_visible(True)


def mass_balance_local_hypso(ddem, ref_dem, roi_mask, plot=True, outfig=None):
    """
    Function to calculate the MB by filling gaps using a local hypsometric approach
    """
    # Calculate mean elevation change within elevation bins
    ddem_bins = xdem.volume.hypsometric_binning(ddem.data[roi_mask], ref_dem.data[roi_mask])

    # Interpolate missing bins
    ddem_bins_filled = xdem.volume.interpolate_hypsometric_bins(ddem_bins, method="linear")

    # Calculate glacier area within those bins
    bins_area = xdem.volume.calculate_hypsometry_area(ddem_bins, ref_dem.data[roi_mask], pixel_size=ref_dem.res)
    obs_area = ddem_bins["count"] * ref_dem.res[0] * ref_dem.res[1]
    frac_obs = obs_area / bins_area

    # Calculate total volume change and mean dh
    dV = np.sum(ddem_bins_filled["value"].values * bins_area.values) / 1e9  # in km^3
    dh_mean = dV * 1e9 / bins_area.sum()

    # Plot
    if plot:
        bin_width = ddem_bins.index.left - ddem_bins.index.right

        ax1 = plt.subplot(111)
        p1 = plt.plot(ddem_bins["value"], ddem_bins.index.mid, linestyle="-", zorder=1)
        p1b = plt.plot(ddem_bins_filled["value"], ddem_bins.index.mid, linestyle=":", zorder=1)
        plt.xlabel("Elevation change (m)")
        plt.ylabel("Elevation (m)")

        ax2 = ax1.twiny()
        p2 = plt.barh(y=ddem_bins.index.mid, width=bins_area / 1e6, height=bin_width, zorder=2, alpha=0.4)
        plt.xlabel("Glacier area per elevation bins (km\u00b2)")

        ax3 = ax1.twiny()
        ax3.spines["top"].set_position(("axes", 1.2))
        make_patch_spines_invisible(ax3)
        make_spine_invisible(ax3, "top")
        p3 = plt.barh(y=ddem_bins.index.mid, width=frac_obs, height=bin_width, zorder=2, alpha=0.4, color="gray")
        plt.xlabel("Fraction of observations")

        plt.tight_layout()

        # Set ticks color
        tkw = dict(size=4, width=1.5)
        ax1.tick_params(axis="x", colors=p1[0].get_color(), **tkw)
        ax2.tick_params(axis="x", colors=p2.patches[0].get_facecolor(), **tkw)
        ax3.tick_params(axis="x", colors=p3.patches[0].get_facecolor(), **tkw)

        if outfig is None:
            plt.show()
        else:
            plt.savefig(outfig, dpi=200)
            plt.close()

    return ddem_bins, bins_area, frac_obs, dV, dh_mean
