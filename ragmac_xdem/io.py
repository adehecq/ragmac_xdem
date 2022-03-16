"""io functions"""

import pathlib

import geoutils as gu
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling

from dask.distributed import Client, LocalCluster
import logging

"""
@author: friedrichknuth
"""

def dask_start_cluster(nproc, threads=1, ip_addres=None):
    """
    Starts a dask cluster. Can provide a custom IP or URL to view the progress dashboard. 
    This may be necessary if working on a remote machine.
    """
    cluster = LocalCluster(n_workers=nproc,
                       threads_per_worker=threads,
                       silence_logs=logging.ERROR)

    client = Client(cluster)
    
    if ip_addres:
        port = str(cluster.dashboard_link.split(':')[-1])
        url = ":".join([ip_addres,port])
        print('\n'+'Dask dashboard at:',url)
    else:
        print('\n'+'Dask dashboard at:',cluster.dashboard_link)
    
    print('Workers:', nproc)
    print('Threads per worker:', threads, '\n')
    return client

def optimize_dask_chunks(ds,
                         min_chunk_size = 1,
                         max_chunk_size = 10):
    """
    Function to optimize shape and size of dask chunk.
    Finds square root of dim length, then increments chunk shape 
    by square root until size > min_chunk_size and < max_chunk_size.
    Chunk sizes measured in MiB.
    """

    ### More info at https://blog.dask.org/2021/11/02/choosing-dask-chunk-sizes#what-to-watch-for-on-the-dashboard
    ### larger chunk sizes (e.g. 100 MiB) performed slower when analyzing the task graph and performance.
    ### WIP
    
    t = len(ds.time)
    x = len(ds.x)
    y = len(ds.y)
    
    #Find something that is close to a multiple of dim length
    multiple_of_x = round(np.sqrt(x))
    while multiple_of_x > 100:
        multiple_of_x = round(np.sqrt(multiple_of_x))
        
    multiple_of_y = round(np.sqrt(y))
    while multiple_of_y > 100:
        multiple_of_y = round(np.sqrt(multiple_of_y))

    #Adjust chunk size
    x = multiple_of_x
    y = multiple_of_y
    chunksize = ds['band1'][:t,:y,:x].nbytes / 1048576 # convert to MiB

    if chunksize < min_chunk_size or chunksize > max_chunk_size:
        while chunksize < min_chunk_size:
            x += multiple_of_x
            y += multiple_of_y
            chunksize = ds['band1'][:t,:y,:x].nbytes / 1048576
        
        while chunksize > max_chunk_size:
            x -= multiple_of_x
            y -= multiple_of_y
            chunksize = ds['band1'][:t,:y,:x].nbytes / 1048576
    
    #Round off single digits
    x = 10* int(np.ceil(x/10))
    y = 10* int(np.ceil(y/10))

    return x, y, t

def dask_get_mapped_tasks(dask_array):
    """
    Finds tasks associated with chunked dask array.
    """
    # TODO There has to be a better way to do this...
    txt = dask_array._repr_html_()
    idx = txt.find('Tasks')
    strings = txt[idx-20:idx].split(' ')
    tasks_count = max([int(i) for i in strings if i.isdigit()])
    return tasks_count

def stack_geotif_arrays(geotif_files_list):
    """
    Simple function to stack raster arrays. Assumes these are already aligned.

    Inputs
    ----------
    geotif_files_list : list of GeoTIFF files

    Returns
    -------
    ma_stack : numpy.ma.core.MaskedArray
    """
    arrays = []
    for i in geotif_files_list:
        src = gu.georaster.Raster(i)
        masked_array = src.data
        arrays.append(masked_array)
    ma_stack = np.ma.vstack(arrays)
    return ma_stack


def xr_read_geotif(geotif_file_path, chunks='auto', masked=True):
    """
    Reads in single or multi-band GeoTIFF as dask array.

    Inputs
    ----------
    GeoTIFF_file_path : GeoTIFF file path

    Returns
    -------
    ds : xarray.Dataset
        Includes rioxarray extension to xarray.Dataset
    """

    da = rioxarray.open_rasterio(geotif_file_path, chunks=chunks, masked=True)

    # Extract bands and assign as variables in xr.Dataset()
    ds = xr.Dataset()
    for i, v in enumerate(da.band):
        da_tmp = da.sel(band=v)
        da_tmp.name = "band" + str(i + 1)

        ds[da_tmp.name] = da_tmp

    # Delete empty band coordinates.
    # Need to preserve spatial_ref coordinate, even though it appears empty.
    # See spatial_ref attributes under ds.coords.variables used by rioxarray extension.
    del ds.coords["band"]

    # Preserve top-level attributes and extract single value from value iterables e.g. (1,) --> 1
    ds.attrs = da.attrs
    for key, value in ds.attrs.items():
        try:
            if len(value) == 1:
                ds.attrs[key] = value[0]
        except TypeError:
            pass

    return ds


def xr_stack_geotifs(geotif_files_list, datetimes_list, reference_geotif_file, resampling="bilinear", save_to_nc=False):

    """
    Stack single or multi-band GeoTiFFs in memory to reference_geotiff.
    Resample as needed.

    Inputs
    ----------
    geotif_files_list     : list of GeoTIFF file paths
    datetimes_list        : list of datetime objects for each GeoTIFF
    reference_geotif_file : GeoTIFF file path

    Returns
    -------
    ds : xr.Dataset()
    """

    ## Check each geotiff has a datetime associated with it.
    if len(datetimes_list) == len(geotif_files_list):
        pass
    else:
        print("length of datetimes does not latch length of GeoTIF file list")
        print("datetimes:", len(datetimes_list))
        print("geotifs:", len(geotif_files_list))
        return None

    ## Choose resampling method. Defaults to bilinear.
    if isinstance(resampling, type(Resampling.bilinear)):
        resampling = resampling
    elif resampling == "bilinear":
        resampling = Resampling.bilinear
    elif resampling == "nearest":
        resampling = Resampling.nearest
    elif resampling == "cubic":
        resampling = Resampling.cubic
    else:
        resampling = Resampling.bilinear

    ## Get target object with desired crs, res, bounds, transform
    ## TODO: Parameterize crs, res, bounds, transform
    ref = xr_read_geotif(reference_geotif_file)

    ## Stack geotifs and dimension in time
    datasets = []

    for index, file_name in enumerate(geotif_files_list):
        src = xr_read_geotif(file_name)
        if not check_xr_rio_ds_match(src, ref):
            print("Resampling", file_name, "to", reference_geotif_file)
            src = src.rio.reproject_match(ref, resampling=resampling)
        src = src.assign_coords({"time": datetimes_list[index]})
        src = src.expand_dims("time")

        if save_to_nc:
            out_fn = str(pathlib.Path(file_name).with_suffix("")) + ".nc"
            src.to_netcdf(out_fn)
            out_dir = str(pathlib.Path(geotif_files_list[index]).parents[0])

        datasets.append(src)

    ds = xr.concat(datasets, dim="time", combine_attrs="no_conflicts")
    return ds


def check_xr_rio_ds_match(ds1, ds2):
    """
    Checks if spatial attributes, crs, bounds, and transform match.

    Inputs
    ----------
    ds1 : xarray.Dataset with rioxarray extension
    ds2 : xarray.Dataset with rioxarray extension

    Returns
    -------
    bool
    """

    if (
        (ds1["spatial_ref"].attrs == ds2["spatial_ref"].attrs)
        & (ds1.rio.crs == ds2.rio.crs)
        & (ds1.rio.bounds() == ds2.rio.bounds())
        & (ds1.rio.resolution() == ds2.rio.resolution())
        & (ds1.rio.transform() == ds2.rio.transform())
    ):
        return True
    else:
        return False
