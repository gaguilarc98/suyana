"""
Data Cleaning Utilities

This module provides functions for cleaning and processing datasets, including:
- Removing duplicates and filling missing values
- Performing empirical CDF matching
- Cleaning and concatenating datasets
- Filling empty time windows with generated data
"""

import numpy as np
import pandas as pd
import xarray as xr


def coarsen_data(ds, params_coarse):
    ds_new = ds.coarsen(params_coarse, boundary = 'pad').mean()
    return ds_new

def remove_duplicates(ds):
    ds_unique = ds.drop_duplicates(..., keep='first')

    return ds_unique


def clean_data(
    ds, 
    date_range = (None, None), 
    na_replace = None, 
    time_dim = 'time', 
    params_smooth = {'time':21}, 
    params_offset = {'lon':0, 'lat':0}
):
    # Remove duplicates
    if time_dim not in ds.dims:
        raise ValueError(f"Input for time_dim must be a subset of dimensions")
    ds = remove_duplicates(ds)
    
    # Convert time to datetime format
    ds[time_dim] = pd.to_datetime(ds[time_dim].values)
    
    # Create a week coordinate attached to the existing time dimension
    ds = ds.assign_coords(dict(
        week = (time_dim, ds[time_dim].dt.isocalendar().week.values)
    ))

    ds = ds.reindex(
        time=pd.date_range(
            start=ds[time_dim].min().values, end=ds[time_dim].max().values, freq="D"
        )
    )

    # Select data within the time window
    if date_range[0] is None and date_range[1] is None:
        pass
    else:
        ds = ds.sel(time = slice(date_range[0], date_range[1]))
    
    # Replace values with nan (65535 for Planet)
    if isinstance(na_replace, (int, float)):
        ds = ds.where(ds != na_replace, np.nan)

    # Apply a rolling mean to have a smooth time series by pixel
    ds = ds.rolling(params_smooth, min_periods = 1).mean()

    # Adjust the coordinates by an offset value
    for key, value in params_offset.items():
        ds[key] = ds[key] + value

    return ds

def summarize_data(ds, red_dims=[None], group_coords=[None]):
    
    if set(red_dims).issubset(set(list(ds.sizes.keys()))):
        # Average over all grouping columns (dimensions)
        ds = ds.mean(dim = red_dims)
    elif set(group_coords).issubset(set(list(ds.coords))):
        for coord in group_coords:
            ds = ds.groupby(coord).mean()
    else:
        raise ValueError(f"Input for red_cols must be a subset of dimensions")
    return ds

# Function to assign a cluster to a single (lon, lat) pair
def assign_cluster(lon, lat, shapes, cluster_var):
    point = Point(lon, lat)
    for _, row in shapes.iterrows():
        if row["geometry"].contains(point):
            return row[cluster_var]  # Assuming 'cluster_id' is in the shapefile
    return np.nan  # Default if no cluster matches

def create_cluster_coord(ds, shapefile, cluster_var, lon_var='lon', lat_var='lat'):
    # Apply cluster assignment to all (lon, lat) pairs
    # Extract lon and lat from the dataset
    lon = ds[lon_var].values
    lat = ds[lat_var].values

    # Create a 2D meshgrid of (lon, lat) pairs
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Use np.vectorize to apply the assign_cluster function
    vectorized_assign_cluster = np.vectorize(assign_cluster, excluded=["shapes", "cluster_var"])
    cluster = vectorized_assign_cluster(lon_grid, lat_grid, shapes=shapefile, cluster_var=cluster_var)

    # Add the cluster as a new coordinate to the dataset
    ds = ds.assign_coords(cluster=((lat_var, lon_var), cluster))
    
    return ds

def empirical_cdf_matching_grouped(df_fix, df_orig, field, group_cols):
    """
    Performs empirical CDF matching, handling NaN values.

    Args:
      x: The data to be adjusted.
      y: The reference data.

    Returns:
      The adjusted data.
    """
    df_base = df_orig[group_cols + [field]].copy()
    df_base['index'] = df_base.groupby(group_cols)[field].rank(method="first")
    df_base = df_base.dropna(subset=[field],axis=0)

    df_base_size = df_base.groupby(group_cols).agg(
        size=(field, 'count')
    ).reset_index()

    df_base = df_base.rename({field: field+'_adjusted'}, axis=1)
    
    df_fix['rank'] = (
        df_fix.groupby(group_cols)[field].rank(method="max")
        / df_fix.groupby(group_cols)[field].transform('count')
    )
    
    df_fix = df_fix.merge(
        df_base_size,
        how = 'left',
        on = group_cols
    )

    df_fix['index'] = np.floor(df_fix['rank']*df_fix['size'])
    
    df_fix = df_fix.merge(
        df_base[group_cols +['index', field+'_adjusted']],
        how = 'left',
        on = group_cols + ['index']
    ).drop(labels=['index', 'size', 'rank'], axis=1)

    return df_fix

def empirical_cdf_matching(df_fix, df_orig, field, group_cols = None):
    if group_cols is None:
        df_fix['aux__var'] = 1
        df_orig['aux__var'] = 1
        group_cols = ['aux__var']
        df_fix = empirical_cdf_matching_grouped(df_fix, df_orig, field, group_cols)
        df_fix.drop(group_cols, axis=1, inplace=True)
        df_orig.drop(group_cols, axis=1, inplace=True)
    else:
        df_fix = empirical_cdf_matching_grouped(df_fix, df_orig, field, group_cols)
    return df_fix

def cdf_matching_grouped(ds_period1, ds_period2, field, group_cols=None, time_dim = 'time', adjust_field_base = False):
    
    # Turn the xarrays into dataframes to apply the empirical_cdf_matching function
    df_fix = ds_period1.to_dataframe().reset_index()
    df_base = ds_period2.to_dataframe().reset_index()
    
    df_fix = empirical_cdf_matching(
        df_fix, df_base, field , group_cols
    )
    if adjust_field_base: 
        ds_period2[field+'_adjusted'] = ds_period2[field] 
    
    # Create an xarray Dataset
    dims = list(ds_period2.sizes.keys())
    coords = list(ds_period2.coords.keys())
    drop_vars = list(set(coords) - set(dims))

    df_fix.drop(['week'], axis=1, inplace=True)

    ds_period1 = df_fix.set_index(dims).to_xarray()
    
    #ds_period1 = ds_period1.set_coords(coords)
    #ds_period1 = ds_period1.drop_vars(drop_vars)
    
    ds_period1 = ds_period1.assign_coords(dict(
        week = (time_dim, ds_period1[time_dim].dt.isocalendar().week.values)
    ))
    return ds_period1, ds_period2

def scale_data_grouped(df_fix, df_orig, field, time_dim, group_cols=None):
    
    min_day = min(df_fix[time_dim].dt.dayofyear.values)
    max_day = max(df_fix[time_dim].dt.dayofyear.values)
    
    df_base = df_orig.loc[(df_orig[time_dim].dt.dayofyear >= min_day) & (df_orig[time_dim].dt.dayofyear <= max_day)].copy()

    df_base = df_base.groupby(group_cols).agg(
        mean_field = (field, 'mean'),
        std_field = (field, 'std'),
    ).reset_index()

    df_fix['local_mean'] = df_fix.groupby(group_cols)[field].transform('mean')
    df_fix['local_std'] = df_fix.groupby(group_cols)[field].transform('std')

    df_fix  = df_fix.merge(
        df_base,
        how = 'left',
        on = group_cols
    )

    df_fix[field + '_adjusted'] = ((df_fix[field] - df_fix['local_mean'])/df_fix['local_std']) *df_fix['std_field']  + df_fix['mean_field']
    
    df_fix = df_fix.drop(labels=['local_mean', 'local_std', 'std_field', 'mean_field'], axis=1)
    
    return df_fix

def scale_data(df_fix, df_orig, field, time_dim = 'time', group_cols = None):
    if group_cols is None:
        df_fix['aux__var'] = 1
        df_orig['aux__var'] = 1
        group_cols = ['aux__var']
        df_fix = scale_data_grouped(df_fix, df_orig, field, time_dim, group_cols)
        df_fix.drop(group_cols, axis=1, inplace=True)
        df_orig.drop(group_cols, axis=1, inplace=True)
    else:
        df_fix = scale_data_grouped(df_fix, df_orig, field, time_dim, group_cols)
    return df_fix

def scaling_grouped(ds_period1, ds_period2, field,  time_dim = 'time', group_cols=None, adjust_field_base = False):
    
    # Turn the xarrays into dataframes to apply the empirical_cdf_matching function
    df_fix = ds_period1.to_dataframe().reset_index()
    df_base = ds_period2.to_dataframe().reset_index()
    
    df_fix = scale_data(
        df_fix, df_base, field , time_dim, group_cols
    )
    if adjust_field_base: 
        ds_period2[field+'_adjusted'] = ds_period2[field] 
    
    # Create an xarray Dataset
    dims = list(ds_period2.sizes.keys())
    coords = list(ds_period2.coords.keys())
    drop_vars = list(set(coords) - set(dims))
    #df_fix = df_fix.drop(['week'], axis=1)
    ds_period1 = df_fix.set_index(dims).to_xarray()
    
    ds_period1 = ds_period1.set_coords(coords)
    #ds_period1 = ds_period1.drop_vars(drop_vars)
    #ds_period1 = ds_period1.assign_coords(dict(
    #    week = (time_dim, ds_period1[time_dim].dt.isocalendar().week.values)
    #))

    return ds_period1, ds_period2

def create_window_w_data(ds_orig, ds_base, date_range = (None, None)):
    
    # Select data within the time window
    if date_range[0] is None and date_range[1] is None:
        ds_fill = ds_orig
    else:
        ds_fill = ds_orig.sel(time = slice(date_range[0], date_range[1]))

    # Select the nearest neighbors from the grid pixels on ds_base
    ds_fill = ds_fill.sel(
        lat = ds_base.lat, lon = ds_base.lon, method = 'nearest'
    )
    ds_fill = ds_fill.assign_coords(
        lat = ds_base.lat, 
        lon = ds_base.lon, 
        time = ds_fill.time.dt.round('d'),
        week = ('time', ds_fill.time.dt.isocalendar().week.values)
    )

    return ds_fill

def create_empty_data(ds_orig, gdf, date_range = (None, None), cluster_var='nro_cluste', lon_var='lon', lat_var='lat'):
    
    # Select data within the time window
    if date_range[0] is None and date_range[1] is None:
        ds_fill = ds_orig
    else:
        ds_fill = ds_orig.sel(time = slice(date_range[0], date_range[1]))

    # Select the nearest neighbors from the grid pixels on ds_base
    ds_fill = create_cluster_coord(ds_fill, gdf, cluster_var, lon_var, lat_var)
    ds_fill = ds_fill.assign_coords(
        time = ds_fill.time.dt.round('d'),
        week = ('time', ds_fill.time.dt.isocalendar().week.values)
    )

    return ds_fill

def concat_arrays(ds_ini, dim_along = None, *ds_arrays):
    if set([dim_along]).issubset(set(list(ds_ini.sizes.keys()))):
        # Create a list of all datasets to concatenate
        all_ds = [ds_ini] + list(ds_arrays)
        
        # Perform concatenation
        ds_final = xr.concat(all_ds, dim=dim_along)
    else:
        raise ValueError(f"Input for dim_along must be a subset of dimensions")
    
    # Sort and rename as needed
    if dim_along in ds_final.dims:
        ds_final = ds_final.sortby(dim_along)
        
    return ds_final

def add_climatology(ds, field, time_dim = 'time', dims = None):
    # Get full years only to calculate the climatology
    # Group by year and count the number of dates
    ds_days_per_year = ds[time_dim].groupby(time_dim+".year").count()

    # Filter for full years
    full_years = ds_days_per_year.where(ds_days_per_year >= 365).dropna('year')
    ds_full = ds.sel(time=ds["time.year"].isin(full_years.year))
    
    if dims == None:
        dims = [time_dim]

    if not set(dims).issubset(set(list(ds.sizes.keys()))):
        raise ValueError(f"Input for dims must be a subset of dimensions")

    climatology = ds_full[field].groupby(time_dim+".dayofyear").mean(dims)
    ds["climatology"] = xr.DataArray(
        climatology.sel(dayofyear=ds[time_dim+".dayofyear"]), 
        dims=list(ds.sizes.keys())
    )

    return ds

def add_anomaly(ds, field):
    # ds["swc"] = ds["swc_adjusted"].rolling(time=21, min_periods=1).mean()
    # First value will not be filled if it is nan, so drop it
    if ds[field].isel(time=0).isnull().any():
        ds = ds.isel(time=slice(1, None))
    ds["anomaly"] = ds[field] - ds["climatology"]
    ds["negative_anomaly"] = ds["anomaly"].where(ds["anomaly"] < 0, np.nan) * (-1)
    ds["negative_anomaly"] = ds["negative_anomaly"].fillna(0)

    return ds

def add_crop_cycles(ds, time_dim):
    # Add a crop_year coordinate and a flag of summer campaign
    ds["crop_year"] = (ds[time_dim] - np.timedelta64(4, 'm')).dt.year
    ds["flag_summer"] = xr.where(ds[time_dim].dt.month.isin([11,12,1,2,3,4]),1,0)

    # Add a crop_cycle coordinate
    ds["crop_cycle"] = 2001 + (ds[time_dim].dt.year - 2002) + (ds[time_dim].dt.month >= 11)
    mask_feb = (ds[time_dim].dt.month == 2) & (ds[time_dim].dt.day >= 16)
    mask_mar_oct = (ds[time_dim].dt.month >= 3) & (ds[time_dim].dt.month <= 10)
    mask = mask_feb | mask_mar_oct
    ds["crop_cycle"] = ds["crop_cycle"].where(~mask)

    ds = ds.set_coords(["crop_year", "flag_summer", "crop_cycle"])
    return ds
