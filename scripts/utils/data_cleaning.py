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


def remove_duplicates_and_fill(ds):
    df = ds.to_dataframe().reset_index()

    # Eliminar duplicados basados en 'time' para cada combinación única de 'lon' y 'lat'
    df_unique = (
        df.groupby(["lon", "lat"])
        .apply(lambda group: group.drop_duplicates(subset="time", keep="first"))
        .reset_index(drop=True)
    )

    # Convertir de vuelta a xarray Dataset
    ds_unique = df_unique.set_index(["lon", "lat", "time"]).to_xarray()

    return ds_unique


def empirical_cdf_matching(x, y):
    """
    Performs empirical CDF matching, handling NaN values.

    Args:
      x: The data to be adjusted.
      y: The reference data.

    Returns:
      The adjusted data.
    """

    # Remove NaNs before calculating CDFs
    x_valid = x[~np.isnan(x)]
    y_valid = y[~np.isnan(y)]

    x_cdf = np.sort(x_valid)
    y_cdf = np.sort(y_valid)

    x_indices = np.searchsorted(x_cdf, x, side="right")
    y_quantiles = x_indices / len(x_cdf)  # Quantiles of x in the valid x range
    y_indices = np.floor(y_quantiles * len(y_cdf)).astype(
        int
    )  # Corresponding indices in y

    adjusted_x = np.take(y_cdf, y_indices, mode="clip")

    # Reinsert NaNs in the adjusted data where they were in the original data
    adjusted_x[np.isnan(x)] = np.nan

    return adjusted_x


def clean_data(ds_2002_2011, ds_2012_2024):
    # Remove Duplicates
    ds_2002_2011 = remove_duplicates_and_fill(ds_2002_2011)
    ds_2012_2024 = remove_duplicates_and_fill(ds_2012_2024)

    # Concat
    ds = xr.concat(
        [
            ds_2002_2011,
            ds_2012_2024,
        ],
        dim="time",
    )

    ds["time"] = pd.to_datetime(ds["time"].values)

    ds = ds.reindex(
        time=pd.date_range(
            start=ds["time"].min().values, end=ds["time"].max().values, freq="D"
        )
    )

    # Fill Nans
    ds = ds.where(ds != 65535, np.nan)

    # Average all pixels
    ds = ds.mean(dim=["lat", "lon"])

    # Remove data from 2024-2025 campaign
    ds = ds.sel(time=slice(None, "2024-10-31"))

    return ds


def cdf_matching(ds):
    # Define the periods
    period1_end = "2011-10-03"
    period2_start = "2012-07-24"

    # Split the dataset into two periods
    ds_period1 = ds.sel(time=slice(None, period1_end))
    ds_period2 = ds.sel(time=slice(period2_start, None))

    # Get the week number for each time point
    ds_period1["week"] = ds_period1["time"].dt.isocalendar().week
    ds_period2["week"] = ds_period2["time"].dt.isocalendar().week
    # ds_period1['month'] = ds_period1['time'].dt.month
    # ds_period2['month'] = ds_period2['time'].dt.month

    # Group by week and apply empirical CDF matching
    adjusted_swc = []

    for week in range(1, 53):  # Assuming a year has 52 weeks
        swc_period1_week = ds_period1["swc"].where(
            ds_period1["week"] == week, drop=True
        )
        swc_period2_week = ds_period2["swc"].where(
            ds_period2["week"] == week, drop=True
        )

        if swc_period1_week.isnull().all() or swc_period2_week.isnull().all():
            raise ValueError("Empty week data.")

        # Apply CDF matching
        adjusted_swc_week = empirical_cdf_matching(
            swc_period1_week.values, swc_period2_week.values
        )

        # Create a new DataArray with the adjusted values and the original time coordinates
        adjusted_swc_week_da = xr.DataArray(
            adjusted_swc_week, coords={"time": swc_period1_week.time}, dims=["time"]
        )
        adjusted_swc.append(adjusted_swc_week_da)

    # Concatenate the adjusted values for all weeks
    ds_period1["swc_adjusted"] = xr.concat(adjusted_swc, dim="time")
    ds_period2["swc_adjusted"] = ds_period2["swc"]
    return ds_period1, ds_period2


def fill_empty_window(ds_period1, ds_period2):
    # Define the empty time window
    empty_window_start = "2011-10-04"
    empty_window_end = "2012-07-24"

    # Create a time range for the empty window
    empty_time_range = pd.date_range(empty_window_start, empty_window_end, freq="D")

    # Generate random values for the empty window
    generated_swc = []
    for day in empty_time_range:
        # Get the week number of the current day
        week = day.isocalendar().week

        # Select data for the corresponding week from period 2
        swc_period2_week = ds_period2["swc"].where(
            ds_period2["time"].dt.isocalendar().week == week, drop=True
        )

        # If there's data for that week in period 2, generate random values
        if len(swc_period2_week) > 0:

            # Debug: I need to add a seed value here
            # Fit an empirical distribution to the weekly data
            empirical_dist = np.random.choice(
                swc_period2_week.values, size=1000, replace=True
            )  # Increase size for smoother distribution

            # Debug: I need to add a seed value here
            # Generate a random value from the empirical distribution
            random_value = np.random.choice(empirical_dist)
        else:
            # Handle cases where there's no data for that week (e.g., set to NaN)
            random_value = np.nan

        generated_swc.append(random_value)

    # Create a DataArray for the generated values
    generated_swc_da = xr.DataArray(
        generated_swc, coords={"time": empty_time_range}, dims=["time"]
    )

    generated_swc_ds = generated_swc_da.to_dataset(name="swc_adjusted")

    # Concatenate the generated data with the adjusted period 1 and period 2 data
    ds_adjusted = xr.concat(
        [ds_period1, generated_swc_ds, ds_period2], dim="time"
    ).sortby("time")

    ds_adjusted = ds_adjusted.rename({"swc": "swc_raw"})

    return ds_adjusted