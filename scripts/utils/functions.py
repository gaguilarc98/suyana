## IMPORTS

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import statsmodels.api as sm
import statsmodels.formula.api as sfm

import folium

from scipy.stats import f, norm
from shapely import Polygon, LineString, Point

from IPython.display import display

from matplotlib.ticker import FuncFormatter


## SETTINGS

pd.set_option('display.max_columns', 50)  # Set max number of columns to display
pd.set_option('display.max_rows', 20)    # Set max number of rows to display


## FUNCTIONS

def compare_models(reduced_model, full_model):
    """
    Compare two nested models (reduced and full) using AIC, BIC, Bayes factor, and F-test.

    Parameters:
        reduced_model: Fitted statsmodels OLS model (reduced model).
        full_model: Fitted statsmodels OLS model (full model).

    Returns:
        dict: A dictionary with AIC, BIC, Bayes factor, F-statistic, and p-value.
    """
    # Extract AIC and BIC
    aic_reduced = reduced_model.aic
    bic_reduced = reduced_model.bic
    aic_full = full_model.aic
    bic_full = full_model.bic

    # Residual Sum of Squares and Degrees of Freedom
    RSS_reduced = reduced_model.ssr
    RSS_full = full_model.ssr
    df_reduced = reduced_model.df_resid
    df_full = full_model.df_resid

    # F-test
    num = (RSS_reduced - RSS_full) / (df_reduced - df_full)  # Numerator
    den = RSS_full / df_full  # Denominator
    F_stat = num / den
    p_value = f.sf(F_stat, df_reduced - df_full, df_full)

    # Bayes Factor (BIC-based approximation)
    bayes_factor = np.exp((bic_reduced - bic_full) / 2)

    return {
        "AIC (Reduced)": aic_reduced,
        "AIC (Full)": aic_full,
        "BIC (Reduced)": bic_reduced,
        "BIC (Full)": bic_full,
        "Bayes Factor": bayes_factor,
        "F-statistic": F_stat,
        "p-value": p_value
    }

def print_hello():
    print("Hello, world!")

def create_quantile_col(df, col_name, field, num_quantiles, asc=False, group_fields=[]):
    """
    Create a quantile column,if necessary within groups.

    df (DataFrame): DataFrame in which to include the quantile column.
    col_name (str): Name of the output column. 
    field (str): Field to use for calculating the quantiles.
    num_quantiles: Number of quantiles to use.
    asc: Calculate quantiles in ascending order.
    group_fields: Groupby this fields before calculating the quantiles.

    Returns:
        df: Same dataframe with an additional column.
    """
    if asc==False:  
        if len(group_fields)==0:
            df[col_name] = (1 + num_quantiles) - (
                 1 + pd.qcut(df[field].rank(method="first"), num_quantiles, labels=False)
            )
        else:
            df[col_name] = (
                df.groupby(group_fields)[field]  # Group by the desired columns
                .rank(method="first")            # Rank the values within each group
                .transform(lambda x: (1 + num_quantiles) - (
                    1 + pd.qcut(x, num_quantiles, labels=False)
                ))
            )
    else:
        if len(group_fields)==0:
            df[col_name] = (
                1 + pd.qcut(df[field].rank(method="first"), num_quantiles, labels=False)
            )
        else:
            df[col_name] = (
                df.groupby(group_fields)[field]  # Group by the desired columns
                .rank(method="first")            # Rank the values within each group
                .transform(lambda x: 1 + pd.qcut(x, num_quantiles, labels=False))
            )
    return df


# Custom function to format latitude values
def format_latitude(x, pos):
    direction = 'N' if x >= 0 else 'S'  # 'N' for North, 'S' for South
    if x.is_integer():
        return f'{abs(x):.0f}° {direction}'
    else:
        return f'{abs(x):.1f}° {direction}'

# Custom function to format longitude values
def format_longitude(x, pos):
    direction = 'E' if x >= 0 else 'W'  # 'E' for East, 'W' for West
    if x.is_integer():
        return f'{abs(x):.0f}° {direction}'
    else:
        return f'{abs(x):.1f}° {direction}'
    
def fill_empty_window(ds_period1, ds_period2):
    # Define the empty time window
    empty_window_start = "2011-10-04"
    empty_window_end = "2012-07-24"

    # Create a temporary dataframe from which to sample
    df_dist = ds_period2.to_dataframe().reset_index()

    # Create a time range for the empty window
    empty_time_range = pd.date_range(empty_window_start, empty_window_end, freq="D")
    lon_range = sorted(df_dist['lon'].unique())
    lat_range = sorted(df_dist['lat'].unique())

    empty_grid = pd.MultiIndex.from_product(
        [lon_range, lat_range, empty_time_range],
        names = ["lon", "lat", "time"]
    ).to_frame(index=False)

    empty_grid['week'] = empty_grid['time'].dt.isocalendar().week

    empty_grid_size = empty_grid.groupby(['lon', 'lat', 'week']).agg(
        size = ("time", "count")
    ).reset_index()

    df_base = df_dist.merge(
        empty_grid_size,
        how = "left",
        on = ['lon', 'lat', 'week']
    )
    
    df_base['size'] = (df_base['size'].fillna(0)).astype(int)
    
    df_base = df_base.groupby(['lon', 'lat', 'week']).apply(
        lambda x: x.sample(
            n=min(len(x), x['size'].iloc[0])  # Ensure not to oversample
        )
    ).reset_index(drop=True)
    
    df_base = df_base[['lon', 'lat', 'week','swc','swc_adjusted']]
    
    df_base['day_index'] = df_base.groupby(['lon', 'lat', 'week']).cumcount()
    # Add a day index to the empty grid
    empty_grid['day_index'] = empty_grid.groupby(['lon', 'lat', 'week']).cumcount()
    
    # Step 5: Merge the expanded sampled data with the empty grid
    empty_grid = empty_grid.merge(
        df_base,
        how = "left",
        on = ['lon', 'lat', 'week', 'day_index']
    ).drop(['day_index'], axis=1)

    ds_empty = empty_grid.set_index(['lon', 'lat', 'time']).to_xarray()
    ds_empty = ds_empty.set_coords('week')
    
    # Concatenate the generated data with the adjusted period 1 and period 2 data
    ds_adjusted = xr.concat(
        [ds_period1, ds_empty, ds_period2], dim="time"
    ).sortby("time")

    ds_adjusted = ds_adjusted.rename({"swc": "swc_raw"})

    return ds_adjusted

def print_help():
    print("Help, world!")