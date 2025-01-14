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