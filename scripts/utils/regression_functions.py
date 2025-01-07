## IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as sfm

import folium

from scipy.stats import f

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