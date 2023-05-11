# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:00:48 2023

@author: HAMZA
"""

import pandas as pd   # Library for data manipulation and analysis
import numpy as np    # Library for numerical computing
import scipy.stats as stats       # Library for statistical analysis
import seaborn as sns             # Library for data visualization
import matplotlib.pyplot as plt   # Library for creating visualizations

# from sklearn import cluster
import sklearn.cluster as cluster  # Library for clustering algorithms
import sklearn.metrics as skmet    # Library for measuring clustering 
                                   # performance

import cluster_tools as ct  # Custom module for clustering analysis

import importlib  # Library to reload modules that have already been imported

import scipy.optimize as opt  # Library for optimization and curve fitting

import errors as err  # Custom module for calculating error ranges of fitted 
                      # curve
                      
""" Defining Functions """

def read_world_health_data(filename):
    """ Define a function that reads in a world health 
    data Excel file and returns two dataframes
    """
    
    # Read the Excel file into a Pandas dataframe, 
    # starting from the 4th row
    df = pd.read_excel(filename, header=3)
    
    # Create a copy of the original dataframe
    countries = df.copy()

    # Set the index of the dataframe to be a multiindex with 
    # 'Country Name' as the first level and 'Indicator Name' as the second level
    countries.set_index(['Country Name', 'Indicator Name'], inplace=True)
    
    # Drop the 'Country Code' and 'Indicator Code' columns
    countries.drop(['Country Code', 'Indicator Code'], axis=1, inplace=True)
    
    # Rename the column levels to 'Years'
    countries.columns.names = ['Years']

    # Convert the column headers to datetime format
    countries.columns = pd.to_datetime(countries.columns, format='%Y')
    
    # Extract the year component from the datetime column
    countries.columns = countries.columns.year
    
    # Drop rows and columns with all NaN values
    countries = countries.dropna(axis=0, how='all')
    countries = countries.dropna(axis=1, how='all')
    
    # Transpose the dataframe to get years as columns
    years = countries.transpose().copy()

    # Return both dataframes
    return years, countries

def subset_indicators(ghg_countries_df, forest_countries_df):
    """
    Subset the input dataframes to include only for years 1990, 
    2000, 2010, and 2019, drop NaN values, and return the 
    resulting dataframes.

    Parameters:
    -----------
    ghg_countries_df, forest_countries_df) : pandas.DataFrames
        Dataframes with multi-indexed rows with country names 
        and indicator names and year columns as datetime format.

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the emissions indicator for years 
        1990, 2000, 2010, and 2019.

    pandas.DataFrame
        Dataframe containing the forest indicator for years 
        1990, 2000, 2010, and 2019.
    """
    # Subset the dataframe to include only the emissions indicator
    emissions = ghg_countries_df.copy()

    # Drop rows and columns with all NaN values
    emissions = emissions.dropna(axis=0, how='all')
    emissions = emissions.dropna(axis=1, how='all')

    # Drop rows that have NaN values for all years
    emissions = emissions[emissions[1990].notna()]
    emissions = emissions[emissions[2000].notna()]
    emissions = emissions[emissions[2010].notna()]
    emissions = emissions[emissions[2019].notna()]

    # Subset the dataframe to include only the renewable indicator
    forest = forest_countries_df.copy()

    # Drop rows and columns with all NaN values
    forest = forest.dropna(axis=0, how='all')
    forest = forest.dropna(axis=1, how='all')

    # Drop rows that have NaN values for any of the years
    forest = forest.dropna(subset=[1990, 2000, 2010, 2019])

    # Keep only the years 1990, 2000, 2010, and 2019 for both dataframes
    emissions = emissions[[1990, 2000, 2010, 2019]].copy()
    forest = forest[[1990, 2000, 2010, 2019]].copy()

    # Return both dataframes
    return forest, emissions

def merge_dataframes(df1, df2, suffix1, suffix2):
    """
    Merge two dataframes on 'Country Name' column with given suffixes.

    Args:
    df1 (pd.DataFrame): First dataframe to be merged.
    df2 (pd.DataFrame): Second dataframe to be merged.
    suffix1 (str): Suffix to be added to column names of df1.
    suffix2 (str): Suffix to be added to column names of df2.

    Returns:
    pd.DataFrame: Merged dataframe with suffixes.
    """
    # Merge the two dataframes on 'Country Name' column with 
    # the given suffixes for the corresponding columns
    merged_df = pd.merge(df1, df2, on='Country Name', how='outer'\
                         , suffixes=[suffix1, suffix2])

    # Drop rows with NaN values
    merged_df = merged_df.dropna()

    return merged_df

