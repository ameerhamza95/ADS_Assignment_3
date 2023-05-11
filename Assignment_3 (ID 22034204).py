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

