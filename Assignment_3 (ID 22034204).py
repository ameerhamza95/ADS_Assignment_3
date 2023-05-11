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

def plot_correlations(df):
    """
    Function that takes a dataframe as input, calculates the 
    correlation matrix, and plots a heatmap and scatter 
    matrix of the correlations.

    Parameters:
    df (pandas.DataFrame): The dataframe to calculate 
    correlations for and plot.

    Returns:
    None
    """

    # Calculate correlation matrix
    corr = df.corr()
    print(corr)

    # Plot heatmap of correlation matrix
    ct.map_corr(df_combine)
    plt.title('Correlation Matrix Heatmap')

    # Plot scatter matrix of correlations
    fig, axs = plt.subplots(nrows=len(df.columns), 
                            ncols=len(df.columns), 
                            figsize=(10, 10))
    scatter_matrix = pd.plotting.scatter_matrix(df, ax=axs, 
                                                s=5, alpha=0.8, 
                                                diagonal='hist')
    # set the title
    plt.suptitle('Correlation Scatter Matrix')
    plt.tight_layout() # to reduce overlapping

    # Rotate x-labels in scatter matrix
    for i in range(len(df.columns)):
        axs[-1, i].set_xlabel(df.columns[i], rotation=90, ha='center', 
                              va='top')
        axs[i, 0].set_ylabel(df.columns[i], rotation=0, ha='right', 
                             va='center')
        axs[i, 0].yaxis.labelpad = 30

    # Display the plots
    plt.show()
    
    return

def calculate_silhouette_score(df, column1, column2):
    """
    Function that takes a dataframe and two column names as input,
    performs KMeans clustering on the selected columns, and returns the
    silhouette score for each number of clusters along with the minimum
    and maximum values of the normalized data.

    Parameters:
    df (pandas.DataFrame): The dataframe to perform clustering on.
    column1 (str): The name of the first column to use for clustering.
    column2 (str): The name of the second column to use for clustering.

    Returns:
    min and max values of normalized data along with normalized dataframe
    """
    # select the desired columns
    cluster_data = df[[column1, column2]].copy()
    
    # enforce reload to eliminate multiple calls
    importlib.reload(ct)
    
    # normalize the data
    cluster_data, df_min, df_max = ct.scaler(cluster_data)

    print("n score")

    # loop over number of clusters
    for ncluster in range(2, 10):
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the data, results are stored in the kmeans object
        kmeans.fit(cluster_data) # fit done on x,y pairs

        # extract labels
        labels = kmeans.labels_

        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_

        # calculate the silhouette score
        print(ncluster, skmet.silhouette_score(cluster_data, labels))

    # return the minimum and maximum values of the normalized data with 
    # dataframe
    return df_min, df_max, cluster_data

def plot_clusters(df_cluster, x_col, y_col, year, n):
    """
    Function that takes a dataframe, two columns x and y, 
    and number of clusters n as input, and returns the 
    cluster centers and a plot of the clusters.

    Parameters:
    df (pandas.DataFrame): The dataframe to cluster and plot.
    x_col (str): The name of the column to use as the x-axis for the plot.
    y_col (str): The name of the column to use as the y-axis for the plot.
    n (int): The number of clusters to use for k-means clustering.
    year (int): The year for which to calculate clusters
    
    Returns:
    cen (numpy.ndarray): The cluster centers as an array of shape (n, 2).
    labels+1: Labels list
    None
    """

    # Fit the data to k-means clustering
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(df_cluster)
    labels = kmeans.labels_  # extract labels
    cen = kmeans.cluster_centers_  # extract centers

    # Plot the clusters
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')  # define the colormap
    # plot the values
    sc = plt.scatter(df_cluster[x_col], df_cluster[y_col],
                     10, c=labels, marker="o", cmap=cm)
    plt.scatter(cen[:, 0], cen[:, 1], 45, "k", marker="d")
    plt.xlabel(x_col)  # set xlabel
    plt.ylabel(y_col)  # set ylabel
    # set the title
    plt.title(f"Greenhouse emissions vs Forest land for year: {year}\n {n} Clusters")
    # set the legend
    handles, _ = sc.legend_elements()
    legend_labels = [f"Cluster {label}" for label in range(1, n+1)]
    plt.legend(handles, legend_labels, title="Cluster")
    
    # display the plot
    plt.show()

    # return centers and labels for clusters
    return cen, labels+1

def plot_clusters_with_centers(cen, df_min, df_max, labels, df, col1, 
                               col2, year, n, ax):
    """
    Function that takes in cluster centers, min/max values of 
    the original dataframe, labels, combined dataframe, 
    and column names for two columns, and returns a scatter 
    plot of the data with cluster centers.

    Parameters:
    cen (numpy.ndarray): The cluster centers as an array of 
    shape (n, 2).
    df_min (pandas.Series): A series containing the minimum 
    values of each column in the original dataframe.
    df_max (pandas.Series): A series containing the maximum 
    values of each column in the original dataframe.
    labels (numpy.ndarray): The cluster labels for each 
    data point in the combined dataframe.
    df_combine (pandas.DataFrame): The combined dataframe 
    with emissions and forest columns.
    col1 (str): The name of the first column to use as 
    the x-axis for the plot.
    col2 (str): The name of the second column to use as 
    the y-axis for the plot.
    year (int): The year for which the plot is generated.
    n (int): The number of clusters.
    ax (matplotlib.axes.Axes): The axis object to plot the figure on.

    Returns:
    None
    """

    # Move the cluster centers to the original scale
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    # Get the counts of each cluster
    counts = df[labels].value_counts()
    
    # Create a dictionary to map cluster numbers to counts
    count_dict = dict(zip(counts.index, counts.values))

    # Plot the data points and cluster centers
    cm = plt.cm.get_cmap('tab10')
    sc = ax.scatter(df[col1], df[col2], 10, c=df[labels].values, 
                    marker="o", cmap=cm)
    ax.scatter(xcen, ycen, 45, "k", marker="d")
    # set xlabel
    ax.set_xlabel('Greenhouse Emissions (kt of CO2 Eqv.)', 
                  fontsize=12)
    # set ylabel
    ax.set_ylabel('Forest Area (sq. km)', fontsize=12)
    # set title
    ax.set_title(f"For year: {year}\n {n} Clusters", fontsize=14, 
                 weight='bold')
    # set legend
    handles, _ = sc.legend_elements()
    legend_labels = [f"Cluster {label}" for label in range(1, n+1)]
    ax.legend(handles, legend_labels, title="Cluster")
    
    # Add text box with cluster counts
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = 'No. of countries belonging\nto each cluster:\n\n' + 
    '\n'.join([f'           Cluster {i+1}: {count_dict.get(i+1, 0)}' \
               for i in range(n)])
    ax.text(0.33, 0.98, textstr, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    # Remove the right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return

