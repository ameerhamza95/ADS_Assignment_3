# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:00:48 2023

@author: HAMZA
"""

import pandas as pd   # Library for data manipulation and analysis
import numpy as np    # Library for numerical computing
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
    # 'Country Name' as the first level and 'Indicator Name' 
    # as the second level
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
    textstr = 'No. of countries belonging\nto each cluster:\n\n' + \
        '\n'.join([f'           Cluster {i+1}: {count_dict.get(i+1, 0)}' \
               for i in range(n)])
    ax.text(0.33, 0.98, textstr, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    # Remove the right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return

def subset_countries(ghg_years_df, forest_years_df):
    """
    Subset the input dataframes to include only for countries
    'China', 'Australia', 'United Kingdom', 'United States', 
    drop NaN values, and return the resulting dataframes.

    Parameters:
    -----------
    df_years_ghg, df_years_forest : pandas.DataFrames
        Dataframes with row index as years and column 
        index as country names.

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing the emissions indicator .

    pandas.DataFrame
        Dataframe containing the forest indicator .
    """
    # Subset the dataframe to include only the selected countries
    emissions = ghg_years_df[['China', 'Australia', 'United Kingdom',
                              'United States']].copy()

    # Drop rows that have NaN values for any of the years
    emissions.dropna(inplace=True)

    # Subset the dataframe to include only the forest indicator
    forest = forest_years_df[['China', 'Australia', 'United Kingdom',
                              'United States']].copy()

    # Drop rows that have NaN values for any of the years
    forest.dropna(inplace=True)

    # Keep only the countries 'China', 'Brazil', 'Indonesia',
    # 'United States' for both dataframes
    emissions = emissions.loc[:, ['China', 'Australia', 
                                  'United Kingdom', 'United States']].copy()
    forest = forest.loc[:, ['China', 'Australia', 
                            'United Kingdom', 'United States']].copy()

    # return the dataframes
    return forest, emissions

def merge_years_dataframes(df1, df2, suffix1, suffix2):
    """
    Merge two dataframes on years with given suffixes.

    Args:
    df1 (pd.DataFrame): First dataframe to be merged.
    df2 (pd.DataFrame): Second dataframe to be merged.
    suffix1 (str): Suffix to be added to column names of df1.
    suffix2 (str): Suffix to be added to column names of df2.

    Returns:
    pd.DataFrame: Merged dataframe with suffixes.
    """
    
    # Drop the second level of multi-index columns from both dataframes
    df1.columns = df1.columns.droplevel(1)
    df2.columns = df2.columns.droplevel(1)
    
    # Merge the two dataframes on years with the given suffixes for 
    # the corresponding columns
    merged_df = pd.merge(df1, df2, on='Years', how='outer', 
                         suffixes=[suffix1, suffix2])

    # Drop rows with NaN values
    merged_df = merged_df.dropna()
    
    # Resetting index and converting datetime value of years column
    merged_df = merged_df.reset_index()
    merged_df["Years"] = pd.to_numeric(merged_df["Years"])
    
    # return the merged dataframe
    return merged_df

def exp_growth(t, scale, growth, t0):
    """ Computes exponential function with 
    scale and growth as free parameters
    """
    
    # defining the growth function
    f = scale * np.exp(growth * (t-t0))
    
    # return the calculated function
    return f

def exp_decay(t, scale, growth, t0):
    """ Computes exponential function with 
    scale and growth as free parameters
    """
    
    # defining the decay function
    f = scale * np.exp(-growth * (t-t0))
    
    # return the calculated function
    return f

def logistics(t, a, k, t0):
    """ Computes logistics function with 
    scale and incr as free parameters
    """
    
    # defining the logistics function
    f = a / (1.0 + np.exp(-k * (t - t0)))
    
    # return the calculated function
    return f

def gaussian(x, a, b, c):
    """ Computes the gaussian function 
    """
    
    # defining the gaussian function
    f = a * np.exp(-(x-b)**2/(2*c**2))
    
    # return the calculated function
    return f

def poly(t, c0, c1, c2, c3):
    """ Computes a polynominal 
    c0 + c1*t + c2*t^2 + c3*t^3
    """
    
    # defining the polynomail function
    t = t - 1990  # as a starting value
    f = c0 + c1*t + c2*t**2 + c3*t**3 
    
    # return the calculated function
    return f

def bell_curve(x, a, b, c, d):
    """
    Calculates the value of a bell curve function at a given point x.

    Args:
    - x (float or array-like): the input variable
    - a (float): the height of the curve's peak
    - b (float): the position of the peak
    - c (float): the width of the curve
    - d (float): the offset of the curve

    Returns:
    - y (float or array-like): the output value(s) of the function at x
    """
    
    # defining the function
    y = a * np.exp(-((x - b) / c)**2) + d
    
    # return the calculated function
    return y

def plot_fit(function, df, x_col, y_col, y0, y1, params, title, ylabel,
             loc, arrow_length, xpos, ypos, units, ax=None):
    """
    Plot a fit of a given function to data from a pandas DataFrame.

    Args:
    - function (callable): the function to fit to the data
    - df (pandas.DataFrame): the DataFrame containing the data
    - x_col (str): the name of the column in `df` containing the x data
    - y_col (str): the name of the column in `df` containing the y data
    - params (tuple): initial parameter values to pass to `curve_fit`
    - title (str): the title of the plot
    - ylabel (str): the label for the y axis
    - arrow length
    - xpos (float): x-coordinate for the text box
    - ypos (float): y-coordinate for the text box
    - units (str): units for the y-axis

    Returns:
    - popt (ndarray): the optimal values for the fit parameters
    """

    # fit the function to the data
    popt, pcorr = opt.curve_fit(function, df[x_col], df[y_col], p0=params)

    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))

    # create extended year range
    years = np.arange(y0, y1)

    # call function to calculate upper and lower limits with extrapolation
    lower, upper = err.err_ranges(years, function, popt, sigmas)

    # plot the data, fit, and error ranges
    if ax is None:
        ax = plt.gca()

    # set the title
    ax.set_title(title, fontsize=14, weight='bold')
    
    # plot the functions
    ax.plot(df[x_col], df[y_col], label="data")
    ax.plot(years, function(years, *popt), label="fit")
    ax.fill_between(years, lower, upper, color='y', alpha=0.5)

    # add an arrow to point to the function line with function name
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    arrow_props = dict(facecolor='black', arrowstyle="->")
    ax.annotate(function.__name__ + " fit", xy=(df[x_col].mean() + 20, 
                function(df[x_col].mean() + 20, *popt)),
                xytext=(df[x_col].mean() + 15, 
                function(df[x_col].mean() + 20, *popt) + arrow_length),
                arrowprops=arrow_props, bbox=bbox_props, fontsize=12)

    # add text box with emissions/land predictions
    textstr = '\n'.join((
        'Predictions for:',
        '2030: {}  +/-  {} e4'\
            .format(np.round(function(2030, *popt) / 1e4, 2),\
                    np.round((upper[40]-lower[40]) / 2.0 / 1e4, 2)),
        '2040: {}  +/-  {} e4'\
            .format(np.round(function(2040, *popt) / 1e4, 2), \
                    np.round((upper[50]-lower[50]) / 2.0 / 1e4, 2)),
        '2050: {}  +/-  {} e4'\
            .format(np.round(function(2050, *popt) / 1e4, 2), \
                    np.round((upper[60]-lower[60]) / 2.0 / 1e4, 2)),
        '\nUnits are in: {}'.format(units)))
    ax.text(xpos, ypos, textstr, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # set axis labels and legend
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc=loc)
    
    # Remove the right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # return the calculated parameters
    return popt

""" Main Program """

# Calling function to read emissions dataframe
df_years_ghg, df_countries_ghg = \
    read_world_health_data('Greenhouse emissions.xls')
    
# Calling function to read forest area dataframe
df_years_forest, df_countries_forest = \
    read_world_health_data('Forest Area.xls')

# Calling the function to subset desired years from both dataframes
df_forest, df_emissions = subset_indicators(df_countries_ghg, 
                                            df_countries_forest)

# Perform statistical analysis
df_forest.describe()

# Perform statistical analysis
df_emissions.describe()

# Calling the function to merge the dataframes
df_combine = merge_dataframes(df_emissions, df_forest, '_emissions', '_forest')

# Perform statistical analysis on merged dataframe
df_combine.describe()

# help on custom module cluster tools
help(ct)

# Calling to show plots and print corr matrix
plot_correlations(df_combine)

# calculate silhouette score for 1990
df_min1990, df_max1990, df_cluster1990 = \
    calculate_silhouette_score(df_combine, "1990_emissions", "1990_forest")

# plot cluters and their centers for 1990
cen_1990, labels_1990 = \
    plot_clusters(df_cluster1990, "1990_emissions", "1990_forest", 1990, 7)

# calculate silhouette score for 2000
df_min2000, df_max2000, df_cluster2000 = \
    calculate_silhouette_score(df_combine, "2000_emissions", "2000_forest")
    
# plot cluters and their centers for 2000
cen_2000, labels_2000 = \
    plot_clusters(df_cluster2000, "2000_emissions", "2000_forest", 2000, 6)

# calculate silhouette score for 2010
df_min2010, df_max2010, df_cluster2010 = \
    calculate_silhouette_score(df_combine, "2010_emissions", "2010_forest")

# plot cluters and their centers for 2010
cen_2010, labels_2010 = \
    plot_clusters(df_cluster2010, "2010_emissions", "2010_forest", 2010, 6)

# calculate silhouette score for 2019
df_min2019, df_max2019, df_cluster2019 = \
    calculate_silhouette_score(df_combine, "2019_emissions", "2019_forest")

# plot cluters and their centers for 2019
cen_2019, labels_2019 = \
    plot_clusters(df_cluster2019, "2019_emissions", "2019_forest", 2019, 5)

# Add labels columns to dataframe
df_combine["labels_1990"] = labels_1990
df_combine["labels_2000"] = labels_2000
df_combine["labels_2010"] = labels_2010
df_combine["labels_2019"] = labels_2019

# Save dataframe to Excel file
df_combine.to_excel("labeled_data.xlsx", index=True)

# Create the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))

# plotting clusters with centers for the years 1990, 2000, 
# 2010 and 2019
plot_clusters_with_centers(cen_1990, df_min1990, df_max1990,
                           'labels_1990', df_combine, 
                           "1990_emissions", "1990_forest", 
                           1990, 7, ax=axes[0, 0])
plot_clusters_with_centers(cen_2000, df_min2000, df_max2000,
                           'labels_2000', df_combine, 
                           "2000_emissions", "2000_forest", 
                           2000, 6, ax=axes[0, 1])
plot_clusters_with_centers(cen_2010, df_min2010, df_max2010,
                           'labels_2010', df_combine, 
                           "2010_emissions", "2010_forest", 
                           2010, 6, ax=axes[1, 0])
plot_clusters_with_centers(cen_2019, df_min2019, df_max2019, 
                           'labels_2019', df_combine, 
                           "2019_emissions", "2019_forest", 
                           2019, 5, ax=axes[1, 1])

# Set the figure face color and edge color
fig.patch.set_facecolor('#F2F2F2')
fig.patch.set_edgecolor('grey')

# Set the title
fig.suptitle('Greenhouse Emissions vs Forest Area showing cluster centers', 
             fontsize=20, y=0.94, weight='bold')

# Save the figure
plt.savefig('Clusters.png', dpi=300)

# Display the subplots
plt.show()

# Calling function to subset countries for both dataframes
df_y_forest, df_y_ghg = subset_countries(df_years_ghg, df_years_forest)

# describe the stats 
df_y_forest.describe()

# describe the stats 
df_y_ghg.describe()

# calling to merge the dataframes on years 
df_merge = merge_years_dataframes(df_y_ghg, df_y_forest, 
                                  '_emissions', '_forest')

# Create the subplots for China
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

plot_fit(logistics, df_merge, 'Years', 'China_emissions', 
         1990, 2051, (0.85e7, 0.08, 2008.0),
         "Greenhouse Emissions over Time", 
         "Greenhouse Emissions (kt)", "upper left", 
         0.53e7, 0.4, 0.3, 'kt of CO2 Eqv.', ax=axes[0])
plot_fit(logistics, df_merge, 'Years', 'China_forest'
         , 1990, 2051, (1.88e6, 0.01, 2005.0),
         "Forest Land over Time", "Forest Land (sq. km)", 
         "upper left", 0.5e6, 0.45, 0.3, 'sq. km', ax=axes[1])

# Set the figure face color and edge color
fig.patch.set_facecolor('#F2F2F2')
fig.patch.set_edgecolor('grey')

# Set the title
fig.suptitle('China: Cluster # 3 Trends with Time', 
             fontsize=20, y=1, weight='bold')

# Save the figure
plt.savefig('China.png', dpi=300)

# Display the subplots
plt.show()      

# Create the subplots for Australia
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

plot_fit(logistics, df_merge, 'Years', 'Australia_emissions',
         1990, 2051, (5e5, 0.02, 1995.0),
         "Greenhouse Emissions over Time", "Greenhouse Emissions (kt)",
         "upper left", 5e4, 0.4, 0.3, 'kt of CO2 Eqv.', ax=axes[0])
plot_fit(bell_curve, df_merge, 'Years', 'Australia_forest', 
         1990, 2051, (1.4e6, 2015, 20, 10),
         "Forest Land over Time", "Forest Land (sq. km)", 
         "lower left", -1.5e4, 0.45, 0.3, 'sq. km', ax=axes[1])


# Set the figure face color and edge color
fig.patch.set_facecolor('#F2F2F2')
fig.patch.set_edgecolor('grey')

# Set the title
fig.suptitle('Australia: Cluster # 1 Trends with Time', 
             fontsize=20, y=1, weight='bold')

# Save the figure
plt.savefig('Australia.png', dpi=300)

# Display the subplots
plt.show()      

# Create the subplots for UK
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

plot_fit(logistics, df_merge, 'Years', 'United Kingdom_emissions', 
         1990, 2051, (6.5e5, 0.009, 2005),
         "Greenhouse Emissions over Time", "Greenhouse Emissions (kt)",
         "lower left", -2.5e5, 0.5, 0.95, 'kt of CO2 Eqv.', ax=axes[0])
plot_fit(logistics, df_merge, 'Years', 'United Kingdom_forest', 
         1990, 2051, (3e4, 0.003, 2005),
         "Forest Land over Time", "Forest Land (sq. km)", "upper left",
         4e3, 0.5, 0.3, 'sq. km', ax=axes[1])

# Set the figure face color and edge color
fig.patch.set_facecolor('#F2F2F2')
fig.patch.set_edgecolor('grey')

# Set the title
fig.suptitle('United Kingdom: Cluster # 1 Trends with Time', 
             fontsize=20, y=1, weight='bold')

# Save the figure
plt.savefig('UK.png', dpi=300)

# Display the subplots
plt.show()      

# Create the subplots for US
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

plot_fit(bell_curve, df_merge, 'Years', 'United States_emissions', 
         1990, 2051, (6.8e6, 2000, 30, 0.5),
         "Greenhouse Emissions over Time", "Greenhouse Emissions (kt)", 
         "upper left", 5e5, 0.5, 0.95, 'kt of CO2 Eqv.', ax=axes[0])
plot_fit(poly, df_merge, 'Years', 'United States_forest', 
         1990, 2051, (3e6, -3.05e3, 521, -11),
         "Forest Land over Time", "Forest Land (sq. km)", 
         "lower left", -5e5, 0.3, 0.3, 'sq. km', ax=axes[1])

# Set the figure face color and edge color
fig.patch.set_facecolor('#F2F2F2')
fig.patch.set_edgecolor('grey')

# Set the title
fig.suptitle('United States: Cluster # 5 Trends with Time', 
             fontsize=20, y=1, weight='bold')

# Save the figure
plt.savefig('US.png', dpi=300)

# Display the subplots
plt.show()      
