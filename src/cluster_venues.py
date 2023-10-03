""" Various functions to group geographic locations into clusters using
multiple implementations of the k-means algorithm. """
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans

from src.helpers import calculate_std_from_dict
from src.helpers import convert_seconds_to_rounded_hours
from src.k_means import k_means


#####################
# Implement K-Means #
#####################

def convert_geographic_to_cartesian(geographic_points):
    """
    Convert geographic coordinates to cartesian coordinates.

    Parameters
    ----------
    geographic_points : numpy.ndarray
        An array of geographic coordinates as a tuple (latitude, longitude).

    Returns
    -------
    cartesian_points : numpy.ndarray
        An array of cartesian coordinates as a tuple (x, y, z).

    """

    x = np.cos(np.deg2rad(geographic_points[:, 0])) * np.cos(np.deg2rad(geographic_points[:, 1]))
    y = np.cos(np.deg2rad(geographic_points[:, 0])) * np.sin(np.deg2rad(geographic_points[:, 1]))
    z = np.sin(np.deg2rad(geographic_points[:, 0]))

    cartesian_points = np.stack((x, y, z), axis=-1)

    return(cartesian_points)


def convert_cartesian_to_geographic(cartesian_points):
    """
    Convert cartesian coordinates to geographic coordinates.

    Parameters
    ----------
    cartesian_points : numpy.ndarray
        An array of cartesian coordinates as a tuple (x, y, z).

    Returns
    -------
    geographic_points : numpy.ndarray
        An array of geographic coordinates as a tuple (latitude, longitude).

    """

    lon = np.rad2deg(np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0]))
    hyp = np.sqrt(np.add(np.square(cartesian_points[:, 0]), np.square(cartesian_points[:, 1])))
    lat = np.rad2deg(np.arctan2(cartesian_points[:, 2], hyp))

    geographic_points = np.stack((lat, lon), axis=-1)

    return(geographic_points)


def run_specified_k_means(points, k, implementation, coordinate_type, distance_metric):
    """
    Classify points into 'k' clusters using the specified k-means
    implementation, coordinate type, and distance metric.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    k : int
        The number of clusters to form.
    implementation : str
        The specific implementation of k-means. Available
        implementations are 'scratch', 'scipy', 'sklearn':

        'scratch': implementation of k-means from scratch.

        'scipy': implementation of k-means from SciPy's kmeans2 method.

        'sklearn': implementation of k-means from scikit-learn's KMeans method.
    coordinate_type : str
        The specific coordinate type. Available coordinate types are
        'cartesian' or 'geographic':

        'cartesian': cartesian points as a tuple (x, y, z).

        'geographic': geographic points as a tuple (latitude,
        longitude).
    distance_metric : str
        Method for the distance calculation. Available methods are
        'euclidean', 'great_circle', and 'vincenty':

        'euclidean': compute the Euclidean distance between two points.

        'great_circle': use spherical geometry to calculate the surface
        distance between two points. Assumes a spherical model of the
        earth resulting in error up to about 0.5%. Method from GeoPy.

        'vincenty': calculate the distance between two points using the
        formula devised by Thaddeus Vincenty, with the most globally
        accurate ellipsoidal model of the earth (WGS-84). Method from
        GeoPy.

    Returns
    -------
    c_points : numpy.ndarray
        An array of 'k' geographic centroids found at the last
        iteration of k-means. Cartesian centroids converted to
        geographic centroids before the return.
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    Raises
    ------
    ValueError : If implementation or coordinate_type parameter specified incorrectly.

    """

    # Convert geographic points to cartesian if necessary
    if coordinate_type == 'cartesian':

        points = convert_geographic_to_cartesian(points)

    # Run k-means from scratch implementation
    if implementation == 'scratch':

        c_points, cluster_assignments = k_means(points, k, distance_metric)

    # Use SciPy or SkLearn implementation of k-means (both use euclidean distance metric)
    else:

        assert distance_metric == 'euclidean'

        if implementation == 'scipy':

            c_points, cluster_assignments = kmeans2(points, k)

        elif implementation == 'sklearn':

            kmeans = KMeans(n_clusters=k).fit(points)
            cluster_assignments = kmeans.predict(points)
            c_points = kmeans.cluster_centers_

        else:

            raise ValueError("implementation parameter must be 'scratch', 'scipy', or 'sklearn', got '{}'".format(implementation))

    # Convert cartesian centroids to geographic centroids if necessary
    if coordinate_type == 'cartesian':

        return(convert_cartesian_to_geographic(c_points), cluster_assignments)

    elif coordinate_type == 'geographic':

        return(c_points, cluster_assignments)

    else:

        raise ValueError("coordinate_type parameter must be 'cartesian' or 'geographic', got '{}'".format(coordinate_type))


###################
# Test Clustering #
###################

def acceptable_num_points_per_cluster(cluster_assignments, max_acceptable_num_points):
    """
    Check if the maximum number of points per cluster in a clustering
    falls within the acceptable maximum.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.
    max_acceptable_num_points : int
        The maximum acceptable number of points per cluster.

    Returns
    -------
    : bool
        True if the maximum number of points per cluster in a clustering
        falls within the acceptable maximum, False otherwise.

    """

    # Determine maximum number of points per cluster in the clustering
    points_max = np.max(np.unique(cluster_assignments, return_counts=True)[1])

    # Check if maximum falls within the acceptable maximum
    if points_max <= max_acceptable_num_points:

        return(True)

    return(False)


def create_tmp_cluster_assignments(cluster_assignments):
    """
    Use a cluster identifier mapping convention {cluster identifer:
    ascending order cluster identifier} to create temporary cluster
    assignments.

    Example
    -------
    cluster_assignments = [1, 0, 0, 0, 1, 3, 2, 1, 1, 1, 4]
    mapping = {1: 0, 0: 1, 3: 2, 2: 3, 4: 4}
    tmp_cluster_assignments = [0, 1, 1, 1, 0, 2, 3, 0, 0, 0, 4]

    Parameters
    ----------
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    Returns
    -------
    tmp_cluster_assignments : numpy.ndarray
        The tmp assignment of clusters where tmp_cluster_assignments[i]
        is the cluster (based on the cluster identifier mapping
        convention) that the i'th point is closest to.

    """

    mapping = {}

    for cluster in cluster_assignments:

        if cluster not in mapping:

            # Set cluster mapping value to the next cluster identifier (ascending)
            mapping[cluster] = len(mapping)

    tmp_cluster_assignments = [mapping[cluster] for cluster in cluster_assignments]

    return(tmp_cluster_assignments)


def find_index_of_dict_with_key_value_pair(list_of_dicts, key, value):
    """
    Find the index of the dictionary containing the specified key,
    value pair.

    Parameters
    ----------
    list_of_dicts : list
        A list of dictionaries.
    key : str
        The dictionary key in the specified key, value pair to find.
    value :
        The dictionary value in the specified key, value pair to find.

    Returns
    -------
    idx : int
        The index of the dictionary containing the specified key, value
        pair or -1 if the key, value pair is not in the list of dictionaries.

    """

    # Iterate over dictionaries in a list of dictionaries
    for idx, di in enumerate(list_of_dicts):

        # Check if dictionary contains specified key, value pair
        if di[key] == value:

            # Return dictionary of index
            return(idx)

    # Key, value pair not found ==> return -1
    return(-1)


#######################
# Collect Clusterings #
#######################

def run_specified_k_means_specified_num_times(points, k, implementation, coordinate_type, distance_metric, num_times, max_acceptable_num_points):
    """
    Run the specified implementation of the k-means algorithm a
    specified number of times. Create a list of clusterings with
    points per cluster within the acceptable maximum. Exclude
    duplicate clusterings from the list.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    k : int
        The number of clusters to form.
    implementation : str
        The specific implementation of k-means. Available
        implementations are 'scratch', 'scipy', 'sklearn':

        'scratch': implementation of k-means from scratch.

        'scipy': implementation of k-means from SciPy's kmeans2 method.

        'sklearn': implementation of k-means from scikit-learn's KMeans method.
    coordinate_type : str
        The specific coordinate type. Available coordinate types are
        'cartesian' or 'geographic':

        'cartesian': cartesian points as a tuple (x, y, z).

        'geographic': geographic points as a tuple (latitude,
        longitude).
    distance_metric : str
        Method for the distance calculation. Available methods are
        'euclidean', 'great_circle', and 'vincenty':

        'euclidean': compute the Euclidean distance between two points.

        'great_circle': use spherical geometry to calculate the surface
        distance between two points. Assumes a spherical model of the
        earth resulting in error up to about 0.5%. Method from GeoPy.

        'vincenty': calculate the distance between two points using the
        formula devised by Thaddeus Vincenty, with the most globally
        accurate ellipsoidal model of the earth (WGS-84). Method from
        GeoPy.
    num_times : int
        The maximum number of times to run the specified implementation
        of k-means.
    max_acceptable_num_points : int
        The maximum acceptable number of points per cluster.

    Returns
    -------
    clusterings_li : list
        A list of clustering information dictionaries with each
        dictionary including the centroid coordinates, the cluster
        assignments, and the tmp cluster assignments.

    """

    clusterings_li = []

    # Temporarily turn warnings into exceptions
    with warnings.catch_warnings():

        warnings.filterwarnings('error')

        for i in range(num_times):

            try:

                c_points, cluster_assignments = run_specified_k_means(points, k, implementation, coordinate_type, distance_metric)

                if acceptable_num_points_per_cluster(cluster_assignments, max_acceptable_num_points):

                    tmp_cluster_assignments = create_tmp_cluster_assignments(cluster_assignments)

                    # Check if new clustering ==> clustering tmp cluster assignments not found in clusterings list
                    if find_index_of_dict_with_key_value_pair(clusterings_li, 'Tmp Cluster Assignments', tmp_cluster_assignments) == -1:

                        # Create data dictionary of clustering information
                        clustering_information = {
                            'Centroid Coordinates': c_points,
                            'Cluster Assignments': cluster_assignments,
                            'Tmp Cluster Assignments': tmp_cluster_assignments
                        }

                        # Add clustering information to clusterings list
                        clusterings_li.append(clustering_information)

            except Warning:

                continue

    return(clusterings_li)



######################
# Analyze Clustering #
######################

def create_cluster_membership_dict(cluster_assignments):
    """
    Create a dictionary of cluster membership.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    Returns
    -------
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]}.

    """

    cluster_membership_dict = {cluster: [] for cluster in cluster_assignments}

    for idx, cluster in enumerate(cluster_assignments):

        cluster_membership_dict[cluster].append(idx)

    return(cluster_membership_dict)


def create_cluster_max_distance_or_duration_dict(cluster_assignments, one_way_distance_or_duration_matrix):
    """
    Create a dictionary with the maximum one-way driving distance or
    duration to travel between points in a cluster.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.
    one_way_distance_or_duration_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles or
        one-way driving durations in seconds between all points.

    Returns
    -------
    cluster_max_distance_or_duration_dict : dict
        A dictionary with the maximum one-way driving distance in miles
        or duration in seconds between all points in a cluster with
        {cluster: maximum driving distance or duration within cluster}.

    """

    cluster_max_distance_or_duration_dict = {}

    cluster_membership_dict = create_cluster_membership_dict(cluster_assignments)

    for cluster, points in cluster_membership_dict.items():

        # Check if one point or fewer in clustering
        if len(points) <= 1:

            # Maximum distance or duration between no points or a point and itself is 0
            cluster_max_distance_or_duration_dict[cluster] = 0

        else:

            cluster_max_distance_or_duration_dict[cluster] = np.max(
                [one_way_distance_or_duration_matrix.iloc[combo[0], combo[1]] for combo in combinations(points, 2)])

    return(cluster_max_distance_or_duration_dict)


def analyze_clustering(cluster_assignments, one_way_distance_matrix, one_way_duration_matrix_s):
    """
    Analyze a clustering based on cluster assignments and return a
    dictionary with the analysis results.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    one_way_duration_matrix_s : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between all points.

    Returns
    -------
    analysis_results : dict
        A data dictionary with clustering analysis results.

    """

    # Create analysis dictionaries
    cluster_membership_dict = create_cluster_membership_dict(cluster_assignments)
    cluster_max_distance_dict = create_cluster_max_distance_or_duration_dict(cluster_assignments, one_way_distance_matrix)
    cluster_max_duration_s_dict = create_cluster_max_distance_or_duration_dict(cluster_assignments, one_way_duration_matrix_s)
    cluster_max_duration_h_dict = {key: convert_seconds_to_rounded_hours(value) for key, value in cluster_max_duration_s_dict.items()}

    # Analyze clustering
    points_max_per_cluster = np.max(list(np.unique(cluster_assignments, return_counts=True)[1]))

    cluster_max_distance_std = calculate_std_from_dict(cluster_max_distance_dict)

    cluster_max_duration_s_std = calculate_std_from_dict(cluster_max_duration_s_dict)

    cluster_max_duration_h_std = calculate_std_from_dict(cluster_max_duration_h_dict)

    # Create data dictionary with analysis results
    analysis_results = {
        'Cluster Assignments': cluster_assignments,
        'Maximum Teams Per Cluster': points_max_per_cluster,
        'Cluster Membership Dict': cluster_membership_dict,
        'Cluster Maximum Distance Dict': cluster_max_distance_dict,
        'Cluster Maximum Distance Std': cluster_max_distance_std,
        'Cluster Maximum Duration in Seconds Dict': cluster_max_duration_s_dict,
        'Cluster Maximum Duration in Seconds Std': cluster_max_duration_s_std,
        'Cluster Maximum Duration in Hours Dict': cluster_max_duration_h_dict,
        'Cluster Maximum Duration in Hours Std': cluster_max_duration_h_std
    }

    return(analysis_results)


########################
# Finalize Clusterings #
########################

def update_clusterings_master_list(master_li, clusterings_li, clusterings_source, one_way_distance_matrix, one_way_duration_matrix_s):
    """
    Update a clusterings master list with information about each
    clustering in a clusterings list. If a clustering is already in the
    master list, append the clustering source information to the master
    list clustering. If a clustering is not already in the master list,
    add the full clustering information to the master list including
    the new source data and additional clustering analysis data.

    Parameters
    ----------
    master_li : list
        A list of clustering dictionaries with each dictionary
        including the clustering source(s), information, and analysis
        results.
    clusterings_li : list
        A list of clustering information dictionaries with each
        dictionary including the centroid coordinates, the cluster
        assignments, and the tmp cluster assignments. Clusterings in
        the clusterings list result from a specific k_means setup as
        described in the clusterings source.
    clusterings_source : str
        An identifier for the the k-means setup containing information
        about the implementation, coordinate type, and distance metric.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    one_way_duration_matrix_s : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between all points.

    Returns
    -------
    master_li : list
        An updated master list with additional data for the clusterings
        in the clusterings list.

    """

    for clustering in clusterings_li:

        duplicate_clustering_idx = find_index_of_dict_with_key_value_pair(
            master_li, 'Tmp Cluster Assignments', clustering['Tmp Cluster Assignments'])

        # Found duplicate clustering in master list
        if duplicate_clustering_idx != -1:

            # Append clustering source to the list of clustering sources
            master_li[duplicate_clustering_idx]['Clustering Source List'].append(clusterings_source)

        # New clustering
        else:

            # Create a data dictionary with clustering analysis results
            data = analyze_clustering(clustering['Cluster Assignments'], one_way_distance_matrix, one_way_duration_matrix_s)

            # Add clustering information to data dictionary
            for key, value in clustering.items():
                data[key] = value

            # Add a clustering source list to the data dictionary
            data['Clustering Source List'] = [clusterings_source]

            # Append data dictionary to master list
            master_li.append(data)

    return(master_li)


def create_clusterings_information_df(master_li):
    """
    Create a clusterings information DataFrame containing clustering
    source(s), information, and analysis results.

    Parameters
    ----------
    master_li : list
        A list of clustering dictionaries with each dictionary
        including the clustering source(s), information, and analysis
        results.

    Returns
    -------
    clusterings_information_df : pd.core.frame.DataFrame
        A clusterings information DataFrame containing clustering
        source(s), information, and analysis results. Additional
        columns for the clustering ID and a count of the clustering
        source(s).

    """

    clusterings_information_df = pd.DataFrame(master_li)

    # Add column for the number of clustering sources
    clusterings_information_df['Clustering Source Count'] = clusterings_information_df['Clustering Source List'].apply(lambda x: len(x))

    # Sort values by clustering source count and reset the index
    clusterings_information_df = clusterings_information_df.sort_values(['Clustering Source Count', 'Cluster Maximum Distance Std'], ascending=[False, True]).reset_index(drop=True)

    # Use the index to create a unique integer clustering ID column
    clusterings_information_df['Clustering ID'] = clusterings_information_df.index

    column_order = [
        'Clustering ID',
        'Cluster Assignments',
        'Centroid Coordinates',
        'Clustering Source Count',
        'Clustering Source List',
        'Tmp Cluster Assignments',
        'Maximum Teams Per Cluster',
        'Cluster Membership Dict',
        'Cluster Maximum Distance Dict',
        'Cluster Maximum Distance Std',
        'Cluster Maximum Duration in Seconds Dict',
        'Cluster Maximum Duration in Seconds Std',
        'Cluster Maximum Duration in Hours Dict',
        'Cluster Maximum Duration in Hours Std']

    clusterings_information_df = clusterings_information_df[column_order]

    return(clusterings_information_df)


def cluster_venues_using_multiple_k_means_setups(points, k, k_means_setups_li, num_times, max_acceptable_num_points, one_way_distance_matrix, one_way_duration_matrix_s):
    """
    Group geographic locations into clusters using multiple
    implementations of the k-means algorithm and running each
    implementation the specified number of times. Create a DataFrame
    including clustering source(s), information, and analysis results
    for all non-duplicate clusterings with no more than the specified
    maximum acceptable number of locations per cluster.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    k : int
        The number of clusters to form.
    k_means_setups_li : list
        A list of k-means setups with each setup as a tuple with
        (implementation, coordinate_type, distance_metric).
        implementation : str
            The specific implementation of k-means. Available
            implementations are 'scratch', 'scipy', 'sklearn':

            'scratch': implementation of k-means from scratch.

            'scipy': implementation of k-means from SciPy's kmeans2
            method.

            'sklearn': implementation of k-means from scikit-learn's
            KMeans method.
        coordinate_type : str
            The specific coordinate type. Available coordinate types are
            'cartesian' or 'geographic':

            'cartesian': cartesian points as a tuple (x, y, z).

            'geographic': geographic points as a tuple (latitude,
            longitude).
        distance_metric : str
            Method for the distance calculation. Available methods are
            'euclidean', 'great_circle', and 'vincenty':

            'euclidean': compute the Euclidean distance between two
            points.

            'great_circle': use spherical geometry to calculate the
            surface distance between two points. Assumes a spherical
            model of the earth resulting in error up to about 0.5%.
            Method from GeoPy.

            'vincenty': calculate the distance between two points using
            the formula devised by Thaddeus Vincenty, with the most
            globally accurate ellipsoidal model of the earth (WGS-84).
            Method from GeoPy.
    num_times : int
        The maximum number of times to run the specified implementation
        of k-means.
    max_acceptable_num_points : int
        The maximum acceptable number of points per cluster.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    one_way_duration_matrix_s : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between all points.

    Returns
    -------
    clusterings_information_df : pandas.core.frame.DataFrame
        A clusterings information DataFrame containing clustering
        source(s), information, and analysis results.

    """

    master_li = []

    for setup in k_means_setups_li:

        implementation = setup[0]
        coordinate_type = setup[1]
        distance_metric = setup[2]

        clusterings_li = run_specified_k_means_specified_num_times(points, k, implementation, coordinate_type, distance_metric, num_times, max_acceptable_num_points)

        clusterings_source = "{} - {} ({})".format(implementation, distance_metric, coordinate_type)

        master_li = update_clusterings_master_list(master_li, clusterings_li, clusterings_source, one_way_distance_matrix, one_way_duration_matrix_s)

    clusterings_information_df = create_clusterings_information_df(master_li)

    return(clusterings_information_df)
