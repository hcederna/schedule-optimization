""" Various functions to implement the k-means algorithm from
scratch. """
import numpy as np
from geopy.distance import great_circle, vincenty


def initialize_random_centroids(points, k):
    """
    Generate 'k' random centroids from within the range (maximum -
    minimum) of the points.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    k : int
        The number of clusters to form (the number of centroids to
        generate).

    Results
    -------
    c_points : ndarray
        An array of 'k' cartesian (if points cartesian) or geographic
        (if points geographic) centroids.

    """

    c_points = np.random.uniform(
        points.min(axis=0),
        points.max(axis=0),
        size=(k, points.shape[1]))

    return(c_points)


def calculate_distance_from_point_to_each_centroid(point, c_points, distance_metric):
    """
    Calculate distance from a point to each centroid using the
    specified distance metric.

    Parameters
    ----------
    point : numpy.ndarray
        A cartesian (x, y, z) or geographic (latitude, longitude)
        point.
    c_points : ndarray
        An array of 'k' cartesian (if points cartesian) or geographic
        (if points geographic) centroids.
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
    ------
    distances: numpy.ndarray
        An array of 'k' distance values.

    Raises
    ------
    ValueError : If distance_metric parameter specified incorrectly.

    """

    if distance_metric == 'euclidean':

        distances = np.linalg.norm(point - c_points, axis=1)

        return(distances)

    elif distance_metric == 'great_circle':

        distances = [great_circle(point, centroid).miles for centroid in c_points]

        return(distances)

    elif distance_metric == 'vincenty':

        distances = [vincenty(point, centroid).miles for centroid in c_points]

        return(distances)

    else:

        raise ValueError("distance_metric parameter should be 'euclidean', 'great_circle', or 'vincenty', got '{}'".format(distance_metric))


def assign_clusters(points, c_points, distance_metric):
    """
    Assign each point to the nearest cluster based on the distance to
    each centroid.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    c_points : numpy.ndarray
        An array of 'k' cartesian (if points cartesian) or geographic
        (if points geographic) centroids.
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
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    """

    # Initialize cluster assignments to an array of zeros
    cluster_assignments = np.zeros(len(points), dtype=np.int)

    for i in range(len(points)):

        # Calculate distance from point to each centroid
        distances = calculate_distance_from_point_to_each_centroid(points[i], c_points, distance_metric)

        # Assign point to nearest cluster (where cluster = index of nearest centroid)
        cluster_assignments[i] = int(np.argmin(distances))

    return(cluster_assignments)


def update_centroids(points, c_points, cluster_assignments):
    """
    Update the centroids by computing the average of the assigned
    points.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    c_points : numpy.ndarray
        An array of 'k' cartesian (if points cartesian) or geographic
        (if points geographic) centroids.
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    Returns
    -------
    updated_c_points : numpy.ndarray
        An array of 'k' updated cartesian (if points cartesian) or
        geographic (if points geographic) centroids.

    """

    # Initialize the updated centroids to an array of zeros
    updated_c_points = np.zeros(c_points.shape)

    for i in range(len(c_points)):

        # Create list of points assigned to centroid
        assigned_points = [points[j] for j in range(len(points)) if cluster_assignments[j] == i]

        # Update centroid as average of assigned points
        updated_c_points[i] = np.mean(assigned_points, axis=0)

    return(updated_c_points)


def k_means(points, k, distance_metric):
    """
    Classify points into 'k' clusters using the k-means algorithm.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    k : int
        The number of clusters to form.
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
        An array of 'k' cartesian (if points cartesian) or geographic
        (if points geographic) centroids found at the last iteration
        of k-means.
    cluster_assignments : numpy.ndarray
        The assignment of clusters where cluster_assignments[i] is the
        centroid the i'th point is closest to.

    """

    # Randomly initialize cluster centroids
    c_points = initialize_random_centroids(points, k)

    # Initialize update_needed boolean
    update_needed = True

    while update_needed:

        # Assign nearest cluster to each point
        cluster_assignments = assign_clusters(points, c_points, distance_metric)

        # Store the old centroid values
        c_points_old = np.array(c_points)

        # Update centroids as the average of each cluster's points
        c_points = update_centroids(points, c_points, cluster_assignments)

        # Determine if another update is needed
        update_needed = (c_points != c_points_old).any()

    return(c_points, cluster_assignments)
