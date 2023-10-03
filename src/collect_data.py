""" Various functions to determine GPS coordinates, driving distance
between locations, and driving duration between locations. Additional
functions to create a coordinates DataFrame, a driving distance
squareform matrix DataFrame, and a driving duration squareform matrix
DataFrame. """
import random
from itertools import combinations

import googlemaps
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

from src.helpers import convert_seconds_to_hhmm
from src.helpers import create_value_id_mapping_dicts_from_df
from src.helpers import create_venue_id_mapping_dicts_from_file


# Enter your API key below...
GOOGLE_MAPS_API_KEY = "Enter your API key here"
METERS_IN_MILE = 1609.34


###################
# GPS Coordinates #
###################

def query_gmaps_api_for_gps_coordinates(venue_names):
    """
    Query Google Maps API for the GPS coordinates for a list of venues.

    Parameters
    ----------
    venue_names : list
        A list of venue names.

    Returns
    -------
    coordinates_data : dict
        A data dictionary with a list of venue names, GPS coordinates
        as a tuple (latitude, longitude), latitudes, and longitudes.

    Raises
    ------
    Exception : If GPS coordinates for a venue not found.

    """

    gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)

    coordinates_data = {
        'Venue Name': [],
        'Coordinates': [],
        'Latitude': [],
        'Longitude': []
    }

    for venue_name in venue_names:

        try:

            results = gmaps.geocode(address=venue_name)

            latitude = results[0]['geometry']['location']['lat']
            longitude = results[0]['geometry']['location']['lng']
            coordinates = (latitude, longitude)

            coordinates_data['Venue Name'].append(venue_name)
            coordinates_data['Coordinates'].append(coordinates)
            coordinates_data['Latitude'].append(latitude)
            coordinates_data['Longitude'].append(longitude)

        except Exception:

            raise Exception("Error finding the GPS coordinates for {}.".format(venue_name))

    return(coordinates_data)


def create_coordinates_df(coordinates_data, venues_mapping_file):
    """
    Create a coordinates DataFrame from a coordinates data dictionary.

    Parameters
    ----------
    coordinates_data : dict
        A data dictionary with a list of venue names, GPS coordinates
        as a tuple (latitude, longitude), latitudes, and longitudes.
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID

    Returns
    -------
    coordinates_df : pandas.core.frame.DataFrame
        A coordinates DataFrame including the venue ID, coordinates,
        latitude, and longitude.

    """

    coordinates_df = pd.DataFrame(coordinates_data)

    name_to_id_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[0]

    coordinates_df['Venue ID'] = coordinates_df['Venue Name'].map(name_to_id_mapping)

    column_order = [
        'Venue ID',
        'Coordinates',
        'Latitude',
        'Longitude']

    coordinates_df = coordinates_df[column_order]

    return(coordinates_df)


def test_venue_coordinates_df(coordinates_df, venues_mapping_file, num_tests, max_acceptable_error):
    """
    Test the GPS coordinates for random venue(s) and return True if the
    test succeeds and False if the test fails.

    Parameters
    ----------
    coordinates_df : pandas.core.frame.DataFrame
        A coordinates DataFrame including the venue ID, coordinates,
        latitude, and longitude.
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    num_tests : int
        The number of tests to run.
    max_acceptable_error : float
        The maximum acceptable difference between latitude values and
        longitude values.

    Returns
    -------
    : bool
        True if the difference between latitude values and longitude
        values falls within the maximum acceptable error, False
        otherwise.

    """

    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[1]

    for test in range(num_tests):

        # Select random venue from coordinates DataFrame
        random_venue_id = random.choice(list(coordinates_df['Venue ID'].unique()))

        # Find venue coordinates in coordinates DataFrame
        df_coordinates = coordinates_df.loc[coordinates_df['Venue ID'] == random_venue_id, 'Coordinates'].item()

        # Query Google Maps API for coordinates
        gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)
        results = gmaps.geocode(address=id_to_name_mapping[random_venue_id])
        latitude = results[0]['geometry']['location']['lat']
        longitude = results[0]['geometry']['location']['lng']
        gmaps_coordinates = (latitude, longitude)

        # Check if latitude or longitude falls outside acceptable error threshold
        if (abs(df_coordinates[0] - gmaps_coordinates[0]) > max_acceptable_error) or (abs(df_coordinates[1] - gmaps_coordinates[1]) > max_acceptable_error):

            print("Test - FAILURE")
            print("...Venue ID: {}".format(random_venue_id))
            print("...Coordinates DataFrame: {}".format(df_coordinates))
            print("...Google Maps API Query: {}".format(gmaps_coordinates))

            return(False)

    return(True)


def create_venue_coordinates_df(venues_mapping_file, num_tests=5, max_acceptable_error=0.00000001):
    """
    Determine the GPS coordinates for a list of venues and create and
    test a coordinates DataFrame.

    Parameters
    ----------
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    num_tests : int (default = 5)
        The number of tests to run.
    max_acceptable_error : float (default = 0.00000001)
        The maximum acceptable difference between latitude values and
        longitude values.

    Returns
    -------
    coordinates_df : pandas.core.frame.DataFrame
        A coordinates DataFrame that passed the specified number of
        tests and includes the venue IDs, GPS coordinates, latitudes, and longitudes.

    Raises
    ------
    Exception : If the venue coordinates DataFrame test fails.

    """

    name_to_id_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[0]
    venue_names = sorted(list(name_to_id_mapping.keys()))

    coordinates_data = query_gmaps_api_for_gps_coordinates(venue_names)
    coordinates_df = create_coordinates_df(coordinates_data, venues_mapping_file)

    test_success = test_venue_coordinates_df(coordinates_df, venues_mapping_file, num_tests, max_acceptable_error)

    if test_success:

        return(coordinates_df)

    else:

        raise Exception("Correct test failure and try again.\n")


def create_centroid_coordinates_df(coordinates_array):
    """
    Create a centroid coordinates DataFrame from an array of centroid
    GPS coordinates.

    Parameters
    ----------
    coordinates_array : numpy.ndarray
        An array of GPS coordinates as [latitude, longitude].

    Returns
    -------
    coordinates_df : pandas.core.frame.DataFrame
        A centroid coordinates DataFrame including the centroid ID,
        coordinates, latitude, and longitude.

    """

    coordinates_df = pd.DataFrame(coordinates_array, columns=['Latitude', 'Longitude'])

    # Create separate coordinates column
    coordinates_df['Coordinates'] = list(zip(coordinates_df['Latitude'], coordinates_df['Longitude']))

    # Create centroid ID column
    coordinates_df['Centroid ID'] = coordinates_df.index

    column_order = [
        'Centroid ID',
        'Coordinates',
        'Latitude',
        'Longitude']

    coordinates_df = coordinates_df[column_order]

    return(coordinates_df)


#############################
# Driving Distance/Duration #
#############################

def query_gmaps_api_for_one_way_driving_distance_and_duration(venue_names):
    """
    Query Google Maps API for the driving distance and duration between
    all combination pairs of venues from a list of venues.

    Parameters
    ----------
    venue_names : list
        A list of venue names.

    Returns
    -------
    distance_duration_data : dict
        A data dictionary with a list of venue 1 names, a list of venue
        2 names, a list of driving distances in miles, and a list of
        driving durations in seconds.

    Raises
    ------
    Exception : If the distance or duration between venue 1 and venue 2
    not found.

    """

    # Instantiate Google Maps API session
    gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)

    # Initialize data dictionary to hold values
    distance_duration_data = {
        'Venue 1': [],
        'Venue 2': [],
        'Distance (mi)': [],
        'Duration (s)': []
    }

    # Collect driving distance and duration data for each venue to venue combination
    for (venue_1, venue_2) in combinations(venue_names, 2):

        try:

            # Query Google Maps API for driving distance and duration
            trip = gmaps.distance_matrix(
                origins=[venue_1],
                destinations=[venue_2],
                mode="driving",
                units="metric")

            # Extract driving distance in meters and convert to miles
            distance_m = trip['rows'][0]['elements'][0]['distance']['value']
            distance_mi = round(distance_m / METERS_IN_MILE)

            # Extract driving duration in seconds
            duration_s = trip['rows'][0]['elements'][0]['duration']['value']

            # Add values to data dictionary
            distance_duration_data['Venue 1'].append(venue_1)
            distance_duration_data['Venue 2'].append(venue_2)
            distance_duration_data['Distance (mi)'].append(distance_mi)
            distance_duration_data['Duration (s)'].append(duration_s)

        except Exception:

            raise Exception("Error finding the distance between {} and {}.".format(venue_1, venue_2))

    return(distance_duration_data)


def create_distance_and_duration_df(distance_duration_data):
    """
    Create a distance and duration DataFrame from a distance and
    duration data dictionary.

    Parameters
    ----------
    distance_duration_data : dict
        A data dictionary with a list of venue 1 names, a list of venue
        2 names, a list of driving distances in miles, and a list of
        driving durations in seconds.

    Returns
    -------
    distance_duration_df : pandas.core.frame.DataFrame
        A distance and duration DataFrame including venue 1, venue 2,
        distance (mi), and duration (s).

    """

    distance_duration_df = pd.DataFrame(distance_duration_data)

    column_order = [
        'Venue 1',
        'Venue 2',
        'Distance (mi)',
        'Duration (s)']

    distance_duration_df = distance_duration_df[column_order]

    return(distance_duration_df)


def create_squareform_matrix_df(distance_duration_df, column_name):
    """
    Create a squareform matrix DataFrame from values in the specified
    column of the distance and duration DataFrame.

    Parameters
    ----------
    distance_duration_df : pandas.core.frame.DataFrame
        A distance and duration DataFrame including venue 1, venue 2,
        distance (mi), and duration (s).
    column_name : str
        The column containing values to use in the squareform matrix.

    Returns
    -------
    squareform_matrix_df : pandas.core.frame.DataFrame
        A squareform matrix DataFrame of the values in the specified
        column.

    """
    # Create list of values sorted by venue
    values = distance_duration_df[column_name].tolist()

    # Create list of venues (sorted by venue)
    venues = list(np.unique(distance_duration_df[['Venue 1', 'Venue 2']].values))

    # Create squareform matrix DataFrame
    squareform_matrix_df = pd.DataFrame(squareform(values), index=venues, columns=venues)

    return(squareform_matrix_df)


def test_distance_matrix(distance_matrix, venues_mapping_file, num_tests, max_acceptable_error):
    """
    Test the distance matrix values for random venue combinations and
    return True if the test succeeds and False if the test fails.

    Parameters
    ----------
    distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between venues represented as venue IDs.
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    num_tests : int
        The number of tests to run.
    max_acceptable_error : int
        The maximum acceptable difference between distance values in
        miles.

    Returns
    -------
    : bool
        True if the difference between distance values falls within the
        maximum acceptable error, False otherwise.

    """

    # Create a dictionary for venue ID to venue name mapping
    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[1]

    # Run specified number of tests
    for test in range(num_tests):

        # Select random venue 1 and venue 2 locations from distance matrix
        random_venue_1 = random.choice(list(distance_matrix.index))
        random_venue_2 = random.choice(list(distance_matrix.index))

        # Find distance value in distance matrix DataFrame
        dm_distance = distance_matrix.iloc[random_venue_1, random_venue_2]

        # Query Google Maps API for distance value in miles
        gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)
        trip = gmaps.distance_matrix(
            origins=id_to_name_mapping[random_venue_1],
            destinations=id_to_name_mapping[random_venue_2],
            mode="driving",
            units="metric")
        gmaps_distance = round(trip['rows'][0]['elements'][0]['distance']['value'] / METERS_IN_MILE)

        # Check if distance value falls outside error threshold
        if abs(dm_distance - gmaps_distance) > max_acceptable_error:

            print("Test - FAILURE")
            print("...Venue 1: {}".format(random_venue_1))
            print("...Venue 2: {}".format(random_venue_2))
            print("...Distance Matrix: {}".format(dm_distance))
            print("...Google Maps API Query: {}".format(gmaps_distance))

            return(False)

    return(True)


def test_duration_matrix_seconds(duration_matrix_seconds, venues_mapping_file, num_tests, max_acceptable_error):
    """
    Test the duration matrix values for random venue combinations and
    return True if the test succeeds and False if the test fails.

    Parameters
    ----------
    duration_matrix_seconds : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between venues represented as venue IDs.
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    num_tests : int
        The number of tests to run.
    max_acceptable_error : int
        The maximum acceptable difference between duration values in
        seconds.

    Returns
    -------
    : bool
        True if the difference between duration values falls within the
        maximum acceptable error, False otherwise.

    """

    # Create a dictionary for venue ID to venue name mapping
    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[1]

    # Run specified number of tests
    for test in range(num_tests):

        # Select random venue 1 and venue 2 locations from duration matrix
        random_venue_1 = random.choice(list(duration_matrix_seconds.index))
        random_venue_2 = random.choice(list(duration_matrix_seconds.index))

        # Find duration value in duration matrix DataFrame
        dm_duration = duration_matrix_seconds.iloc[random_venue_1, random_venue_2]

        # Query Google Maps API for duration value in seconds
        gmaps = googlemaps.Client(GOOGLE_MAPS_API_KEY)
        trip = gmaps.distance_matrix(
            origins=id_to_name_mapping[random_venue_1],
            destinations=id_to_name_mapping[random_venue_2],
            mode="driving",
            units="metric")
        gmaps_duration = trip['rows'][0]['elements'][0]['duration']['value']

        # Check if duration value falls outside error threshold
        if abs(dm_duration - gmaps_duration) > max_acceptable_error:

            print("Test - FAILURE")
            print("...Venue 1: {}".format(random_venue_1))
            print("...Venue 2: {}".format(random_venue_2))
            print("...Duration Matrix: {}".format(dm_duration))
            print("...Google Maps API Query: {}".format(gmaps_duration))

            return(False)

    return(True)


def create_one_way_distance_matrix_and_duration_matrix(venues_mapping_file=None, coordinates_array=None, num_tests=5, dist_max_acceptable_error=5, dur_max_acceptable_error=600):
    """
    Create a squareform one-way driving distance matrix and a
    squareform one-way driving duration matrix for venues specified in
    a venues mapping file or for centroids specified in an array of GPS
    coordinates.

    Parameters
    ----------
    venues_mapping_file : str (default = None)
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    coordinates_array : numpy.ndarray (default = None)
        An array of centroid GPS coordinates as [latitude, longitude].
    num_tests : int (default = 5)
        The number of tests to run on the distance matrix and on the
        duration matrix.
    dist_max_acceptable_error : int (default = 5)
        The maximum acceptable difference in miles between a distance
        value from the distance matrix and the distance value from a
        corresponding Google Maps API query.
    dur_max_acceptable_error : int (default = 600)
        The maximum acceptable difference in seconds between a duration
        value from the duration matrix and the duration value from a
        corresponding Google Maps API query.

    Returns
    -------
    distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between venues or centroids.
    duration_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between venues or centroids.

    """

    assert venues_mapping_file is not None or coordinates_array is not None

    if venues_mapping_file is not None:

        # Create a dictionary for venue name to venue ID mapping
        name_to_id_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[0]

        # Extract venue names as the locations
        locations = sorted(list(name_to_id_mapping.keys()))

    elif coordinates_array is not None:

        # Create a centroid coordinates DataFrame
        coordinates_df = create_centroid_coordinates_df(coordinates_array)

        # Extract centroid GPS coordinates as the locations
        locations = coordinates_df['Coordinates']

        # Create a dictionary for centroid coordinates to centroid ID mapping
        name_to_id_mapping = create_value_id_mapping_dicts_from_df(coordinates_df, 'Coordinates', 'Centroid ID')[0]

    # Query Google Maps API for driving distance and duration values
    distance_duration_data = query_gmaps_api_for_one_way_driving_distance_and_duration(locations)

    # Create a distance and duration DataFrame
    distance_duration_df = create_distance_and_duration_df(distance_duration_data)

    # Map names/coordinates to IDs in distance and duration DataFrame
    distance_duration_df['Venue 1'] = distance_duration_df['Venue 1'].map(name_to_id_mapping)
    distance_duration_df['Venue 2'] = distance_duration_df['Venue 2'].map(name_to_id_mapping)

    # Create a squareform distance matrix
    distance_matrix = create_squareform_matrix_df(distance_duration_df, 'Distance (mi)')

    # Create squareform duration matrix
    duration_matrix = create_squareform_matrix_df(distance_duration_df, 'Duration (s)')

    # Run tests for venue matrices
    if venues_mapping_file is not None:

        distance_test_success = test_distance_matrix(distance_matrix, venues_mapping_file, num_tests, dist_max_acceptable_error)

        duration_test_success = test_duration_matrix_seconds(duration_matrix, venues_mapping_file, num_tests, dur_max_acceptable_error)

        if distance_test_success and duration_test_success:

            return(distance_matrix, duration_matrix)

        else:

            raise Exception("Correct test failure and try again.")

    # Do not run tests for centroid matrices
    else:

        return(distance_matrix, duration_matrix)


def convert_duration_matrix_seconds_to_hhmm(duration_matrix_s):
    """
    Convert a duration matrix in seconds to hours and minutes in HH:MM
    format.

    Parameters
    ----------
    duration_matrix_s : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in seconds
        between venues or centroids.

    Returns
    -------
    duration_matrix_hhmm : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations in hours and
        minutes in HH:MM format between venues or centroids.

    """

    # Vectorize function to convert seconds to HH:MM
    v_convert_seconds_to_hhmm = np.vectorize(convert_seconds_to_hhmm)

    # Apply the vectorized function to the duration matrix in seconds
    hhmm_values = v_convert_seconds_to_hhmm(duration_matrix_s)

    # Create a DataFrame from the resulting HH:MM values
    duration_matrix_hhmm = pd.DataFrame(hhmm_values)

    return(duration_matrix_hhmm)


################
# Save Results #
################

def determine_centroid_distance_matrix_filename(clustering_id):
    """
    Determine a filename for the centroid distance matrix based on
    the clustering ID.

    Parameters
    ----------
    clustering_id : int
        A unique integer ID for the clustering.

    Returns
    -------
    filename : str
        A centroid distance matrix filename including the folder path.

    """

    filename = "data/location/centroid_one_way_distance_matrix_clustering_" + str(clustering_id) + ".csv"

    return(filename)
