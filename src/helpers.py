""" Various helper functions for mapping, conversion, and calculation. """
import os

import numpy as np
import pandas as pd


###################
# Mapping Helpers #
###################

def create_venue_id_mapping_dicts_from_file(venues_mapping_file):
    """
    Create a venue name to unique integer venue ID mapping dictionary
    with {venue name: ID} and a unique integer venue ID to venue name
    mapping dictionary with {ID: venue name} from a venue mapping text
    file.

    Parameters
    ----------
    venues_mapping_file : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID

    Returns
    -------
    name_to_id_mapping : dict
        A dictionary for venue name to unique integer venue ID mapping
        with {venue name: ID}.
    id_to_name_mapping : dict
        A dictionary for unique integer venue ID to venue name mapping
        with {ID: venue name}.

    """

    name_to_id_mapping = {}
    id_to_name_mapping = {}

    with open(venues_mapping_file, 'r') as f:

        for line in f:

            venue_name, venue_id = line.strip().split(',')

            name_to_id_mapping[venue_name] = int(venue_id)
            id_to_name_mapping[int(venue_id)] = venue_name

    return(name_to_id_mapping, id_to_name_mapping)


def create_value_id_mapping_dicts_from_df(df, value_column, id_column):
    """
    Create a value to unique integer ID mapping dictionary with {value:
    ID} and a unique integer ID to value mapping dictionary with {ID:
    value} from a DataFrame containing a value column and a
    corresponding unique ID column.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame containing a value column and a corresponding
        unique ID column.
    value_column : str
        The name of the column containing the values.
    id_column : str
        The name of the column containing the corresponding IDs.

    Returns
    -------
    value_to_id_mapping : dict
        A dictionary for value to unique integer ID mapping with
        {value: ID}.
    id_to_value_mapping : dict
        A dictionary for unique integer ID to value mapping with {ID:
        value}.

    """

    value_to_id_mapping = {}
    id_to_value_mapping = {}

    for idx, row in df.iterrows():

        value = row[value_column]
        corresponding_id = row[id_column]

        value_to_id_mapping[value] = corresponding_id
        id_to_value_mapping[corresponding_id] = value

    return(value_to_id_mapping, id_to_value_mapping)


####################
# Schedule Helpers #
####################

def create_team_schedule_df(schedule_df):
    """
    Create a team schedule DataFrame where column team 1 holds the
    team and column team 2 holds the opponent for each team's matches.

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing clean schedule details.

    Returns
    -------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    """

    # Initialize list to hold match information
    team_schedule = []

    for idx, row in schedule_df.iterrows():

        # Create dictionary with match information
        match = {column: row[column] for column in list(schedule_df.columns)}

        # Store in list
        team_schedule.append(match)

        # Copy dictionary with match information
        switched = match.copy()

        # Switch team 1 and team 2 match information
        switched['Team 1'] = row['Team 2']
        switched['Team 2'] = row['Team 1']

        # Store in list
        team_schedule.append(switched)

    # Create team conference schedule DataFrame
    team_schedule_df = pd.DataFrame(team_schedule)

    # Use same column order as schedule DataFrame
    column_order = list(schedule_df.columns)
    team_schedule_df = team_schedule_df[column_order]

    # Sort values in NESCAC schedule
    if 'Date' in team_schedule_df:

        team_schedule_df = team_schedule_df.sort_values(['Team 1', 'Date', 'Location']).reset_index(drop=True)

    # Sort values in optimized schedule
    else:

        team_schedule_df = team_schedule_df.sort_values(['Team 1', 'Week', 'Time Slot', 'Location']).reset_index(drop=True)

    return(team_schedule_df)


################
# File Helpers #
################

def create_team_schedule_filenames_list_from_path(path):
    """
    Create a list of team schedule filenames from a path.

    Parameters
    ----------
    path : str
        The path for a folder that contains team schedule files.

    Returns
    -------
    team_schedule_filenames : str
        A list of team schedule filenames (including the path) found
        in the specified path.

    """

    team_schedule_filenames = [os.path.join(path, filename) for filename in os.listdir(path) if "team_schedule" in filename]

    return(team_schedule_filenames)


######################
# Conversion Helpers #
######################

def convert_seconds_to_hhmm(seconds):
    """
    Convert a value from seconds to hours and minutes in HH:MM format.

    Parameters
    ----------
    seconds : float or int
        The number of seconds.

    Returns
    -------
    hhmm : str
        The number of hours and minutes in HH:MM format.

    """

    # Determine number of full minutes and remainder seconds
    minutes, seconds = divmod(seconds, 60)

    # Determine number of full hours and remainder minutes
    hours, minutes = divmod(minutes, 60)

    # Convert hours and minutes to HH:MM format
    hhmm = "{:d}:{:02d}".format(hours, minutes)

    return(hhmm)


def convert_seconds_to_rounded_hours(seconds):
    """
    Convert a value from seconds to rounded hours.

    Parameters
    ----------
    seconds : float or int
        The value in seconds.

    Returns
    -------
    rounded_hours : int
        The value in rounded hours.
    """

    rounded_hours = int(round(seconds / 60 / 60))

    return(rounded_hours)


#######################
# Calculation Helpers #
#######################

def calculate_total_from_dict(di):
    """
    Calculate the sum of the dictionary values.

    Parameters
    ----------
    di : dict
        A dictionary with {key: numeric value}.

    Returns
    -------
    : int, float
        The sum of the dictionary values.

    """

    return(sum(di.values()))


def calculate_std_from_dict(di):
    """
    Calculate the standard deviation of dictionary values.

    Parameters
    ----------
    di : dict
        A dictionary with {key: numeric value}.

    Returns
    -------
    : float
        The standard deviation of the dictionary values.

    """

    return(np.std(list(di.values())))


def calculate_min_max_rng_from_dict(di):
    """
    Calculate the minimum, maximum, and range of dictionary values.

    Parameters
    ----------
    di : dict
        A dictionary with {key: numeric value}.

    Returns
    -------
    : tuple
        A tuple with the minimum, maximum, and range of the dictionary values.

    """

    di_min = min(di.values())
    di_max = max(di.values())
    di_rng = di_max - di_min

    return(di_min, di_max, di_rng)