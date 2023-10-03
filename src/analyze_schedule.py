""" Various functions to analyze a volleyball conference team schedule. """
import itertools

import numpy as np
import pandas as pd

from src.helpers import calculate_min_max_rng_from_dict
from src.helpers import calculate_total_from_dict
from src.helpers import convert_seconds_to_rounded_hours
from src.helpers import create_venue_id_mapping_dicts_from_file


####################
# SINGLE SCHEDULES #
####################

def create_teams_list_from_df(df):
    """
    Create a list of teams from a schedule DataFrame.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame with team 1 and team 2 columns.

    Returns
    -------
    teams_list : list
        A list of teams from the team 1 and team 2 DataFrame columns.

    """

    teams_list = list(np.unique(df[['Team 1', 'Team 2']].values))

    return(teams_list)


#########################
# Analysis Dictionaries #
#########################

def is_weekend_overnight(previous_match, current_match, one_way_distance_matrix, weekend_overnight_threshold):
    """
    Check if the current match is a Saturday overnight match where a
    team plays on both Friday and Saturday at a location with a Friday
    one-way driving distance outside the weekend overnight threshold.

    Parameters
    ----------
    previous_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the day (str), date or week, team 1 and location.
    current_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the day (str), date or week, team 1 and location.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    weekend_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight between Friday and Saturday matches.

    Returns
    -------
    : bool
        True if the current match is a Saturday overnight match, False
        otherwise.

    """

    # Check if team 1 in previous and current match the same
    if previous_match['Team 1'] == current_match['Team 1']:

        # Check if current match a Saturday game
        if "Sat." in current_match['Day (Str)']:

            # Check previous match and current match has date value ==> indicates a NESCAC team schedule
            if "Date" in previous_match and "Date" in current_match:

                # Check if previous match ocurred yesterday (Friday)
                if (pd.to_datetime(current_match['Date']) - pd.to_datetime(previous_match['Date'])).days == 1:

                    # Check if Friday match one-way travel distance outside overnight threshold
                    if one_way_distance_matrix.iloc[previous_match['Team 1'], previous_match['Location']] > weekend_overnight_threshold:

                        return(True)

            # Check previous match and current match has week value ==> indicates an optimized team schedule
            elif "Week" in previous_match and "Week" in current_match:

                # Check if previous match ocurred yesterday (Friday in same week)
                if previous_match['Week'] == current_match['Week'] and "Fri." in previous_match['Day (Str)']:

                    # Check if Friday match one-way travel distance outside overnight threshold
                    if one_way_distance_matrix.iloc[previous_match['Team 1'], previous_match['Location']] > weekend_overnight_threshold:

                        return(True)

    return(False)


def is_distance_overnight(current_match, one_way_distance_matrix, distance_overnight_threshold):
    """
    Check if the current match requires team to travel to a location
    with a one-way driving distance outside the distance overnight
    threshold.

    Parameters
    ----------
    current_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the day (str), date or week, team 1 and location.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    distance_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight.

    Returns
    -------
    : bool
        True if the current match is a distance overnight match, False
        otherwise.

    """

    # Check if match one-way travel distance outside distance overnight threshold
    if one_way_distance_matrix.iloc[current_match['Team 1'], current_match['Location']] > distance_overnight_threshold:

        return(True)

    return(False)


def create_team_overnight_dict(team_schedule_df, one_way_distance_matrix, weekend_overnight_threshold, distance_overnight_threshold):
    """
    Create a dictionary with the total number of overnights for each
    team. A weekend overnight occurs when a team plays on both Friday
    and Saturday at a location with a Friday one-way driving distance
    outside the weekend overnight threshold. A distance overnight
    occurs if a team does not have a weekend overnight but travels to
    a location with a one-way driving distance outside the distance
    overnight threshold.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    weekend_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight between Friday and Saturday matches.
    distance_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight.

    Returns
    -------
    team_overnight_dict : dict
        A dictionary with the total number of overnights for each team
        as {team id: total number of overnights}.

    """

    teams = create_teams_list_from_df(team_schedule_df)
    team_overnight_dict = {team: 0 for team in teams}

    # Temporarily add a week column to the NESCAC team schedule DataFrame
    if "Week" not in team_schedule_df.columns:

        team_schedule_df['Date'] = pd.to_datetime(team_schedule_df['Date'])
        team_schedule_df['Week'] = team_schedule_df['Date'].dt.strftime('%U')

    # Group team schedule DataFrame by team 1 and week
    for team_week, team_week_df in team_schedule_df.groupby(['Team 1', 'Week']):

        team_week_df = team_week_df.reset_index(drop=True)

        curr_team = team_week[0]
        distance_overnight = False

        # Iterate over team 1 matches for the week
        for idx, row in team_week_df.iterrows():

            current_match = team_week_df.loc[idx]

            # Check for a distance overnight
            if is_distance_overnight(current_match, one_way_distance_matrix, distance_overnight_threshold):

                distance_overnight = True

            # Check for a weekend overnight
            if idx > 0:

                previous_match = team_week_df.loc[idx - 1]

                if is_weekend_overnight(previous_match, current_match, one_way_distance_matrix, weekend_overnight_threshold):

                    # Count weekend overnight
                    team_overnight_dict[curr_team] += 1

                    # Remove distance overnight because have weekend overnight ==> already staying overnight
                    distance_overnight = False

                    break

        # Count distance overnight
        if distance_overnight:

            team_overnight_dict[curr_team] += 1

    return(team_overnight_dict)


def calculate_roundtrip_distance_team_1(match, one_way_distance_matrix):
    """
    Calculate the roundtrip driving distance from team 1 home -->
    match location --> team 1 home.

    Parameters
    ----------
    match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the team 1 and location.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.

    Returns
    -------
    roundtrip_distance : int
        The roundtrip driving distance from team 1 home --> match
        location --> team 1 home.

    """

    roundtrip_distance = 2 * one_way_distance_matrix.iloc[match['Team 1'], match['Location']]

    return(roundtrip_distance)


def calculate_adjustment_to_skip_returning_home_between_matches(previous_match, current_match, one_way_distance_matrix):
    """
    Calculate the distance adjustment if team 1 travels from home -->
    previous match location --> current match location --> home rather
    than roundtrip to previous match location and roundtrip to current
    match location.

    Parameters
    ----------
    previous_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the team 1 and location.
    current_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the team 1 and location.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.

    Returns
    -------
    adjustment : int
        The distance adjustment if team 1 travels from home -->
        previous match location --> current match location --> home
        rather than roundtrip to previous match location and
        roundtrip to current match location.

    """

    adjustment = 0

    # Subtract one-way distance from previous match location --> home
    # Team does not return home after previous match
    adjustment -= one_way_distance_matrix.iloc[previous_match['Location'], previous_match['Team 1']]

    # Add one-way distance from previous match location --> current match location
    # Team travels directly from previous match to current match
    adjustment += one_way_distance_matrix.iloc[previous_match['Location'], current_match['Location']]

    # Subtract one-way distance from home --> current match location
    # Team does not travel from home to current match
    adjustment -= one_way_distance_matrix.iloc[current_match['Team 1'], current_match['Location']]

    return(adjustment)


def is_same_day(previous_match, current_match):
    """
    Check if the current match for team 1 occurs on the same day as
    the previous match for team 1.

    Parameters
    ----------
    previous_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the date or week and day (str), and team 1.
    current_match : pandas.core.series.Series
        A row from the team schedule DataFrame representing a match
        including the date or week and day (str), and team 1.

    Returns
    -------
    : bool
        True if the current match occurs on the same day as the
        previous match for team 1, False otherwise.

    """

    # Check if team 1 previous and current match the same
    if previous_match['Team 1'] == current_match['Team 1']:

        # Check previous match and current match has date value ==> indicates a NESCAC team schedule
        if "Date" in previous_match and "Date" in current_match:

            # Check previous match has the same date as current match
            if pd.to_datetime(previous_match['Date']) == pd.to_datetime(current_match['Date']):

                return(True)

        # Check previous match and current match has week value ==> indicates an optimized team schedule
        elif "Week" in previous_match and "Week" in current_match:

            # Check previous match has the same week and day (str) as current match
            if previous_match['Week'] == current_match['Week'] and previous_match['Day (Str)'] == current_match['Day (Str)']:

                return(True)

    return(False)


def create_team_distance_or_duration_dict(team_schedule_df, one_way_distance_or_duration_matrix, overnight_one_way_distance_matrix, weekend_overnight_threshold):
    """
    Create a dictionary with the total driving distance or duration
    for each team.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.
    one_way_distance_or_duration_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles or
        one-way driving durations in seconds between all points.
    overnight_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points. Used to determine whether a match is a
        weekend overnight.
    weekend_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight between Friday and Saturday matches.

    Returns
    -------
    team_distance_or_duration_dict : dict
        A dictionary with the total driving distance or duration for
        each team as {team id: total distance or duration}.

    """

    teams = create_teams_list_from_df(team_schedule_df)
    team_distance_or_duration_dict = {team: 0 for team in teams}

    for idx in team_schedule_df.index:

        current_match = team_schedule_df.loc[idx]

        # Start with roundtrip distance or duration
        distance_or_duration = calculate_roundtrip_distance_team_1(current_match, one_way_distance_or_duration_matrix)

        # Exclude first index so can make distance or duration adjustments (if necessary) based on previous match
        if idx > team_schedule_df.index[0]:

            previous_match = team_schedule_df.loc[idx - 1]

            if is_weekend_overnight(previous_match, current_match, overnight_one_way_distance_matrix, weekend_overnight_threshold):

                # Skip returning home between previous match and current match
                distance_or_duration += calculate_adjustment_to_skip_returning_home_between_matches(
                    previous_match, current_match, one_way_distance_or_duration_matrix)

            elif is_same_day(previous_match, current_match):

                # Check if previous match location the same as current match location
                if previous_match['Location'] == current_match['Location']:

                    # Skip traveling for current match (team already traveled for previous match)
                    distance_or_duration -= calculate_roundtrip_distance_team_1(
                        current_match, one_way_distance_or_duration_matrix)

                # Otherwise, previous location different than current location
                else:

                    # Skip returning home between previous match and current match
                    distance_or_duration += calculate_adjustment_to_skip_returning_home_between_matches(
                        previous_match, current_match, one_way_distance_or_duration_matrix)

        team_distance_or_duration_dict[current_match['Team 1']] += distance_or_duration

    return(team_distance_or_duration_dict)


def create_team_match_location_dict(team_schedule_df):
    """
    Create a dictionary with a list of unique match locations for
    each team.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    Returns
    -------
    team_match_location_dict : dict
        A dictionary with a list of unique match locations for each
        team as {team id: [unique match locations]}.

    """

    team_match_location_dict = {team: list(team_matches_df['Location'].unique()) for team, team_matches_df in team_schedule_df.groupby('Team 1')}

    return(team_match_location_dict)


def create_team_home_game_dict(team_schedule_df):
    """
    Create a dictionary with the number of home games for each team.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    Returns
    -------
    team_home_game_dict : dict
        A dictionary with the number of home games for each team as
        {team id: number of home games}.

    """

    team_home_game_dict = {}

    for team, team_matches_df in team_schedule_df.groupby('Team 1'):

        team_home_game_dict[team] = sum(1 for location in team_matches_df['Location'] if location == team)

    return(team_home_game_dict)


#################
# Cost Analysis #
#################

def calculate_total_overnight_cost(num_overnights, rooms_per_overnight, cost_per_room):
    """
    Calculate the total cost for the specified number of overnights.

    Parameters
    ----------
    num_overnights : int
        The number of overnights.
    rooms_per_overnight : int
        The number of rooms required to house all support staff and
        players for an overnight.
    cost_per_room : float
        The cost for one room.

    Returns
    -------
    total_overnight_cost : float
        The total cost for all overnights.

    """

    total_overnight_cost = num_overnights * rooms_per_overnight * cost_per_room

    return(total_overnight_cost)


def calculate_total_distance_cost(total_distance, cost_per_unit_distance):
    """
    Calculate the total cost to travel the specified distance.

    Parameters
    ----------
    total_distance : int
        The total distance to travel.
    cost_per_unit_distance : float
        The cost to travel one unit of distance (likely miles).

    Returns
    -------
    total_distance_cost : float
        The total cost to travel the specified distance.

    """

    total_distance_cost = total_distance * cost_per_unit_distance

    return(total_distance_cost)


def create_team_cost_dict(team_distance_dict, team_overnight_dict, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Create a dictionary with total cost for each team.

    Parameters
    ----------
    team_distance_dict : dict
        A dictionary with the total driving distance for each team as
        {team id: total distance}.
    team_overnight_dict : dict
        A dictionary with the total number of overnights for each team
        as {team id: total number of overnights}.
    rooms_per_overnight : int
        The number of rooms required to house all support staff and
        players for an overnight.
    cost_per_room : float
        The cost for one room.
    cost_per_unit_distance : float
        The cost to travel one unit of distance (likely miles).

    Returns
    -------
    team_cost_dict : dict
        A dictionary with the total cost for each team as {team id:
        total team cost}.

    """

    team_cost_dict = {}

    for team in team_distance_dict:

        team_distance_cost = calculate_total_distance_cost(team_distance_dict[team], cost_per_unit_distance)

        team_overnight_cost = calculate_total_overnight_cost(team_overnight_dict[team], rooms_per_overnight, cost_per_unit_distance)

        team_cost_dict[team] = team_distance_cost + team_overnight_cost

    return(team_cost_dict)


#####################
# Fairness Analysis #
#####################

def calculate_fairness_index(values):
    """
    Calculate Jain's fairness index for a list of values.

    See https://www.cse.wustl.edu/~jain/papers/ftp/fairness.pdf for
    fairness equation.

    Parameters
    ----------
    values : list
        A list of values to calculate fairness.

    Returns
    -------
    fairness_index : float
        Jain's fairness index for the specified values.

    """

    numerator = sum(values) ** 2
    denominator = len(values) * sum([val ** 2 for val in values])

    if denominator > 0:

        fairness_index = 100 * (numerator / denominator)

    else:

        fairness_index = 100

    return(fairness_index)


###########
# Testing #
###########

def create_team_opponent_dict(team_schedule_df):
    """
    Create a dictionary with a sorted list of opponents for each team.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    Returns
    -------
    team_opponent_dict : dict
        A dictionary with a list of opponents for each team as {team
        id: [sorted opponents]}.

    """

    team_opponent_dict = {team: sorted(list(team_matches_df['Team 2'])) for team, team_matches_df in team_schedule_df.groupby('Team 1')}

    return(team_opponent_dict)


def test_each_team_plays_every_other_team_once(team_schedule_df, team_opponent_dict):
    """
    Test that each team plays every other team once and return True if
    the test succeeds and False if the test fails.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.
    team_opponent_dict : dict
        A dictionary with a list of opponents for each team as {team
        id: [sorted opponents]}.

    Returns
    -------
    : bool
        True if each team plays every other team once, False
        otherwise.

    """

    teams = create_teams_list_from_df(team_schedule_df)
    test_success = True

    for team in teams:

        target = create_teams_list_from_df(team_schedule_df)
        # Remove current team from target because teams do not play themselves
        target.remove(team)

        actual = team_opponent_dict[team]

        if actual != target:

            test_success = False

            print("Test - FAILURE")
            print("...Actual: {}".format(actual))
            print("...Target: {}".format(target))

    return(test_success)


def create_team_time_slot_dict(team_schedule_df):
    """
    Create a dictionary with a list of time slots for each team.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    Returns
    -------
    team_time_slot_dict : dict
        A dictionary with a list of time slots for each team as {team
        id: [time slots]}.

    """

    teams = create_teams_list_from_df(team_schedule_df)
    team_time_slot_dict = {team: [] for team in teams}

    for idx in team_schedule_df.index:

        team_time_slot_dict[team_schedule_df['Team 1'][idx]].append("%s.%s" % (team_schedule_df['Week'][idx], team_schedule_df['Time Slot'][idx]))

    return(team_time_slot_dict)


def test_each_team_plays_no_more_than_one_match_per_time_slot(team_time_slot_dict):
    """
    Test that each team plays no more than one match per time slot
    and return True if the test succeeds and False if the test fails.

    Parameters
    ----------
    team_time_slot_dict : dict
        A dictionary with a list of time slots for each team as {team
        id: [time slots]}.

    Returns
    -------
    : bool
        True if each team plays no more than one match per time slot,
        False otherwise.

    """

    test_success = True

    for team, time_slots in team_time_slot_dict.items():

        errors = [time_slot for time_slot in time_slots if time_slots.count(time_slot) != 1]

        if errors:

            test_success = False

            print("Test - FAILURE")
            print("...Team: {}".format(team))
            print("...Time Slots: {}".format(time_slots))
            print("...Failed Time Slots: {}".format(errors))

    return(test_success)


def create_location_time_slot_dict(team_schedule_df):
    """
    Create a dictionary with a list of time slots for each location.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.

    Returns
    -------
    location_time_slot_dict : dict
        A dictionary with a list of time slots for each location as
        {location id: [time slots]}. Each time slot appears two times
        because created using the team schedule DataFrame.

    """

    locations = list(np.unique(team_schedule_df['Location'].values))
    location_time_slot_dict = {location: [] for location in locations}

    for idx in team_schedule_df.index:

        location_time_slot_dict[team_schedule_df['Location'][idx]].append("%s.%s" % (team_schedule_df['Week'][idx], team_schedule_df['Time Slot'][idx]))

    return(location_time_slot_dict)


def test_each_location_hosts_no_more_than_one_match_per_time_slot(location_time_slot_dict):
    """
    Test that each location hosts no more than one match per time slot
    and return True if the test succeeds and False if the test fails.

    Parameters
    ----------
    location_time_slot_dict : dict
        A dictionary with a list of time slots for each location as
        {location id: [time slots]}. Each time slot appears two times
        because created using the team schedule DataFrame.

    Returns
    -------
    : bool
        True if each location hosts no more than one match per time
        slot, False otherwise.

    """

    test_success = True

    for location, time_slots in location_time_slot_dict.items():

        errors = [time_slot for time_slot in time_slots if time_slots.count(time_slot) != 2]

        if errors:

            test_success = False

            print("Test - FAILURE")
            print("...Location: {}".format(location))
            print("...Time Slots: {}".format(time_slots))
            print("...Failed Time Slots: {}".format(errors))

    return(test_success)


#################################
# Full Single Schedule Analysis #
#################################

def analyze_team_schedule_df(team_schedule_df, one_way_distance_matrix, one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Analyze a team schedule DataFrame and return a dictionary with
    analysis results.

    Parameters
    ----------
    team_schedule_df : pandas.core.frame.DataFrame
        A team schedule DataFrame where column team 1 holds the team
        and column team 2 holds the opponent for each team's matches.
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    one_way_duration_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations between all
        points.
    weekend_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight between Friday and Saturday matches.
    distance_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight.
    rooms_per_overnight : int
        The number of rooms required to house all support staff and
        players for an overnight.
    cost_per_room : float
        The cost for one room.
    cost_per_unit_distance : float
        The cost to travel one unit of distance (likely miles).

    Returns
    -------
    analysis_results : dict
        A data dictionary with team schedule DataFrame analysis results.

    """

    # Create analysis dictionaries
    team_distance_dict = create_team_distance_or_duration_dict(team_schedule_df, one_way_distance_matrix, one_way_distance_matrix, weekend_overnight_threshold)
    team_overnight_dict = create_team_overnight_dict(team_schedule_df, one_way_distance_matrix, weekend_overnight_threshold, distance_overnight_threshold)

    team_cost_dict = create_team_cost_dict(team_distance_dict, team_overnight_dict, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

    team_duration_s_dict = create_team_distance_or_duration_dict(team_schedule_df, one_way_duration_matrix, one_way_distance_matrix, weekend_overnight_threshold)
    team_duration_h_dict = {key: convert_seconds_to_rounded_hours(value) for key, value in team_duration_s_dict.items()}

    team_home_game_dict = create_team_home_game_dict(team_schedule_df)

    team_match_location_dict = create_team_match_location_dict(team_schedule_df)

    # Analyze schedule
    total_distance = calculate_total_from_dict(team_distance_dict)

    total_duration_s = calculate_total_from_dict(team_duration_s_dict)
    total_duration_h = convert_seconds_to_rounded_hours(total_duration_s)

    total_overnight = calculate_total_from_dict(team_overnight_dict)

    total_distance_cost = calculate_total_distance_cost(total_distance, cost_per_unit_distance)
    total_overnight_cost = calculate_total_overnight_cost(total_overnight, rooms_per_overnight, cost_per_room)
    total_cost = total_distance_cost + total_overnight_cost

    team_duration_h_min_max_rng = calculate_min_max_rng_from_dict(team_duration_h_dict)
    team_duration_h_avg = np.mean(list(team_duration_h_dict.values()))
    team_duration_h_fairness_index = calculate_fairness_index(list(team_duration_h_dict.values()))

    team_overnight_min_max_rng = calculate_min_max_rng_from_dict(team_overnight_dict)
    team_overnight_avg = np.mean(list(team_overnight_dict.values()))
    team_overnight_fairness_index = calculate_fairness_index(list(team_overnight_dict.values()))

    team_home_game_min_max_rng = calculate_min_max_rng_from_dict(team_home_game_dict)
    team_home_game_avg = np.mean(list(team_home_game_dict.values()))
    team_home_game_fairness_index = calculate_fairness_index(list(team_home_game_dict.values()))

    # Test schedule
    team_opponent_dict = create_team_opponent_dict(team_schedule_df)
    test_success = test_each_team_plays_every_other_team_once(team_schedule_df, team_opponent_dict)

    # Check team schedule does not have date value ==> indicates an optimized team schedule
    if 'Date' not in team_schedule_df:

        # Run additional test
        team_time_slot_dict = create_team_time_slot_dict(team_schedule_df)
        second_test_success = test_each_team_plays_no_more_than_one_match_per_time_slot(team_time_slot_dict)

        # Run additional test
        location_time_slot_dict = create_location_time_slot_dict(team_schedule_df)
        third_test_success = test_each_location_hosts_no_more_than_one_match_per_time_slot(location_time_slot_dict)

        # Update test success ==> True if all tests passed
        test_success = all([test_success, second_test_success, third_test_success])

    # Create data dictionary with analysis results
    analysis_results = {
        'Total Distance': total_distance,
        # 'Total Duration in Seconds': total_duration_s,
        'Total Duration in Hours': total_duration_h,
        'Total Overnight': total_overnight,
        'Total Cost': total_cost,
        'Total Distance Cost': total_distance_cost,
        'Total Overnight Cost': total_overnight_cost,
        'Team Distance Dict': team_distance_dict,
        # 'Team Duration in Seconds Dict': team_duration_s_dict,
        'Team Duration in Hours Dict': team_duration_h_dict,
        'Team Duration in Hours (Min, Max, Rng)': team_duration_h_min_max_rng,
        'Team Duration in Hours Average': team_duration_h_avg,
        'Team Duration in Hours Fairness Index': team_duration_h_fairness_index,
        'Team Overnight Dict': team_overnight_dict,
        'Team Overnight (Min, Max, Rng)': team_overnight_min_max_rng,
        'Team Overnight Average': team_overnight_avg,
        'Team Overnight Fairness Index': team_overnight_fairness_index,
        'Team Cost Dict': team_cost_dict,
        'Team Home Game Dict': team_home_game_dict,
        'Team Home Game (Min, Max, Rng)': team_home_game_min_max_rng,
        # 'Team Home Game Average': team_home_game_avg,
        'Team Home Game Fairness Index': team_home_game_fairness_index,
        'Team Match Location Dict': team_match_location_dict,
        'Test Success': test_success,
    }

    return(analysis_results)


def analyze_team_schedule_from_file(team_schedule_filename, venues_mapping_file, one_way_distance_matrix, one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Analyze a team schedule from a file and return a dictionary with
    analysis results.

    Parameters
    ----------
    team_schedule_filename : str
        A team schedule filename including the path.
    venues_mapping_file : str
        The file path for a file containing venue/team names and unique
        integer venue/team IDs. The file is formatted with one
        venue/team name per line represented as Venue/Team Name,ID
    one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    one_way_duration_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving durations between all
        points.
    weekend_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight between Friday and Saturday matches.
    distance_overnight_threshold : int
        The maximum number of miles to drive one-way without staying
        for an overnight.
    rooms_per_overnight : int
        The number of rooms required to house all support staff and
        players for an overnight.
    cost_per_room : float
        The cost for one room.
    cost_per_unit_distance : float
        The cost to travel one unit of distance (likely miles).

    Returns
    -------
    analysis_results : dict
        A data dictionary with team schedule DataFrame analysis results.

    """

    team_schedule_df = pd.read_csv(team_schedule_filename, index_col=0)

    # Create a dictionary for venue name to venue ID mapping
    name_to_id_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[0]

    # Map school name to school ID - necessary for analysis
    team_schedule_df = team_schedule_df.replace({'Team 1': name_to_id_mapping,
                                                 'Team 2': name_to_id_mapping,
                                                 'Location': name_to_id_mapping})

    analysis_results = analyze_team_schedule_df(team_schedule_df, one_way_distance_matrix, one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

    # Extract schedule ID from filename by removing .csv and path
    schedule_id = team_schedule_filename.split('.')[0].split('/')[-1]

    analysis_results['Schedule ID'] = schedule_id

    return(analysis_results)


def create_analysis_df(analysis_results):
    """
    Create a schedule and analysis results DataFrame containing
    schedule-related information and analysis results.

    Parameters
    ----------
    analysis_results : dict
        A data dictionary with schedule-related information and
        analysis results.

    Returns
    -------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.

    """

    analysis_df = pd.DataFrame(analysis_results)

    analysis_df = analysis_df.sort_values('Total Cost', ascending=True)
    analysis_df = analysis_df.reset_index(drop=True)

    end_column_order = [
        'Total Distance',
        # 'Total Duration in Seconds',
        'Total Duration in Hours',
        'Total Overnight',
        'Total Cost',
        'Total Distance Cost',
        'Total Overnight Cost',
        'Team Distance Dict',
        # 'Team Duration in Seconds Dict',
        'Team Duration in Hours Dict',
        'Team Duration in Hours (Min, Max, Rng)',
        'Team Duration in Hours Average',
        'Team Duration in Hours Fairness Index',
        'Team Overnight Dict',
        'Team Overnight (Min, Max, Rng)',
        'Team Overnight Average',
        'Team Overnight Fairness Index',
        'Team Cost Dict',
        'Team Home Game Dict',
        'Team Home Game (Min, Max, Rng)',
        # 'Team Home Game Average',
        'Team Home Game Fairness Index',
        'Team Match Location Dict',
        'Test Success']

    if 'Clustering ID' in analysis_results:

        column_order = [
            'Schedule ID',
            'Home Game Constraint Combo',
            'Clustering ID',
            'Matches List']

        column_order.extend(end_column_order)

    else:

        column_order = [
            'Schedule ID']

        column_order.extend(end_column_order)

    analysis_df = analysis_df[column_order]

    return(analysis_df)


########################
# SCHEDULE PAIRS/QUADS #
########################


def create_schedule_combo_list(schedule_analysis_df, num_schedules, temporal):
    """
    Create a list of possible schedule combinations with each
    combination containing the specified number of schedules.

    Parameters
    ----------
    schedule_analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule and analysis information including
        a schedule ID column.
    num_schedules : int
        Number of schedule indices to include in each schedule
        combination.
    temporal : bool
        True to create schedule combinations that account for time (use
        this for previous NESCAC schedules because want to compare 2009
        to 2010, 2010 to 2011, ...). Assumes sorting schedules by
        schedule ID ascending will put the schedules in order
        temporally. False to create all possible schedule combinations
        without considering time (use this for optimized schedules
        because want to compare schedule 1 to 2, 1 to 3, ...).

    Returns
    -------
    idx_combos : list
        A list of schedule combinations. Each schedule combination
        contains the specified number of schedule indices.

    """

    # Create list of index combinations where time matters
    if temporal:

        # Sort DataFrame by Schedule ID ascending and therefore by time since schedule ID = YYYY_...
        tmp = schedule_analysis_df.copy()
        tmp = tmp.sort_values(['Schedule ID'], ascending=[True]).reset_index()

        idx_combos = []

        # Iterate over all schedules excluding the last few accounted for in the idx combo construction
        for idx in tmp.index[:(num_schedules * -1) + 1]:

            idx_combo = tuple(tmp['index'][idx + i] for i in range(num_schedules))

            idx_combos.append(idx_combo)

    # Create list of index combinations where time does not matter
    else:

        idx_combos = list(itertools.combinations(schedule_analysis_df.index, num_schedules))

    return(idx_combos)


##################################
# Location Analysis Dictionaries #
##################################

def create_team_unique_and_different_location_dicts(schedule_analysis_df, idx_combo):

    # Create list of team match location dicts for schedules in the index combination
    team_match_location_dicts = [schedule_analysis_df['Team Match Location Dict'][idx] for idx in idx_combo]

    # Initialize team unique location dict as {team: number of unique locations}
    team_unique_location_dict = {}

    # Initialize team different location dict as {team: [number of different locations between each pair of schedules]}
    team_different_location_dict = {team: [] for team in team_match_location_dicts[0]}

    for team in team_match_location_dicts[0]:

        # Calculate the number of unique match locations
        team_unique_location_dict[team] = len(set([location for di in team_match_location_dicts for location in di[team]]))

        # Iterate over all team match location dicts excluding the last one accounted for in the different location score calculation
        for i in range(len(team_match_location_dicts) - 1):

            # Calculate the number of different match locations between pair of schedules
            different_location_score = sum(1 for location in team_match_location_dicts[i][team] if location not in team_match_location_dicts[i + 1][team]) + sum(1 for location in team_match_location_dicts[i + 1][team] if location not in team_match_location_dicts[i][team])

            # Append number of different locations dict
            team_different_location_dict[team].append(different_location_score)

    return(team_unique_location_dict, team_different_location_dict)


#############################################
# Full Schedule Pair/Quad Location Analysis #
#############################################

def analyze_team_schedule_df_locations(schedule_analysis_df, idx_combo):
    """
    Analyze a team schedule DataFrame and return a dictionary with
    location analysis results.
    """

    # Create analysis dictionaries
    team_unique_location_dict, team_different_location_dict = create_team_unique_and_different_location_dicts(schedule_analysis_df, idx_combo)

    # Extract list of different location scores and unique location scores for all teams
    different_location_scores = [el for value_list in team_different_location_dict.values() for el in value_list]
    unique_location_scores = list(team_unique_location_dict.values())

    team_different_location_avg = np.mean(different_location_scores)
    team_different_location_fairness_index = calculate_fairness_index(different_location_scores)

    team_unique_location_avg = np.mean(unique_location_scores)
    team_unique_location_fairness_index = calculate_fairness_index(unique_location_scores)

    location_analysis_results = {
        'Schedule Indexes': idx_combo,
        'Team Different Location Dict': team_different_location_dict,
        'Team Different Location Average': team_different_location_avg,
        'Team Different Location Fairness Index': team_different_location_fairness_index,
        'Team Unique Location Dict': team_unique_location_dict,
        'Team Unique Location Average': team_unique_location_avg,
        'Team Unique Location Fairness Index': team_unique_location_fairness_index,
    }

    return(location_analysis_results)


def average_specified_analysis_results(results, values_to_consider):
    """
    Average (where applicable) the specified values in a schedule and
    analysis results dictionary.

    Parameters
    ----------
    results : dict
        A data dictionary with schedule and analysis results as {key:
        [list of values]}.
    values_to_consider : list
        A list of data points in the analysis results dictionary to
        average (where applicable) for the average results dictionary.

    Returns
    -------
    average_results : dict
        A data dictionary with the average (where applicable) of
        specified values in a schedule and analysis results
        dictionary as {key: average value}.

    """

    # Initialize data dictionary to hold average values
    average_results = {}

    # Empty results dictionary
    if not results:

        # Return empty average results dictionary
        return(average_results)

    # Iterate over values to consider
    for key in values_to_consider:

        if key in ['Clustering ID', 'Schedule ID', 'Home Game Constraint Combo']:

            average_results[key] = results[key][0]

        elif '(Min, Max, Rng)' in key:

            # Calculate the average of each value in the tuple
            average_results[key] = tuple([(sum(values) / len(values)) for values in zip(*results[key])])

        elif 'Test Success' in key:

            average_results[key] = all(results[key])

        else:

            average_results[key] = np.mean(results[key])

    return(average_results)


def sum_specified_analysis_results(results, values_to_consider):
    """
    Sum (where applicable) the specified values in a schedule and
    analysis results dictionary.

    Parameters
    ----------
    results : dict
        A data dictionary with schedule and analysis results as {key:
        [list of values]}.
    values_to_consider : list
        A list of data points in the analysis results dictionary to
        sum (where applicable) for the sum results dictionary.

    Returns
    -------
    sum_results : dict
        A data dictionary with the sum (where applicable) of
        specified values in a schedule and analysis results
        dictionary as {key: sum of values}.

    """

    # Initialize data dictionary to hold sum values
    sum_results = {}

    # Empty results dictionary
    if not results:

        # Return empty sum results dictionary
        return(sum_results)

    # Iterate over values to consider
    for key in values_to_consider:

        if key in ['Clustering ID', 'Schedule ID', 'Home Game Constraint Combo']:

            sum_results[key] = results[key][0]

        elif '(Min, Max, Rng)' in key:

            # Calculate the sum of each value in the tuple
            sum_results[key] = tuple([sum(values) for values in zip(*results[key])])

        elif 'Test Success' in key:

            sum_results[key] = all(results[key])

        else:

            sum_results[key] = sum(results[key])

    return(sum_results)


def create_location_analysis_df(schedule_analysis_df, num_schedules, temporal, values_to_consider):
    """
    Create a DataFrame with schedule information, average or sum
    analysis results for all combinations of schedules with each
    combination containing the specified number of schedules.

    Parameters
    ----------
    schedule_analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule and analysis information including
        a schedule ID column.
    num_schedules : int
        Number of schedule indices to include in each schedule
        combination.
    temporal : bool
        True to create schedule combinations that account for time (use
        this for previous NESCAC schedules because want to compare 2009
        to 2010, 2010 to 2011, ...). Assumes sorting schedules by
        schedule ID ascending will put the schedules in order
        temporally. False to create all possible schedule combinations
        without considering time (use this for optimized schedules
        because want to compare schedule 1 to 2, 1 to 3, ...).
    values_to_consider : list
        A list of data points to average (where applicable) or sum
        (where applicable) for the results DataFrame. Data points that
        contain "Total" but do not contain "Score" are summed.
        Remaining numeric data points are averaged.

    Returns
    -------
    schedule_combo_location_score_df : pandas.core.frame.DataFrame
        A DataFrame with average and sum analysis information for
        each schedule combination.

    """

    data = []

    idx_combos = create_schedule_combo_list(schedule_analysis_df, num_schedules, temporal)

    for idx_combo in idx_combos:

        combo_data = {}

        # Extract data for schedules in the index combination
        analysis_data = schedule_analysis_df.loc[list(idx_combo)].to_dict(orient="list")

        # Determine which values to sum and which to average
        values_to_sum = [each for each in values_to_consider if 'Total' in each and 'Score' not in each]
        values_to_avg = [each for each in values_to_consider if each not in values_to_sum]

        # Calculate average data for schedules and add to combo data dict
        avg_analysis_data = average_specified_analysis_results(analysis_data, values_to_avg)
        for key, value in avg_analysis_data.items():
            combo_data["(Average) " + key] = value

        # Calculate sum data for schedules and add to combo data dict
        sum_analysis_data = sum_specified_analysis_results(analysis_data, values_to_sum)
        for key, value in sum_analysis_data.items():
            combo_data["(Sum) " + key] = value

        location_analysis_results = analyze_team_schedule_df_locations(schedule_analysis_df, idx_combo)

        # Add location score data to combo data dict
        for key, value in location_analysis_results.items():
            combo_data[key] = value

        # Add schedule ID data to combo data dict
        for i in range(len(idx_combo)):
            combo_data['Schedule ID {}'.format(i)] = schedule_analysis_df['Schedule ID'][idx_combo[i]]

        data.append(combo_data)

    # Create DataFrame of results
    location_analysis_df = pd.DataFrame(data)
    location_analysis_df = location_analysis_df.sort_values('(Sum) Total Cost', ascending=True).reset_index(drop=True)

    return(location_analysis_df)


###################
# SCORE SCHEDULES #
###################

def create_rescaled_columns(analysis_df, col_names):
    """
    Rescale column values to a [0, 100] range where:

        x_rescaled = 100 * ( (x - min(x)) / (max(x) - min(x)) )

    """
    for col in col_names:

        new_col = col + ' Rescaled'

        # Catch divide by zero
        if (analysis_df[col].max() - analysis_df[col].min()) == 0:

            analysis_df[new_col] = 0

        else:
            analysis_df[new_col] = 100 * ((analysis_df[col] - analysis_df[col].min()) / (analysis_df[col].max() - analysis_df[col].min()))

    return(analysis_df)


def create_inverse_columns(analysis_df, col_names, col_prefix):
    """
    Calculate (100 - col_values) for specified column.
    """
    for col_name in col_names:

        col_name = col_prefix + col_name
        analysis_df['Inverse ' + col_name] = 100 - analysis_df[col_name]

    return(analysis_df)


def create_score_columns(analysis_df, score_total_cost_weight):
    """
    Calculate the team experience and schedule score for each single
    schedule. Use features rescaled to a [0, 100] range in the score
    calculation where:

        x_rescaled = 100 * ( (x - min(x)) / (max(x) - min(x)) )

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results including the columns total cost, team duration in
        hours fairness index and average, team home game fairness
        index, and team overnight fairness index and average.
    schedule_score_total_cost_weight : float
        Amount as a decimal percent to weight the total cost in the
        schedule score calculation.

    Returns
    -------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results plus the additional columns:

            Team Duration in Hours Average Rescaled
            Team Duration in Hours Fairness Index Rescaled
            Team Home Game Fairness Index Rescaled
            Team Overnight Average Rescaled
            Team Overnight Fairness Index Rescaled
            Total Cost Rescaled

            Inverse Team Duration in Hours Fairness Index Rescaled
            Inverse Team Home Game Fairness Index Rescaled
            Inverse Team Overnight Fairness Index Rescaled

            Team Duration in Hours Score
            Team Home Game Score
            Team Overnight Score

            Team Experience Score
            Schedule Score

    """

    # Columns in single and pair/quad schedules to rescale or inverse
    rescale_cols = [
        'Team Home Game Fairness Index',
        'Team Duration in Hours Average',
        'Team Duration in Hours Fairness Index',
        'Team Overnight Average',
        'Team Overnight Fairness Index'
    ]

    inverse_cols =[
        'Team Duration in Hours Fairness Index Rescaled',
        'Team Overnight Fairness Index Rescaled',
        'Team Home Game Fairness Index Rescaled'
    ]

    # Check if pair/quad schedule analysis
    if '(Sum) Total Cost' in analysis_df.columns:

        # Columns in only pair/quad schedules to rescale or inverse
        pq_rescale_cols = [
            '(Sum) Total Cost',
            'Team Different Location Average',
            'Team Different Location Fairness Index',
            'Team Unique Location Average',
            'Team Unique Location Fairness Index'
        ]

        pq_inverse_cols = [
            'Team Different Location Average Rescaled',
            'Team Different Location Fairness Index Rescaled',
            'Team Unique Location Average Rescaled',
            'Team Unique Location Fairness Index Rescaled'
        ]

        # Rescale cols to a [0, 100] range
        analysis_df = create_rescaled_columns(
            analysis_df=analysis_df,
            col_names=pq_rescale_cols+['(Average) ' + col for col in rescale_cols])

        # Calculate inverse col values
        analysis_df = create_inverse_columns(
            analysis_df=analysis_df,
            col_names=inverse_cols,
            col_prefix='(Average) ')

        analysis_df = create_inverse_columns(
            analysis_df=analysis_df,
            col_names=pq_inverse_cols,
            col_prefix='')

        # Calculate team different location score
        # Want to maximize fairness index (minimize inverse fairness
        # index) & maximize team different location average rescaled
        # (minimize inverse team different location average rescaled)
        analysis_df['Team Different Location Score'] = analysis_df[[
            'Inverse Team Different Location Fairness Index Rescaled',
            'Inverse Team Different Location Average Rescaled']].mean(axis=1)

        # Calculate team unique location score
        # Want to maximize fairness index (minimize inverse fairness
        # index) & maximize team unique location average rescaled
        # (minimize inverse team unique location average rescaled)
        analysis_df['Team Unique Location Score'] = analysis_df[[
            'Inverse Team Unique Location Fairness Index Rescaled',
            'Inverse Team Unique Location Average Rescaled']].mean(axis=1)

        # Calculate location score as mean of location score
        # components
        analysis_df['Location Score'] = analysis_df[[
            'Inverse Team Different Location Fairness Index Rescaled',
            'Inverse Team Different Location Average Rescaled',
            'Inverse Team Unique Location Fairness Index Rescaled',
            'Inverse Team Unique Location Average Rescaled']].mean(axis=1)

        # Calculate pair/quad score as weighted mean of total cost and
        # team experience and location score components
        analysis_df['Pair/Quad Score'] = score_total_cost_weight * analysis_df['(Sum) Total Cost Rescaled'] + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse (Average) Team Duration in Hours Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['(Average) Team Duration in Hours Average Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse (Average) Team Overnight Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['(Average) Team Overnight Average Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse (Average) Team Home Game Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse Team Different Location Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse Team Different Location Average Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse Team Unique Location Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 9) * analysis_df['Inverse Team Unique Location Average Rescaled'])
    else:

        # Rescale cols to a [0, 100] range
        analysis_df = create_rescaled_columns(
            analysis_df=analysis_df,
            col_names=['Total Cost']+rescale_cols)

        # Calculate inverse col values
        analysis_df = create_inverse_columns(
            analysis_df=analysis_df,
            col_names=inverse_cols,
            col_prefix='')

        # Calculate team duration in hours score
        # Want to maximize fairness index (minimize inverse fairness
        # index) & minimize team duration average
        analysis_df['Team Duration in Hours Score'] = analysis_df[[
            'Inverse Team Duration in Hours Fairness Index Rescaled',
            'Team Duration in Hours Average Rescaled']].mean(axis=1)

        # Calculate team overnight score
        # Want to maximize fairness index (minimize inverse fairness
        # index) & minimize team overnight average
        analysis_df['Team Overnight Score'] = analysis_df[[
            'Inverse Team Overnight Fairness Index Rescaled',
            'Team Overnight Average Rescaled']].mean(axis=1)

        # Calculate team home game score
        # Want to maximize fairness index (minimize inverse fairness
        # index)
        analysis_df['Team Home Game Score'] = analysis_df['Inverse Team Home Game Fairness Index Rescaled']

        # Calculate team experience score as mean of team score
        # components
        analysis_df['Team Experience Score'] = analysis_df[[
            'Inverse Team Duration in Hours Fairness Index Rescaled',
            'Team Duration in Hours Average Rescaled',
            'Inverse Team Overnight Fairness Index Rescaled',
            'Team Overnight Average Rescaled',
            'Inverse Team Home Game Fairness Index Rescaled']].mean(axis=1)

        # Calculate schedule score as weighted mean of total cost and
        # team experience score components
        analysis_df['Schedule Score'] = (score_total_cost_weight * analysis_df['Total Cost Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 5) * analysis_df['Inverse Team Duration in Hours Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 5) * analysis_df['Team Duration in Hours Average Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 5) * analysis_df['Inverse Team Overnight Fairness Index Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 5) * analysis_df['Team Overnight Average Rescaled']) + \
            ((1 - score_total_cost_weight) * (1 / 5) * analysis_df['Inverse Team Home Game Fairness Index Rescaled'])

    return(analysis_df)


#########################################
# COMPARE OPTIMIZED TO NESCAC SCHEDULES #
#########################################

def calculate_column_value_change_optimized_nescac(optimized_row, nescac_row, col_name, col_prefix):
    """
    For a specified column, calculate the total value change between an optimized row and a nescac row. Define the col_prefix so function works with single and pair/quad schedules.
    """
    optimized_value = optimized_row[col_prefix + col_name]
    nescac_value = nescac_row[col_prefix + col_name]

    return(optimized_value - nescac_value)


def calculate_column_percent_change_optimized_nescac(optimized_row, nescac_row, col_name, col_prefix):
    """
    For a specified column, calculate the total precent change between an optimized row and a nescac row. Define the col_prefix so function works with single and pair/quad schedules.
    """
    optimized_value = optimized_row[col_prefix + col_name]
    nescac_value = nescac_row[col_prefix + col_name]

    return(100 * ((optimized_value - nescac_value) / nescac_value))


def create_optimized_change_columns(analysis_df):
    """
    Calculate value change and percent change for total cost, total duration in hours, and total overnight columns.
    """

    # Adjust column names based on single or pair/quad schedules
    if 'Pair/Quad Score' in analysis_df.columns:
        id_col_suffix = ' 0'
        total_col_prefix = '(Sum) '
    else:
        id_col_suffix = ''
        total_col_prefix = ''

    # Extract optimized schedules with correct scores
    optimized_schedule_analysis_df = analysis_df[analysis_df['Schedule ID' + id_col_suffix].str.contains('optimized')]

    # Extract nescac schedules with correct scores
    nescac_schedule_analysis_df = analysis_df[analysis_df['Schedule ID' + id_col_suffix].str.contains('nescac')]

    # Prevent setting with copy warning
    nescac_schedule_analysis_df = nescac_schedule_analysis_df.copy()

    # Add optimized change columns
    for opt_idx, opt_row in optimized_schedule_analysis_df.iterrows():

        opt_abbrev_sched_id = opt_row['Abbreviated Schedule ID']

        for nes_idx, nes_row in nescac_schedule_analysis_df.iterrows():

            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Cost Change".format(opt_abbrev_sched_id)] = calculate_column_value_change_optimized_nescac(opt_row, nes_row, 'Total Cost', total_col_prefix)
            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Cost Percent Change".format(opt_abbrev_sched_id)] = calculate_column_percent_change_optimized_nescac(opt_row, nes_row, 'Total Cost', total_col_prefix)

            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Duration in Hours Change".format(opt_abbrev_sched_id)] = calculate_column_value_change_optimized_nescac(opt_row, nes_row, 'Total Duration in Hours', total_col_prefix)
            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Duration in Hours Percent Change".format(opt_abbrev_sched_id)] = calculate_column_percent_change_optimized_nescac(opt_row, nes_row, 'Total Duration in Hours', total_col_prefix)

            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Overnight Change".format(opt_abbrev_sched_id)] = calculate_column_value_change_optimized_nescac(opt_row, nes_row, 'Total Overnight', total_col_prefix)
            nescac_schedule_analysis_df.loc[nes_idx, "{} Total Overnight Percent Change".format(opt_abbrev_sched_id)] = calculate_column_percent_change_optimized_nescac(opt_row, nes_row, 'Total Overnight', total_col_prefix)

    # Recombine into one DataFrame
    analysis_df = optimized_schedule_analysis_df.append(nescac_schedule_analysis_df)

    return(analysis_df)