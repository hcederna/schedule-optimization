""" Various helper functions to survey venue clusterings and home game
constraint parameters. """
from itertools import combinations_with_replacement

import pandas as pd

from src.analyze_schedule import analyze_team_schedule_df
from src.analyze_schedule import average_specified_analysis_results
from src.analyze_schedule import create_score_columns
from src.collect_data import create_one_way_distance_matrix_and_duration_matrix
from src.collect_data import determine_centroid_distance_matrix_filename
from src.helpers import create_team_schedule_df
from src.optimize_schedule import create_schedule_df
from src.optimize_schedule import find_optimized_schedule


def optimize_and_analyze_schedule_specified_num_times(num_times, max_acceptable_home_game_rng, num_weeks, clustering_information, home_game_constraint_combo, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Optimize and analyze the NESCAC volleyball conference schedule a
    specified number of times. Create a dictionary with schedule and
    analysis results. Include only schedules that are optimal and
    within the maximum acceptable home game range.

    Parameters
    ----------
    num_times : int
        The maximum number of times to optimize and analyze the
        schedule.
    max_acceptable_home_game_rng : int
        The maximum acceptable home game range.
    num_weeks : int
        The number of weeks in the season.
    clustering_information : pandas.core.series.Series
        A row from the clusterings information DataFrame that contains
        clustering source(s), information, and analysis results.
    home_game_constraint_combo : tuple
        A tuple of integers as (a, b, c, d) where:
            a = minimum acceptable number of home games for groups with
            one team

            b = maximum acceptable number of home games for groups
            with one team

            c = minimum acceptable number of home games for groups with
            two or more teams

            d = maximum acceptable number of home games for groups
            with two or more teams
    venue_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    venue_one_way_duration_matrix : pandas.core.frame.DataFrame
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
    results : dict
        A data dictionary with schedule and analysis results as {key:
        [list of values]}.

    """

    # Initialize data dictionary to hold values
    results = {}

    # Extract relevant information about clustering
    clustering_id = clustering_information['Clustering ID']
    cluster_membership_dict = clustering_information['Cluster Membership Dict']
    centroid_coordinates = clustering_information['Centroid Coordinates']

    # Determine filename for centroid distance matrix
    centroid_distance_matrix_filename = determine_centroid_distance_matrix_filename(clustering_id)

    try:

        # Load centroid one-way distance matrix
        centroid_one_way_distance_matrix = pd.read_csv(centroid_distance_matrix_filename, index_col=0)

    except Exception:

        # Query Google Maps API to create centroid one-way distance matrix
        centroid_one_way_distance_matrix = create_one_way_distance_matrix_and_duration_matrix(
            coordinates_array=centroid_coordinates)[0]

        # Save centroid one-way distance matrix
        centroid_one_way_distance_matrix.to_csv(centroid_distance_matrix_filename)

    for i in range(num_times):

        optimization_status, schedule = find_optimized_schedule(num_weeks, cluster_membership_dict, home_game_constraint_combo, centroid_one_way_distance_matrix)

        if optimization_status != "Optimal":

            continue

        schedule_df = create_schedule_df(schedule)
        team_schedule_df = create_team_schedule_df(schedule_df)

        analysis_results = analyze_team_schedule_df(team_schedule_df, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

        if analysis_results['Team Home Game (Min, Max, Rng)'][2] > max_acceptable_home_game_rng:

            continue

        # Add additional values to analysis results
        analysis_results['Home Game Constraint Combo'] = home_game_constraint_combo
        analysis_results['Clustering ID'] = clustering_id
        analysis_results['Matches List'] = schedule

        # Store results as dictionary of lists
        for key, value in analysis_results.items():

            try:

                results[key].append(value)

            except KeyError:

                results[key] = [value]

    return(results)


def create_home_game_constraint_combinations_list(min_acceptable_one_team=2, max_acceptable_one_team=5, min_acceptable_multi_team=1, max_acceptable_multi_team=5):
    """
    Create a list of possible home game constraint combinations.

    Parameters
    ----------
    min_acceptable_one_team : int (default = 2)
        The minimum acceptable number of home games for groups with
        one team.
    max_acceptable_one_team : int (default = 5)
        The maximum acceptable number of home games for groups with
        one team.
    min_acceptable_multi_team : int (default = 1)
        The minimum acceptable number of home games for groups with
        two or more teams.
    max_acceptable_multi_team : int (default = 5)
        The maximum acceptable number of home games for groups with
        two or more teams.

    Returns
    -------
    home_game_constraint_combo : tuple
        A tuple of integers as (a, b, c, d) where:
            a = minimum acceptable number of home games for groups with
            one team

            b = maximum acceptable number of home games for groups
            with one team

            c = minimum acceptable number of home games for groups with
            two or more teams

            d = maximum acceptable number of home games for groups
            with two or more teams

    """

    # Create list of acceptable values for one team and multi team
    acceptable_one_team_values = list(range(min_acceptable_one_team, max_acceptable_one_team + 1))
    acceptable_multi_team_values = list(range(min_acceptable_multi_team, max_acceptable_multi_team + 1))

    # Create list of (min, max) combos for one team and multi team
    one_team_combos = list(combinations_with_replacement(acceptable_one_team_values, 2))
    multi_team_combos = list(combinations_with_replacement(acceptable_multi_team_values, 2))

    # Create list of (min_one_team, max_one_team, min_multi_team, max_multi_team) combos
    constraint_combos = [(one[0], one[1], multi[0], multi[1]) for one in one_team_combos for multi in multi_team_combos]

    return(constraint_combos)


def survey_home_game_constraint_combinations(min_acceptable_one_team, max_acceptable_one_team, min_acceptable_multi_team, max_acceptable_multi_team, values_to_consider, num_times, max_acceptable_home_game_rng, num_weeks, clustering_information, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Create a DataFrame with the average results from optimizing and
    analyzing a specified number of schedules for each home game
    constraint combination. The average results include only schedules
    that are optimal and within the maximum acceptable home game range.

    Parameters
    ----------
    min_acceptable_one_team : int (default = 2)
        The minimum acceptable number of home games for groups with
        one team.
    max_acceptable_one_team : int (default = 5)
        The maximum acceptable number of home games for groups with
        one team.
    min_acceptable_multi_team : int (default = 1)
        The minimum acceptable number of home games for groups with
        two or more teams.
    max_acceptable_multi_team : int (default = 5)
        The minimum acceptable number of home games for groups with
        two or more teams.
    values_to_consider : list
        A list of data points in the analysis results dictionary to
        average (where applicable) for the average results dictionary.
    num_times : int
        The maximum number of times to optimize and analyze the
        schedule for each home game constraint combination.
    max_acceptable_home_game_rng : int
        The maximum acceptable home game range.
    num_weeks : int
        The number of weeks in the season.
    clustering_information : pandas.core.series.Series
        A row from the clusterings information DataFrame that contains
        clustering source(s), information, and analysis results.
    home_game_constraint_combo : tuple
        A tuple of integers as (a, b, c, d) where:
            a = minimum acceptable number of home games for groups with
            one team

            b = maximum acceptable number of home games for groups
            with one team

            c = minimum acceptable number of home games for groups with
            two or more teams

            d = maximum acceptable number of home games for groups
            with two or more teams
    venue_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    venue_one_way_duration_matrix : pandas.core.frame.DataFrame
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
    survey_results : dict
        A data dictionary with the average (where applicable) schedule
        and analysis results for each home game constraint
        combination.

    """

    # Initialize data dictionary to hold values
    survey_results = {}

    # Create list of home game constraint combinations
    constraint_combos = create_home_game_constraint_combinations_list(min_acceptable_one_team, max_acceptable_one_team, min_acceptable_multi_team, max_acceptable_multi_team)

    for home_game_constraint_combo in constraint_combos:

        results = optimize_and_analyze_schedule_specified_num_times(num_times, max_acceptable_home_game_rng, num_weeks, clustering_information, home_game_constraint_combo, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

        average_results = average_specified_analysis_results(results, values_to_consider)

        # Store survey results as dictionary of lists
        for key, value in average_results.items():

            try:

                survey_results[key].append(value)

            except KeyError:

                survey_results[key] = [value]

    return(survey_results)


def drop_duplicate_results_keep_the_best(df, duplicate_metric, schedule_score_total_cost_weight):
    """
    Drop duplicate results based on the duplicate metric. Keep the best
    of the duplicates with best defined as the result with the lowest
    schedule score.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame with a duplicate metric column with potential
        duplicate values and a schedule score column.
    duplicate_metric : str
        The name of the column to check for duplicate values.
    schedule_score_total_cost_weight : float
        Amount as a decimal percent to weight the total cost in the
        schedule score calculation.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        A DataFrame with a duplicate metric column with potential
        duplicate values removed and a schedule score column.

    """

    # Temporarily add rescaled and score columns
    cols_before = list(df.columns)
    df = create_score_columns(df, schedule_score_total_cost_weight)

    # Sort DataFrame on schedule score (best first)
    df = df.sort_values('Schedule Score', ascending=True)

    # Drop duplicate results (keep the best of the duplicates)
    df = df.drop_duplicates(duplicate_metric, keep='first')
    df = df[cols_before]

    return(df)


def survey_clusterings_and_home_game_constraint_combinations(duplicate_metric, schedule_score_total_cost_weight, num_to_keep, min_acceptable_one_team, max_acceptable_one_team, min_acceptable_multi_team, max_acceptable_multi_team, values_to_consider, num_times, max_acceptable_home_game_rng, num_weeks, clusterings_information_df, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Create a DataFrame with the best specified number of non-duplicate
    home game constraint combination survey results for each clustering
    with best defined as the result with the lowest schedule score.

    Parameters
    ----------
    duplicate_metric : str
        The name of the column to check for duplicate values.
    schedule_score_total_cost_weight : float
        Amount as a decimal percent to weight the total cost in the
        schedule score calculation.
    num_to_keep : int
        The number of home game constraint combination results to
        keep for each clustering.
    min_acceptable_one_team : int (default = 2)
        The minimum acceptable number of home games for groups with
        one team.
    max_acceptable_one_team : int (default = 5)
        The maximum acceptable number of home games for groups with
        one team.
    min_acceptable_multi_team : int (default = 1)
        The minimum acceptable number of home games for groups with
        two or more teams.
    max_acceptable_multi_team : int (default = 5)
        The minimum acceptable number of home games for groups with
        two or more teams.
    values_to_consider : list
        A list of data points in the analysis results dictionary to
        average (where applicable) for the average results dictionary.
    num_times : int
        The maximum number of times to optimize and analyze the
        schedule for each home game constraint combination.
    max_acceptable_home_game_rng : int
        The maximum acceptable home game range.
    num_weeks : int
        The number of weeks in the season.
    clusterings_information_df : pandas.core.frame.DataFrame
        A clusterings information DataFrame containing clustering
        source(s), information, and analysis results.
    home_game_constraint_combo : tuple
        A tuple of integers as (a, b, c, d) where:
            a = minimum acceptable number of home games for groups with
            one team

            b = maximum acceptable number of home games for groups
            with one team

            c = minimum acceptable number of home games for groups with
            two or more teams

            d = maximum acceptable number of home games for groups
            with two or more teams
    venue_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all points.
    venue_one_way_duration_matrix : pandas.core.frame.DataFrame
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
    survey_results_df : pandas.core.frame.DataFrame
        A DataFrame with the best specified number of home game
        constraint combination survey results for each clustering with
        best defined as the result with the lowest weighted total score.

    """

    # Initialize DataFrame to hold values
    survey_results_df = pd.DataFrame()

    # Iterate over clusterings
    for idx, clustering_information in clusterings_information_df.iterrows():

        print("... Surveying home game constraint combinations for clustering {} ...".format(clustering_information['Clustering ID']))

        hg_survey_results = survey_home_game_constraint_combinations(min_acceptable_one_team, max_acceptable_one_team, min_acceptable_multi_team, max_acceptable_multi_team, values_to_consider, num_times, max_acceptable_home_game_rng, num_weeks, clustering_information, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

        if not hg_survey_results:

            print("... No suitable results found.\n")
            continue

        else:

            # Create DataFrame from HG survey results
            hg_survey_results_df = pd.DataFrame(hg_survey_results)

            # Drop duplicate results (keep the best of the duplicates)
            hg_survey_results_df = drop_duplicate_results_keep_the_best(hg_survey_results_df, duplicate_metric, schedule_score_total_cost_weight)

            # Append specified number of results to survey results DataFrame
            survey_results_df = survey_results_df.append(hg_survey_results_df.head(num_to_keep))

            print("... {} suitable result(s) found.\n".format(len(hg_survey_results_df)))

    if survey_results_df.empty:

        print("No suitable survey results found. Change acceptable home game threshold and try again.\n")

    else:

        # Add rescaled and score columns to full survey results DataFrame
        survey_results_df = create_score_columns(survey_results_df, schedule_score_total_cost_weight)
        survey_results_df = survey_results_df.sort_values('Schedule Score', ascending=True).reset_index(drop=True)

    return(survey_results_df)