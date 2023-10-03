""" Various helper functions to create and save the best optimized
NESCAC volleyball conference schedules. """
import pandas as pd

from src.analyze_schedule import create_score_columns
from src.helpers import create_team_schedule_df
from src.helpers import create_venue_id_mapping_dicts_from_file
from src.optimize_schedule import create_schedule_df
from src.survey_parameters import drop_duplicate_results_keep_the_best
from src.survey_parameters import optimize_and_analyze_schedule_specified_num_times


##############################
# Create Optimized Schedules #
##############################

def add_optimized_schedule_id_column(df):
    """
    Create a schedule ID column for the optimized schedules.

    Example Schedule IDs
    --------------------
    1_2315_optimized_team_schedule_0
    2_2214_optimized_team_schedule_1

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame with optimized schedules information including
        clustering ID and home game constraint combo columns.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        A DataFrame with optimized schedules information including
        clustering ID, home game constraint combo, and an added
        schedule ID column.

    """

    df = df.reset_index(drop=True)
    df['Schedule ID'] = df.index

    df['Schedule ID'] = df['Schedule ID'].apply(lambda idx: str(df['Clustering ID'][idx]) + "_" + "".join(str(val) for val in df['Home Game Constraint Combo'][idx]) + "_optimized_team_schedule_" + str(idx))

    return(df)


def drop_duplicate_results_within_hg_groups_keep_the_best(df, duplicate_metric, schedule_score_total_cost_weight):
    """
    Group the DataFrame by team home game (min, max, rng) and drop
    duplicate results based on the duplicate metric in each group. Keep
    the best of the duplicates in each group with best defined as the
    result with the lowest schedule score.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame with a duplicate metric column with potential
        duplicate values and team home game (min, max, rng) and
        schedule score columns.
    duplicate_metric : str
        The name of the column to check for duplicate values.
    schedule_score_total_cost_weight : float
        Amount as a decimal percent to weight the total cost in the
        schedule score calculation.

    Returns
    -------
    no_duplicates_df : pandas.core.frame.DataFrame
        A DataFrame with a duplicate metric column with potential
        duplicate values removed within each team home game (min, max,
        rng) group.

    """

    # Group DataFrame by the number of home games
    hg_grouped = df.groupby('Team Home Game (Min, Max, Rng)')

    # Initialize DataFrame to hold non_duplicate values
    no_duplicates_df = pd.DataFrame()

    for name, tmp_hg_group_df in hg_grouped:

        # Create copy of HG group DataFrame to prevent SettingWithCopyWarning
        hg_group_df = tmp_hg_group_df.copy()

        # Drop duplicate results in current home game group (keep the best of the duplicates)
        hg_group_df = drop_duplicate_results_keep_the_best(hg_group_df, duplicate_metric, schedule_score_total_cost_weight)

        # Append home game group without duplicates to no duplicates DataFrame
        no_duplicates_df = no_duplicates_df.append(hg_group_df)

    return(no_duplicates_df)


def create_and_analyze_best_schedules(best_survey_results_df, duplicate_metric, schedule_score_total_cost_weight, num_to_keep, num_times, max_acceptable_home_game_rng, num_weeks, clusterings_information_df, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance):
    """
    Create a DataFrame with the best specified number of schedules for
    each of the results in the best parameter survey results DataFrame.
    Best schedules defined as the schedules with the lowest schedule
    score.

    Parameters
    ----------
    best_survey_results_df : pandas.core.frame.DataFrame
        A DataFrame with the best specified number of home game
        constraint combination survey results for each of the specified
        clustering IDs with best defined as the result with the lowest
        weighted total score.
    duplicate_metric : str
        The name of the column to check for duplicate values.
    schedule_score_total_cost_weight : float
        Amount as a decimal percent to weight the total cost in the
        schedule score calculation.
    num_to_keep : int
        The number of schedules with analysis results to keep for
        each best parameter survey result.
    num_times : int
        The maximum number of times to optimize and analyze the
        schedule for each parameter setting.
    max_acceptable_home_game_rng : int
        The maximum acceptable home game range.
    num_weeks : int
        The number of weeks in the season.
    clusterings_information_df : pandas.core.frame.DataFrame
        A clusterings information DataFrame containing clustering
        source(s), information, and analysis results.
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
    optimized_schedule_analysis_df : pandas.core.frame.DataFrame
        A DataFrame with the best specified number of schedules for
        each of the results in the best parameter survey results
        DataFrame. Best schedules defined as the schedules with the
        lowest weighted total score.

    """

    # Initialize DataFrame to hold best schedules (includes duplicate results between schedule settings)
    tmp_best_schedules_df = pd.DataFrame()

    counter = 0

    # Iterate over schedule settings
    for idx, best_result in best_survey_results_df.iterrows():

        print("... Creating and analyzing schedules for parameter setting {} of {} ...".format(counter, len(best_survey_results_df) - 1))

        counter += 1

        # Extract relevant information about clustering
        clustering_id = best_result['Clustering ID']
        home_game_constraint_combo = best_result['Home Game Constraint Combo']

        # Extract relevant clustering information
        for idx, clustering_information in clusterings_information_df[clusterings_information_df['Clustering ID'] == clustering_id].iterrows():

            # Optimize and analyze specified number of schedules
            schedules = optimize_and_analyze_schedule_specified_num_times(num_times, max_acceptable_home_game_rng, num_weeks, clustering_information, home_game_constraint_combo, venue_one_way_distance_matrix, venue_one_way_duration_matrix, weekend_overnight_threshold, distance_overnight_threshold, rooms_per_overnight, cost_per_room, cost_per_unit_distance)

            if not schedules:

                print("... No suitable schedules found.\n")
                continue

            else:

                print()

                # Create DataFrame from schedule results
                schedules_df = pd.DataFrame(schedules)

                # Drop duplicate results for current schedule setting (keep the best of the duplicates) & return DataFrame with best schedules first
                schedules_df = drop_duplicate_results_keep_the_best(schedules_df, duplicate_metric, schedule_score_total_cost_weight)

                # Add schedule ID column
                schedules_df = add_optimized_schedule_id_column(schedules_df)

                # Append specified number of results to tmp best schedules DataFrame
                tmp_best_schedules_df = tmp_best_schedules_df.append(schedules_df.head(num_to_keep))

    # Drop duplicate results within home game groups (keep the best of the duplicates)
    optimized_schedule_analysis_df = drop_duplicate_results_within_hg_groups_keep_the_best(tmp_best_schedules_df, duplicate_metric, schedule_score_total_cost_weight)

    # Add rescaled and score columns to full survey results DataFrame
    optimized_schedule_analysis_df = create_score_columns(optimized_schedule_analysis_df, schedule_score_total_cost_weight)
    optimized_schedule_analysis_df = optimized_schedule_analysis_df.sort_values('Schedule Score', ascending=True).reset_index(drop=True)

    return(optimized_schedule_analysis_df)


###################################
# Select Best Optimized Schedules #
###################################


def create_best_optimized_schedule_analysis_df(optimized_schedule_analysis_df, num_to_keep):
    """
    Create a DataFrame with the best specified number of optimized
    schedule analysis results for each number of home games with best
    defined as the result with the lowest schedule score.

    Parameters
    ----------
    optimized_schedule_analysis_df : pandas.core.frame.DataFrame
        A DataFrame with the best specified number of schedules for
        each of the results in the parameter survey DataFrame. Best schedules defined as the schedules with the
        lowest schedule score.
    num_to_keep : int
        The number of best optimized schedule analysis results to keep
        for each number of home games.

    Returns
    -------
    best_optimized_schedule_analysis_df : pandas.core.frame.DataFrame
        A DataFrame with the best specified number of optimized
        schedule analysis results for each number of home games with
        best defined as the result with the lowest schedule score.

    """

    # Determine the team home game (min, max, rng) column name - could be ... or (average) ...
    team_hg_mmr_col = [col for col in optimized_schedule_analysis_df.columns if "Team Home Game (Min, Max, Rng)" in col][0]

    # Determine sorting column
    if "Pair/Quad Score" in optimized_schedule_analysis_df.columns:
        sorting_col = "Pair/Quad Score"
    else:
        sorting_col = "Schedule Score"

    hg_grouped = optimized_schedule_analysis_df.groupby(team_hg_mmr_col)

    best_optimized_schedule_analysis_df = pd.DataFrame()

    for name, hg_group_df in hg_grouped:

        hg_group_df = hg_group_df.sort_values(sorting_col, ascending=True)

        best_optimized_schedule_analysis_df = best_optimized_schedule_analysis_df.append(hg_group_df.head(num_to_keep))

    best_optimized_schedule_analysis_df = best_optimized_schedule_analysis_df.sort_values(sorting_col, ascending=True).reset_index(drop=True)

    return(best_optimized_schedule_analysis_df)


def create_abbreviated_schedule_id_column(analysis_df):

    # Determine the team home game (min, max, rng) column name - could be ... or (average) ...
    team_hg_mmr_col = [col for col in analysis_df.columns if "Team Home Game (Min, Max, Rng)" in col][0]

    # Determine schedule id columns - could be ... or ... 0 or ... 1 or ... 2 or ... 3
    sched_id_cols = [col for col in analysis_df.columns if "Schedule ID" in col and "Abbreviated" not in col]

    for idx, row in analysis_df.iterrows():

        # Create NESCAC schedule abbreviated IDs
        if "nescac" in row[sched_id_cols[0]]:

            abbreviated_schedule_id = row[sched_id_cols[0]].split("_")[0]

            for col in sched_id_cols[1:]:

                # Extract yy from yyyy
                abbreviated_schedule_id = abbreviated_schedule_id + "_" + row[col].split("_")[0][-2:]

            analysis_df.loc[idx, 'Abbreviated Schedule ID'] = abbreviated_schedule_id

        # Create optimized schedule abbreviated IDS
        else:

            # Determine prefix based on single, pair, or quad of schedules
            prefix_di = {
                1: "s",
                2: "p",
                4: "q"}

            hg_str = "_".join(['{0:g}'.format(n) for n in row[team_hg_mmr_col][0:2]])

            analysis_df.loc[idx, 'Abbreviated Schedule ID'] = "{}{}_{}".format(prefix_di[len(sched_id_cols)], idx, hg_str)

    return(analysis_df)


def save_schedules_and_team_schedules_from_df(venues_mapping_file, analysis_df, schedule_file_path):

    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_file)[1]

    # Iterate over best optimized schedules to save
    for idx, row in analysis_df.iterrows():

        schedule_id = row["Schedule ID"]

        print("Building optimized conference schedule {} ...".format(idx))
        try:
            schedule_df = create_schedule_df(row['Matches List'])
        except Exception:
            schedule_df = create_schedule_df(ast.literal_eval(row['Matches List']))

        schedule_df = schedule_df.replace({
                'Team 1': id_to_name_mapping,
                'Team 2': id_to_name_mapping,
                'Location': id_to_name_mapping})

        print("Saving conference schedule to CSV ...".format(idx))
        schedule_filename = schedule_file_path + schedule_id.replace('_team', '') + ".csv"
        schedule_df.to_csv(schedule_filename)
        print("... Saved as \"{}\"".format(schedule_filename))

        print("Preview:")
        display(schedule_df.head())
        print()

        print("Building optimized team conference schedule {} ...".format(idx))
        team_schedule_df = create_team_schedule_df(schedule_df)

        print("Saving team conference schedule to CSV ...")
        team_schedule_filename = schedule_file_path + schedule_id + ".csv"
        team_schedule_df.to_csv(team_schedule_filename)
        print("... Saved as \"{}\"".format(team_schedule_filename))

        print("Preview:")
        display(team_schedule_df.head())
        print()


##############################
# Best Consecutive Schedules #
##############################

def reduce_schedules_by_cluster(optimized_schedule_analysis_df, num_clusterings, num_to_keep):
    """
    Reduce the number of schedules by selecting the best (based on schedule score) specified number of schedules to keep for each clustering in each team home game (min, max, rng) group.
    """

    # Group optimized schedule analysis results by number of home games
    hg_grouped = optimized_schedule_analysis_df.groupby('Team Home Game (Min, Max, Rng)')

    # Initialize DataFrame to hold best optimized schedule analysis results
    best_optimized_schedule_analysis_df = pd.DataFrame()

    # Iterate over home game groups
    for name, hg_group_df in hg_grouped:

        # Iterate over clustering groups
        for clustering_id, clustering_group_df in hg_group_df.groupby('Clustering ID'):

            # Check if clustering one of the best specified number of clusterings
            if clustering_id in hg_group_df['Clustering ID'].unique()[:num_clusterings]:

                clustering_group_df = clustering_group_df.sort_values('Schedule Score', ascending=True)
                best_optimized_schedule_analysis_df = best_optimized_schedule_analysis_df.append(clustering_group_df.head(num_to_keep))

    best_optimized_schedule_analysis_df = best_optimized_schedule_analysis_df.sort_values('Schedule Score', ascending=True)

    return(best_optimized_schedule_analysis_df)
