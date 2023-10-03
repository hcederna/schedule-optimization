""" Various functions to optimize a NESCAC volleyball conference
schedule. """
import itertools
import math
import random

import pandas as pd
import pulp

from src.helpers import calculate_min_max_rng_from_dict


###############
# Group Level #
###############

def calculate_total_num_group_matches(num_groups):
    """
    Calculate the total number of group matches.

    We know that each group plays every other group including itself
    one time. We also know that group i playing group j is the same as
    group j playing group i. Accordingly, we will use the combination
    formula where the number of possible combinations of r objects from
    a set of n objects is:

        n! / (r! * (n - r)!)

    We can simplify the fomrula since r = 2 (we need to choose group i
    and group j) to:

        (n * (n - 1)) / 2

    We then need to add n to the total number of combinations to
    account for each group playing itself one time:

        ((n * (n - 1)) / 2) + n

    Parameters
    ----------
    num_groups : int
        The total number of groups.

    Returns
    -------
    num_group_matches : int
        The total number of group matches.

    """
    num_group_matches = ((num_groups * (num_groups - 1)) / 2) + num_groups

    return(num_group_matches)


def optimize_group_schedule_minimize_travel(num_weeks, cluster_membership_dict, home_game_constraint_combo, centroid_one_way_distance_matrix):
    """
    Run a binary integer program to determine the group schedule that
    minimizes total driving distance traveled. Decision variables are
    X(i,j,k) = 1 if group i plays at home against group j in week k,
    0 otherwise.

    Parameters
    ----------
    num_weeks : int
        The number of weeks in the season.
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]} or in this case with {group: [teams in the
        group]}.
    home_game_constraint_combo : tuple
        A tuple of integers as (a, b, c, d) where:
            a = minimum acceptable number of home games for groups with one team

            b = maximum acceptable number of home games for groups
            with one team

            c = minimum acceptable number of home games for groups with two or more teams

            d = maximum acceptable number of home games for groups
            with two or more teams
    centroid_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all centroids in the clustering.

    Returns
    -------
    optimization_status : str
        The status from the solver either "Optimal", "Not Solved",
        "Infeasible", "Unbounded", or "Undefined".
    group_schedule : list
        A list of matches that occur in an optimal schedule.
        Matches represented as a tuple (i,j,k) that indicate group
        i plays at home against group j in week k. If no optimal group
        schedule found, returns None.

    """

    # Create a list of group IDs
    groups = [group for group in cluster_membership_dict]

    # Create a list of week IDs
    weeks = [week for week in range(num_weeks)]

    # Create a list of (i, j, k) decision variables
    matches = [(i, j, k) for i in groups for j in groups for k in weeks]

    # Add decision variables to PuLP
    X = pulp.LpVariable.dicts("match", matches, cat='Binary')

    # Instantiate minimization problem in PuLP
    schedule = pulp.LpProblem("schedule", pulp.LpMinimize)

    # Calculate the objective function as the sum of the roundtrip
    # distance from away group to home group for all matches that occur
    # (where X[match] = 1 if match occurs)
    objective = sum([X[match] * (2 * centroid_one_way_distance_matrix.iloc[match[1], match[0]]) for match in matches])

    # Add objective to PuLP
    schedule += objective

    # Add constraint to PuLP that each group plays every other group
    # (including itself) one time
    for i in groups:
        for j in groups:
            # Avoid duplicate constraints
            # Example:
                # sum(X(1,2,k) + X(2,1,k)) needed
                # sum(X(2,1,k) + X(1,2,k)) duplicate
            if i <= j:
                # Sum of all matches where i plays j (or j plays i) == 1
                schedule += sum([X[match] for match in matches if (match[0] == i and match[1] == j) or (match[0] == j and match[1] == i)]) == 1

    # Add constraint to PuLP that each group plays no more than 1 group per week
    for k in weeks:
        for i in groups:
            # Sum of all matches in each week for each group should be <= 1
            schedule += sum([X[match] for match in matches if match[2] == k and (match[0] == i or match[1] == i)]) <= 1

    # Add constraint to PuLP that each location hosts no more than 1 group per week
    for k in weeks:
        for i in groups:
            # Sum of all matches in each week for each location should be <= 1
            schedule += sum([X[match] for match in matches if match[2] == k and match[0] == i]) <= 1

    # Add constraint to PuLP that matches are evenly distributed throughout the weeks
    # Calculate the total number of group matches
    num_matches = calculate_total_num_group_matches(len(groups))
    # Determine the maximum number of matches per week to ensure even distribution
    weekly_matches_max = math.ceil(num_matches / len(weeks))
    for k in weeks:
        # Sum of all matches in each week should be <= maximum number of matches per week to ensure even distribution
        schedule += sum([X[match] for match in matches if match[2] == k]) <= weekly_matches_max

    # Add constraint to PuLP that home games are evenly distributed throughout the groups
    for i in groups:
        # Define lower bound and upper bound if one team in group
        if len(cluster_membership_dict[i]) < 2:
            schedule += sum([X[match] for match in matches if match[0] == i]) >= home_game_constraint_combo[0]
            schedule += sum([X[match] for match in matches if match[0] == i]) <= home_game_constraint_combo[1]
        # Define lower bound and upper bound for two or more teams in group
        else:
            schedule += sum([X[match] for match in matches if match[0] == i]) >= home_game_constraint_combo[2]
            schedule += sum([X[match] for match in matches if match[0] == i]) <= home_game_constraint_combo[3]

    # Solve integer program to determine group schedule
    schedule.solve()

    # Determine optimization status
    optimization_status = pulp.LpStatus[schedule.status]

    # Create list of matches in schedule
    group_schedule = [match for match in matches if X[match].varValue == 1.0]

    return(optimization_status, group_schedule)


#################
# Team Location #
#################

def calculate_total_num_opponents_group_j(match, cluster_membership_dict):
    """
    Determine the number of opponents in group j.

    Parameters
    ----------
    match : tuple
        A match represented as a tuple (i,j,k) that indicates group i
        plays at home against group j in week k.
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]} or in this case with {group: [teams in the
        group]}.

    Returns
    -------
    num_opponents : int
        The number of opponents in the group j.

    """

    # Extract group i and group j from the (i,j,k) match
    group_i = match[0]
    group_j = match[1]

    # Calculate the number of opponents in group j
    num_opponents = len(cluster_membership_dict[group_j])

    # Decrement the number of opponents if the group plays itself (because a team does not play itself)
    if group_i == group_j:

        num_opponents -= 1

    return(num_opponents)


def assign_match_locations_evenly_distribute_home_matches(group_schedule, cluster_membership_dict):
    """
    Assign match locations to evenly distribute the number of home
    matches among teams in a group. Outputs schedule with matches
    represented as a tuple (i,j,k,l) that indicate group i plays
    group j at team (a member of group i) location l during week k.

    Parameters
    ----------
    group_schedule : list
        A list of matches represented as a tuple (i,j,k) that indicate
        group i plays at home against group j in week k.
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]} or in this case with {group: [teams in the
        group]}.

    Returns
    -------
    team_location_group_schedule : list
        A list of matches with locations assigned to evenly distribute
        the number of home matches among teams in a group. Matches
        represented as a tuple (i,j,k,l) that indicate group i plays
        group j at team (a member of group i) location l during week k.

    """

    # Create a list of group IDs
    groups = [group for group in cluster_membership_dict]

    # Initialize list to hold (i,j,k,l) matches
    team_location_group_schedule = []

    for group in groups:

        # Create list of matches where group plays at home
        group_home_matches = [match for match in group_schedule if match[0] == group]

        # Create list of teams in the group
        group_teams = cluster_membership_dict[group]

        # Create list of possible location combinations
        location_combinations = list(itertools.product(group_teams, repeat=len(group_home_matches)))

        best_rng = 100000  # Arbitrary large number to minimize

        # Iterate over each location combination
        for loc_combo in location_combinations:

            # Initialize dictionary to count the number of home matches for each team in the group
            group_team_home_game_dict = {team: 0 for team in group_teams}

            # Iterate over each location
            for idx in range(len(loc_combo)):

                # Extract home team (the location) from the location combination
                home_team = loc_combo[idx]

                # Extract corresponding match
                match = group_home_matches[idx]

                # Add number of opponents in away group j to dictionary
                group_team_home_game_dict[home_team] += calculate_total_num_opponents_group_j(match, cluster_membership_dict)

            # Calculate the range of home matches
            rng = calculate_min_max_rng_from_dict(group_team_home_game_dict)[2]

            # Check if found better location combination
            if rng < best_rng:
                best_loc_combo = loc_combo
                best_rng = rng

                # Check if found best possible location combination
                # Break to save computation time
                if best_rng == 0:
                    break

        # Add location from best location combination to (i,j,k) resulting in (i,j,k,l)
        for idx in range(len(group_home_matches)):
            team_location_group_schedule.append(group_home_matches[idx] + (best_loc_combo[idx],))

    return(team_location_group_schedule)


##############
# Team Level #
##############

def assign_team_matchups_and_time_slots(team_location_group_schedule, cluster_membership_dict):
    """
    Assign team from group i versus team from group j matchups to
    ensure that each team in group i plays each team in group j.
    Randomly assign time slots during the week for each matchup to
    occur. Outputs schedule with matches represented as a tuple
    (i,j,k,m,l) that indicate team i plays team j at location l
    during week k time slot m.

    Parameters
    ----------
    team_location_group_schedule : list
        A list of matches represented as a tuple (i,j,k,l) that
        indicate group i plays group j at location l during week k.
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]} or in this case with {group: [teams in the
        group]}.

    Returns
    -------
    schedule : list
        A list of matches represented as a tuple (i,j,k,m,l) that
        indicate team i plays team j at location l during week k
        time slot m.

    """

    # Initialize list to hold (i,j,k,m,l) matches
    schedule = []

    for match in team_location_group_schedule:

        # Initialize variable to count time slots
        time_slot = 0

        # Extract week and location from (i,j,k,l) match
        week = match[2]
        location = match[3]

        # Extract group i and group j from (i,j,k,l) match
        group_i = match[0]
        group_j = match[1]

        # Extract group i teams and group j teams
        group_i_teams = cluster_membership_dict[group_i]
        group_j_teams = cluster_membership_dict[group_j]

        # Group plays itself
        if group_i == group_j:

            matchups = list(itertools.combinations(group_i_teams, 2))

        # Group plays another group
        else:

            matchups = [(team_i, team_j) for team_i in group_i_teams for team_j in group_j_teams]

            # Determine Friday matchups
            # Removes possibility for team to have 2 Friday matches
            random.shuffle(group_i_teams)
            random.shuffle(group_j_teams)
            fri_matchups = [(group_i_teams[idx], group_j_teams[idx]) for idx in range(0, min(len(group_i_teams), len(group_j_teams)))]

            # Assign Friday team matchup and time slot to (i,j,k,l) resulting in (i,j,k,m,l)
            for matchup in fri_matchups:

                schedule.append(matchup + (week, time_slot, location,))

                time_slot += 1

                # Remove from matchups so not assigned twice
                matchups.remove(matchup)

        # Shuffle matchups to randomly assign (remaining) time slots
        random.shuffle(matchups)

        # Add team matchup and time slot to (i,j,k,l) resulting in (i,j,k,m,l)
        for matchup in matchups:

            schedule.append(matchup + (week, time_slot, location,))

            time_slot += 1

    return(schedule)


###################
# Create Schedule #
###################

def find_optimized_schedule(num_weeks, cluster_membership_dict, home_game_constraint_combo, centroid_one_way_distance_matrix):
    """
    Find a schedule that minimizes the total distance traveled by
    groups and evenly distributes the number of home matches among
    teams in a group.

    Parameters
    ----------
    num_weeks : int
        The number of weeks in the season.
    cluster_membership_dict : dict
        A dictionary of cluster membership with {cluster: [points in
        the cluster]} or in this case with {group: [teams in the
        group]}.
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
    centroid_one_way_distance_matrix : pandas.core.frame.DataFrame
        A squareform matrix of one-way driving distances in miles
        between all centroids in the clustering.

    Returns
    -------
    optimization_status : str
        The status from the solver either "Optimal", "Not Solved",
        "Infeasible", "Unbounded", or "Undefined".
    schedule : list
        A list of matches represented as a tuple (i,j,k,m,l) that
        indicate team i plays team j at location l during week k
        time slot m.

    """

    # Run a binary integer program to determine the group schedule that
    # minimizes total distance traveled
    optimization_status, group_schedule = optimize_group_schedule_minimize_travel(num_weeks, cluster_membership_dict, home_game_constraint_combo, centroid_one_way_distance_matrix)

    # No optimal group schedule found
    if optimization_status != "Optimal":

        return(optimization_status, None)

    # Assign match locations to evenly distribute the number of home
    # matches among teams in a group.
    team_location_group_schedule = assign_match_locations_evenly_distribute_home_matches(group_schedule, cluster_membership_dict)

    # Assign team from group i versus team from group j matchups to
    # ensure that each team in group i plays each team in group j.
    # Randomly assign time slots during the week for each matchup to
    # occur.
    schedule = assign_team_matchups_and_time_slots(team_location_group_schedule, cluster_membership_dict)

    return(optimization_status, schedule)


def assign_days_str(schedule_df):
    """
    Assign a day of the week to each match. Matches played on Friday or
    Saturday. Determine the maximum number of matches that occur during
    any week at any location. For any week and location, if there are
    more than a third of the maximum number of matches rounded up to
    the nearest whole match, assign the first third of those matches
    rounded down to the nearest whole match to Friday and the remaining
    matches to Saturday. Otherwise assign all matches to Saturday.

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing schedule details including each
        match week, time slot, team 1, team 2, and location.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing schedule details including each
        match week, day (str), time slot, team 1, team 2, and location.

    """

    # Determine the maximum number of matches that occur during any week at any location
    # Add 1 to max value because time slots assigned from 0
    num_matches_max = schedule_df['Time Slot'].max() + 1
    num_matches_third = math.ceil(num_matches_max / 3)

    # Intialize dictionary to hold day (str) assignments
    day_assignments = {}

    # Ensure schedule DataFrame sorted by week and location
    schedule_df = schedule_df.sort_values(['Week', 'Location']).reset_index(drop=True)

    # Iterate over groups of matches with the same week and location
    for matches_group_name, matches_group in schedule_df.groupby(['Week', 'Location']):

        # Determine the number of matches that occur in the current week and location
        num_curr_matches = len(matches_group['Time Slot'])
        num_curr_matches_third = math.ceil(num_curr_matches / 3)

        # Determine the first index in the group
        start_idx = matches_group.index[0]

        # Iterate over matches (idx) in the group of matches
        for idx in matches_group.index:

            # Match in first third of current group matches and there are more than a third of the maximum number of matches in the current group
            if idx < (start_idx + num_curr_matches_third) and num_curr_matches >= num_matches_third:

                # Match occurs on a Friday
                day_assignments[idx] = 'Fri.'

            else:

                # Match occurs on a Saturday
                day_assignments[idx] = 'Sat.'

    # Add day (str) column to schedule DataFrame
    schedule_df['Day (Str)'] = pd.Series(day_assignments)

    return(schedule_df)


def create_schedule_df(matches_li):
    """
    Create a schedule DataFrame containing schedule details including
    each match week, day (str), time slot, team 1, team 2, and location.

    Parameters
    ----------
    matches_li : list
        A list of matches represented as a tuple (i,j,k,m,l) that
        indicate team i plays team j at location l during week k
        time slot m.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing schedule details including each
        match week, day (str), time slot, team 1, team 2, and location.

    """

    # Initialize list to hold match information
    schedule = []

    for match in matches_li:

        # Extract relevant information from (i,j,k,m,l)
        match_information = {
            'Week': match[2],
            'Time Slot': match[3],
            'Team 1': match[0],
            'Team 2': match[1],
            'Location': match[4]
        }

        schedule.append(match_information)

    # Create schedule DataFrame
    schedule_df = pd.DataFrame.from_records(schedule).sort_values(['Week', 'Location']).reset_index(drop=True)

    # Add a day (str) column by assigning a day of the week to each match
    schedule_df = assign_days_str(schedule_df)

    column_order = [
        'Week',
        'Day (Str)',
        'Time Slot',
        'Team 1',
        'Team 2',
        'Location'
    ]

    schedule_df = schedule_df[column_order]

    return(schedule_df)
