#############
# Hard Data #
#############

# File path for a file containing venue/team names and unique integer
# venue/team IDs. The file is formatted with one venue/team name per
# line represented as Venue/Team Name,ID
VENUES_MAPPING_FILENAME = "hard_data/venues_with_mapping.txt"


##################
# Create Figures #
##################

# Output NESCAC map filename
NESCAC_MAP_FILENAME = "figures/nescac_map.png"

# Output plot file path
PLOT_PATH = "figures/"


###########################
# Scrape NESCAC Schedules #
###########################

# Output NESCAC schedule file path
NESCAC_SCHEDULE_PATH = "data/nescac_schedule/"


################################
# Distance & Duration Matrices #
################################

# Number of tests to run on the distance matrix and on the duration
# matrix
# Recommended = 5
NUM_DIST_DUR_MATRIX_TESTS = 5

# Maximum acceptable difference in miles between a distance value from
# the distance matrix and the distance value from a corresponding
# Google Maps API query
# Recommended = 5
DIST_MAX_ACCEPTABLE_ERROR = 5

# Maximum acceptable difference in seconds between a duration value
# from the duration matrix and the duration value from a corresponding
# Google Maps API query
# Recommended = 600
DUR_MAX_ACCEPTABLE_ERROR = 600

# Output distance and duration matrix filenames
DISTANCE_MATRIX_FILENAME = "data/location/venue_one_way_distance_matrix.csv"
DURATION_MATRIX_SECONDS_FILENAME = "data/location/venue_one_way_duration_matrix_seconds.csv"
DURATION_MATRIX_HHMM_FILENAME = "data/location/venue_one_way_duration_matrix_hh_mm.csv"


####################
# Analyze Schedule #
####################

# Maximum number of miles to drive one-way without staying for an
# overnight between Friday and Saturday matches
# Recommended = 53
# ==> max one-way distance with duration < 1 hour
WEEKEND_OVERNIGHT_THRESHOLD = 53

# Maximum number of miles to drive one-way without staying for an
# overnight
# Recommended = 153
# ==> max one-way distance with duration < 3 hours
DISTANCE_OVERNIGHT_THRESHOLD = 153

# Number of rooms required to house all support staff and players
# for an overnight
# Recommended = 7-9
# ==> 14 players per team with 2 per room = 7 rooms, 2-3 per room = 5
# rooms; 2-4 support staff per team with 1-2 per room = 2 rooms
ROOMS_PER_OVERNIGHT = 9

# Cost for one room for one night
# Recommended = 133
# ==> from Hotels.com Hotel Price Index 2017 - average price for a room
# for domestic travelers
# source : https://hpi.hotels.com/us-2017/what-did-people-spend-in-the-usa/
COST_PER_ROOM = 133

# Cost to travel one mile
# Recommended = 3.12
# ==> from BusRates Guide to Charter Bus Vehicles - national average
# rate for trips priced per mile October 2013)
# ==> source : https://www.busrates.com/bustypes
COST_PER_MILE = 3.12

# Amount as a decimal percent to weight the total cost in the schedule
# score calculation
# Recommended : 0.5
# ==> weight total cost 50% and team experience 50% in the schedule
# score
SCHEDULE_SCORE_TOTAL_COST_WEIGHT = 0.5

# Output NESCAC schedule analysis filename
NESCAC_SCHEDULE_ANALYSIS_FILENAME = "analysis/nescac_schedule_analysis.pickle"


#########################
# Coordinates DataFrame #
#########################

# Number of tests to run on the coordinates DataFrame
# Recommended = 5
NUM_COORDINATES_TESTS = 5

# Maximum acceptable difference between latitude values and longitude
# values from the coordinates DataFrame and a corresponding Google Maps
# API query
# Recommended = 0.00000001
COORDS_MAX_ACCEPTABLE_ERROR = 0.00000001

# Output venues coordinates filename
VENUES_COORDINATES_FILENAME = "data/location/venues_coordinates.csv"


##################
# Cluster Venues #
##################

# Number of clusters to form
# Recommended = 5
K = 5

# List of k-means setups with each setup as a tuple (implementation,
# coordinate_type, distance_metric)
K_MEANS_SETUP_LIST = [
    ('scratch', 'geographic', 'euclidean'),
    ('scratch', 'geographic', 'great_circle'),
    ('scratch', 'geographic', 'vincenty'),
    ('scipy', 'geographic', 'euclidean'),
    ('sklearn', 'geographic', 'euclidean'),
    ('scratch', 'cartesian', 'euclidean'),
    ('scipy', 'cartesian', 'euclidean'),
    ('sklearn', 'cartesian', 'euclidean')
]

# Maximum acceptable number of points per cluster
# Recommended = 3
MAX_ACCEPTABLE_NUM_POINTS = 3

# Output clusterings information filename
CLUSTERINGS_INFORMATION_FILENAME = "analysis/clusterings_information.pickle"


#####################
# Optimize Schedule #
#####################

# Number of weeks in the season
# Recommended = 5
NUM_WEEKS = 5


#####################
# Survey Parameters #
#####################

# Name of the column to check for duplicate values
# Recommended = 'Total Distance'
DUPLICATE_METRIC = 'Total Distance'

# Acceptable minimum and maximum values for the home game constraint
# combination for groups with one team and groups with two or more
# teams
MIN_ACCEPTABLE_HG_ONE_TEAM = 2
MAX_ACCEPTABLE_HG_ONE_TEAM = 5
MIN_ACCEPTABLE_HG_MULTI_TEAM = 1
MAX_ACCEPTABLE_HG_MULTI_TEAM = 5

# Data points to consider and average for the home game constraint
# survey results
VALUES_TO_CONSIDER_SURVEY = [
    'Home Game Constraint Combo',
    'Clustering ID',
    'Total Cost',
    'Team Home Game Fairness Index',
    # 'Team Home Game Average',
    'Team Home Game (Min, Max, Rng)',
    'Team Duration in Hours Fairness Index',
    'Team Duration in Hours Average',
    'Team Duration in Hours (Min, Max, Rng)',
    'Team Overnight Fairness Index',
    'Team Overnight Average',
    'Team Overnight (Min, Max, Rng)',
    'Total Distance',
    'Total Duration in Hours',
    'Total Overnight',
    'Total Distance Cost',
    'Total Overnight Cost',
    'Test Success']

# Maximum acceptable home game range for the optimized schedules
# Calculated as the difference ih home games between team with the
# most and team with the least
# Recommended = 2
MAX_ACCEPTABLE_HG_RNG = 2

# Output parameter survey filename
PARAMETER_SURVEY_RESULTS_FILENAME = "analysis/parameter_survey_results.csv"


####################
# Create Schedules #
####################

# Output optimized schedule analysis filename
OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/optimized_schedule_analysis.pickle"


#######################################
# Select + Save Best Single Schedules #
#######################################

# Output best single optimized schedule analysis filename
BEST_SINGLE_OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/single/analysis/best_single_optimized_schedule_analysis.pickle"

# Output optimized schedule file path
BEST_SINGLE_OPTIMIZED_SCHEDULE_PATH = "results/single/optimized_schedules/"

# Output single clusterings information filename
SINGLE_CLUSTERINGS_INFORMATION_FILENAME = "results/single/analysis/single_clusterings_information.pickle"

# Output single clusterings plot file path
SINGLE_CLUSTERINGS_PLOT_PATH = "results/single/figures/"


###########################################
# Select + Save Best Schedule Pairs/Quads #
###########################################

# Number of schedule indices to include in each schedule combination
# Pair ==> 2
PAIR = 2

# Number of schedule indices to include in each schedule combination
# Quad ==> 4
QUAD = 4

# A list of data points to average (where applicable) or sum (where applicable) for the results DataFrame. Data points that contain "Total" but do not contain "Score" are summed. Remaining numeric data points are averaged.
VALUES_TO_CONSIDER_PAIR_QUAD = [
    'Schedule Score',
    'Total Cost',
    'Team Experience Score',
    'Team Home Game (Min, Max, Rng)',
    'Team Home Game Fairness Index',
    'Team Home Game Score',
    'Team Duration in Hours Average',
    'Team Duration in Hours Fairness Index',
    'Team Duration in Hours Score',
    'Team Overnight Average',
    'Team Overnight Fairness Index',
    'Team Overnight Score',
    'Test Success',
    'Total Distance',
    'Total Duration in Hours',
    'Total Overnight',
    'Total Distance Cost',
    'Total Overnight Cost',
]

# Output pair optimized schedule analysis filename
PAIR_OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/pair_optimized_schedule_analysis.pickle"


# Output best pair optimized schedule analysis filename
BEST_PAIR_OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/pair/analysis/best_pair_optimized_schedule_analysis.pickle"

# Output optimized schedule file path
BEST_PAIR_OPTIMIZED_SCHEDULE_PATH = "results/pair/optimized_schedules/"

# Output pair clusterings information filename
PAIR_CLUSTERINGS_INFORMATION_FILENAME = "results/pair/analysis/pair_clusterings_information.pickle"

# Output pair clusterings plot file path
PAIR_CLUSTERINGS_PLOT_PATH = "results/pair/figures/"

# Output pair NESCAC schedule analysis filename
PAIR_NESCAC_SCHEDULE_ANALYSIS_FILENAME = "results/pair/analysis/pair_nescac_schedule_analysis.pickle"

# Output quad optimized schedule analysis filename
QUAD_OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/quad_optimized_schedule_analysis.pickle"

# Output best quad optimized schedule analysis filename
BEST_QUAD_OPTIMIZED_SCHEDULE_ANALYSIS_FILENAME = "results/quad/analysis/best_quad_optimized_schedule_analysis.pickle"

# Output optimized schedule file path
BEST_QUAD_OPTIMIZED_SCHEDULE_PATH = "results/quad/optimized_schedules/"

# Output quad clusterings information filename
QUAD_CLUSTERINGS_INFORMATION_FILENAME = "results/quad/analysis/quad_clusterings_information.pickle"

# Output quad clusterings plot file path
QUAD_CLUSTERINGS_PLOT_PATH = "results/quad/figures/"

# Output quad NESCAC schedule analysis filename
QUAD_NESCAC_SCHEDULE_ANALYSIS_FILENAME = "results/quad/analysis/quad_nescac_schedule_analysis.pickle"


############################
# Compare Single Schedules #
############################

# Output single schedule ccomparison filename
SINGLE_SCHEDULE_COMPARISON_FILENAME = "results/single/single_schedule_comparison.pickle"

# Output results plot file path
SINGLE_RESULTS_PLOT_PATH = "results/single/figures/"


##########################
# Compare Pair Schedules #
##########################

# Output pair schedule ccomparison filename
PAIR_SCHEDULE_COMPARISON_FILENAME = "results/pair/pair_schedule_comparison.pickle"

# Output results plot file path
PAIR_RESULTS_PLOT_PATH = "results/pair/figures/"


##########################
# Compare Quad Schedules #
##########################

# Output quad schedule ccomparison filename
QUAD_SCHEDULE_COMPARISON_FILENAME = "results/quad/quad_schedule_comparison.pickle"

# Output results plot file path
QUAD_RESULTS_PLOT_PATH = "results/quad/figures/"