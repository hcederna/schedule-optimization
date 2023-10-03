""" Various functions to scrape, clean, and create a NESCAC volleyball
conference schedule from a NESCAC website saved HTML file. """
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.helpers import create_venue_id_mapping_dicts_from_file


##################
# Scrape Website #
##################

def scrape_conference_schedule(html_file):
    """
    Scrape the NESCAC volleyball conference schedule from the
    NESCAC website into an uncleaned schedule DataFrame where team 1
    is considered the home team and team 2 is considered the away team.
    Only includes conference matches not in the post-season.

    Partial source:

        https://chihacknight.org/blog/2014/11/26/an-intro-to-web-
        scraping-with-python.html

    Parameters
    ----------
    html_file : str
        The file path for an HTML file containing the NESCAC volleyball
        schedule for a specific year.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing uncleaned schedule details
        including each match date, month, team 1, team 2, and location.

    """

    # Parse the HTML
    with open(html_file, 'r') as fin:
        soup = BeautifulSoup(fin.read(), 'lxml')

    # Extract the schedule table
    table = soup.find_all('table')[0]

    # Extract the table rows - exclude header rows
    rows = table.find_all('tr')[1:]

    # Initialize data dictionary to hold column values
    data = {
        'Date': [],
        'Month': [],
        'Team 1': [],
        'Team 2': [],
        'Location': []
    }

    # Initialize current date and current month
    curr_date = None
    curr_month = None

    # Iterate over table rows
    for row in rows:

        # Extract column values
        cols = row.find_all('td')

        # Check if row contains a month value
        if len(cols) == 1:

            # Set month value as current month
            curr_month = cols[0].get_text()

        # Check if row contains sufficient data
        elif len(cols) > 1:

            # Find value for date column
            date = cols[0].get_text().strip()

            # Check if date column contains date
            if date:

                # Set date value as current date
                curr_date = date

            # Find values for remaining relevant columns
            away_team = cols[1].get_text().strip()
            home_team = cols[2].get_text().strip()
            location = cols[3].get_text().strip()

            # Check if a conference match (*) not in post-season (%)
            if '*' in away_team and '%' not in away_team:

                # Add values to data dictionary
                data['Date'].append(curr_date)
                data['Month'].append(curr_month)
                data['Team 1'].append(home_team)
                data['Team 2'].append(away_team)
                data['Location'].append(location)

    # Create uncleaned conference schedule DataFrame
    schedule_df = pd.DataFrame(data)

    return(schedule_df)


##################
# Clean Schedule #
##################

def clean_columns_team(schedule_df):
    """
    Remove extraneous characters from the team 1 and team 2 columns.

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing unclean team 1 and team 2
        columns.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing clean team 1 and team 2
        columns.

    """

    for column in ['Team 1', 'Team 2']:

        schedule_df[column] = schedule_df[column].map(
            lambda x: x.split('\n')[0])

    return(schedule_df)


def clean_columns_date_related(schedule_df, year):
    """
    Create two date-related columns:

        Day (Str) = the abbreviated day of the week

        Date = the date in yyyy-mm-dd format

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing unclean date-related columns.
    year : int
        The schedule year.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing clean date-related columns day
        (str) and date.

    """

    # Create day (str) column as all non-digit characters extracted
    # from date column
    schedule_df['Day (Str)'] = schedule_df['Date'].str.extract(
        '(\D+)', expand=True)

    # Remove extraneous white spaces from day (str) column
    schedule_df['Day (Str)'] = schedule_df['Day (Str)'].apply(
        lambda x: x.strip())

    # Create year column
    schedule_df['Year'] = year

    # Create dictionary for month mapping
    month_mapping = {
        "September": 9,
        "October": 10,
        "November": 11
    }

    # Map string month to integer month
    schedule_df['Month'] = schedule_df['Month'].map(month_mapping)

    # Create day column as all digit characters extracted from date
    # column
    schedule_df['Day'] = schedule_df['Date'].str.extract(
        '(\d+)', expand=True)

    # Create date column as date in yyyy-mm-dd format
    schedule_df['Date'] = pd.to_datetime(
        schedule_df[['Year', 'Month', 'Day']])

    # Remove unneccesary date-related columns from DataFrame
    schedule_df = schedule_df.drop(['Year', 'Month', 'Day'], axis=1)

    return(schedule_df)


def clean_columns_location(schedule_df):
    """
    Clean and map city, state location values to venue names.

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing an unclean match location
        column.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing a clean match location column.

    """

    # Create dictionary of one-off location values to change
    locations_to_replace = {
        "Green Dot Match  New London, Conn.": "New London, Conn.",
        "Waterville, Maine.": "Waterville, Maine",
        "Middletown,  Conn.": "Middletown, Conn."
    }

    # Replace one-off location values in DataFrame
    schedule_df['Location'] = schedule_df['Location'].replace(locations_to_replace)

    # Remove match scores with parenthesis
    schedule_df['Location'] = schedule_df['Location'].map(
        lambda x: x.split(')')[-1].strip())

    # Remove match scores without parenthesis
    # Remove commas if preceded by a digit
    schedule_df['Location'] = schedule_df['Location'].str.replace(
        '(?<=\d),', '')

    # Remove digits and hyphens
    schedule_df['Location'] = schedule_df['Location'].str.replace(
        '[\-\d]+', '')

    # Remove leading and trailing whitespaces
    schedule_df['Location'] = schedule_df['Location'].map(
        lambda x: x.strip())

    # Create dictionary for location city, state to venue name mapping
    locations_mapping = {
        'Brunswick, Maine': 'Bowdoin',
        'Medford, Mass.': 'Tufts',
        'Amherst, Mass.': 'Amherst',
        'Williamstown, Mass.': 'Williams',
        'Clinton, N.Y.': 'Hamilton',
        'Hartford, Conn.': 'Trinity',
        'Middletown, Conn.': 'Wesleyan',
        'Waterville, Maine': 'Colby',
        'New London, Conn.': 'Connecticut College',
        'Middlebury, Vt.': 'Middlebury',
        'Lewiston, Maine': 'Bates'
    }

    # Map location city, state to venue name in DataFrame
    schedule_df = schedule_df.replace(locations_mapping)

    return(schedule_df)


def correct_missing_location_values(schedule_df, year):
    """
    Export a schedule DataFrame with missing match location values to
    CSV. (User must manually correct missing match location values in
    the exported file, and save corrected file with "missing" replacing
    "added" in the filename.) Try to load a CSV file with the added
    match location values and throw an error if file does not exist.

    Parameters
    ----------
    schedule df : pandas.core.frame.DataFrame
        A schedule DataFrame possibly containing missing values in the
        match location column.
    year : int
        The schedule year.

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame without missing values in the match
        location column.

    Raises
    ------
    Exception : If a CSV file with added match location values
    not found.

    """

    # Determine filename for schedule with missing location values
    missing_locations_filename = "hard_data/" + str(year) + "_nescac_schedule_locations_missing.csv"

    # Determine filename for schedule with locations added
    added_locations_filename = "hard_data/" + str(year) + "_nescac_schedule_locations_added.csv"

    # Check if one or fewer unique location values exist
    if len(schedule_df['Location'].unique()) <= 1:

        # Export schedule with missing location values to CSV
        schedule_df = schedule_df.to_csv(missing_locations_filename)

        try:

            # Load schedule with location values added from CSV
            schedule_df = pd.read_csv(added_locations_filename, index_col=0)

            return(schedule_df)

        except Exception:

            raise Exception("Need to add missing match location values to '{}' and save file as '{}'.".format(missing_locations_filename, added_locations_filename))

    return(schedule_df)


def correct_team_and_venue_names(schedule_df, venues_mapping_file):
    """
    Replace abbreviated team and venue names with expanded team and
    venue names. Example: replace 'Amherst' with 'Amherst College'.

    Parameters
    ----------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing abbreviated team and venue
        names.
    venues_mapping_file : str
        The file path for a file containing venue/team names and unique
        integer venue/team IDs. The file is formatted with one
        venue/team name per line represented as Venue/Team Name,ID

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing expanded team and venue names.

    """

    # Create venue name to ID mapping dictionary and extract list of venue names
    expanded_names = list(create_venue_id_mapping_dicts_from_file(venues_mapping_file)[0].keys())

    # Create list of abbreviated names from DataFrame
    abbreviated_names = np.unique(schedule_df[['Team 1', 'Team 2', 'Location']].values)

    # Create abbreviated to expanded venue names mapping dictionary
    abbreviated_to_expanded_mapping = dict(zip(sorted(abbreviated_names), sorted(expanded_names)))

    # Update dictionary to avoid key: value matching error
    abbreviated_to_expanded_mapping = {k: v for k, v in abbreviated_to_expanded_mapping.items() if k != v}

    # Replace abbreviated with expanded team and venue names in DataFrame
    schedule_df = schedule_df.replace({
        'Team 1': abbreviated_to_expanded_mapping,
        'Team 2': abbreviated_to_expanded_mapping,
        'Location': abbreviated_to_expanded_mapping})

    return(schedule_df)


###################
# Create Schedule #
###################

def scrape_clean_create_nescac_schedule_df(year, html_file, venues_mapping_file):
    """
    Scrape the NESCAC volleyball conference schedule from the NESCAC
    website into a schedule DataFrame. Clean the DataFrame values and
    return a cleaned NESCAC volleyball conference schedule DataFrame.

    Parameters
    ----------
    year : int
        The schedule year.
    html_file : str
        The file path for an HTML file containing the NESCAC volleyball
        schedule for a specific year.
    venues_mapping_file : str
        The file path for a file containing venue/team names and unique
        integer venue/team IDs. The file is formatted with one
        venue/team name per line represented as Venue/Team Name,ID

    Returns
    -------
    schedule_df : pandas.core.frame.DataFrame
        A schedule DataFrame containing clean schedule details
        including each match day (str), date, team 1, team 2, and
        location.

    """

    # Scrape conference schedule from NESCAC website
    schedule_df = scrape_conference_schedule(html_file)

    # Clean schedule DataFrame
    schedule_df = clean_columns_team(schedule_df)
    schedule_df = clean_columns_location(schedule_df)
    schedule_df = correct_missing_location_values(schedule_df, year)
    schedule_df = correct_team_and_venue_names(schedule_df, venues_mapping_file)
    schedule_df = clean_columns_date_related(schedule_df, year)

    # Correct 2012 NESCAC website error
    # Bowdoin vs. Wesleyan @ Bowdoin (NESCAC website reported @ Wesleyan incorrectly)
    if year == 2012:

        schedule_df.loc[(schedule_df['Team 1'] == 'Bowdoin College') & (schedule_df['Team 2'] == 'Wesleyan University'), 'Location'] = 'Bowdoin College'

    column_order = [
        'Day (Str)',
        'Date',
        'Team 1',
        'Team 2',
        'Location']

    schedule_df = schedule_df[column_order]

    schedule_df = schedule_df.sort_values(['Date', 'Team 1', 'Location']).reset_index(drop=True)

    return(schedule_df)
