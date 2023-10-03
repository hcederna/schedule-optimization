import pandas as pd
from bs4 import BeautifulSoup


def scrape_uncleaned_schedule(year):

    # Determine filename of saved websited HTML
    html_file = 'hard_data/' + str(year) + '_nescac_schedule.html'

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

    # Clean team 1 and team 2 columns
    for column in ['Team 1', 'Team 2']:
        schedule_df[column] = schedule_df[column].map(
            lambda x: x.split('\n')[0])
        
    return(schedule_df)