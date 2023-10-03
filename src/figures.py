import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap

from src.helpers import create_venue_id_mapping_dicts_from_file


# Figure font sizes
large_font_size = 24
medium_font_size = 18
small_font_size = 14

# Figure colors and palettes
nescac_green = "seagreen"
nescac_grey = "#595959"
green_palette = sns.light_palette(nescac_green, n_colors=5)
grey_palette = sns.light_palette(nescac_grey, n_colors=7)
neutral_palette = sns.color_palette("coolwarm", 3)

# Define bar chart colors
best_value_color = green_palette[2]     # medium green
worst_value_color = grey_palette[4]     # dark grey
neutral_color = neutral_palette[1]      # light grey
optimized_colors = (green_palette[4],   # very dark green
                    green_palette[3])   # dark green
bar_pattern_color = grey_palette[2]     # very light grey
bar_label_color = "white"

# Define map colors
land_color = neutral_palette[1]     # light grey
water_color = "#A7CDF2"             # light blue
line_color = grey_palette[4]        # dark grey

# Seaborn muted colors excluding muted red ("#D65F5F")
muted_colors = ["#4878CF", "#6ACC65",
                "#B47CC7", "#C4AD66", "#77BEDB"]
muted_cmap = ListedColormap(sns.color_palette(muted_colors).as_hex())


##############
# NESCAC Map #
##############

def plot_and_save_nescac_map(coordinates_df, venues_mapping_filename, nescac_map_filename):
    """
    Save and display a map of the NESCAC schools.

    Parameters
    ----------
    coordinates_df : pandas.core.frame.DataFrame
        A coordinates DataFrame including the venue ID, coordinates,
        latitude, and longitude.
    venues_mapping_filename: str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    nescac_map_filename : str
        The filename and path to use when saving the NESCAC map.

    """

    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_filename)[1]
    id_to_abbrev_name_mapping = {venue_id: venue_name[0:3].upper() for venue_id, venue_name in id_to_name_mapping.items()}

    lats = coordinates_df['Latitude'].values
    lons = coordinates_df['Longitude'].values
    lat_mean = lats.mean()
    lon_mean = lons.mean()

    # Initialize plot
    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots()

    # Add subplot title
    plt.title("The NESCAC", ha='center', va='bottom',
              fontsize=large_font_size, fontweight='bold')

    # Draw map background
    m = Basemap(resolution='i',
                projection='lcc',
                area_thresh=10000,
                lat_0=lat_mean,
                lon_0=lon_mean,
                llcrnrlon=-76.0,
                llcrnrlat=40.75,
                urcrnrlon=-66.0,
                urcrnrlat=47.75)
    m.drawmapboundary(fill_color=water_color)
    m.fillcontinents(color=land_color, lake_color=water_color)
    m.drawcoastlines(color=line_color)
    m.drawcountries(color=line_color)
    m.drawstates(color=line_color)

    # Plot school locations
    x, y = m(lons, lats)
    ax.scatter(x, y, c='black', s=60, zorder=10)

    # Add abbreviated school names
    labels = list(id_to_abbrev_name_mapping.values())
    for label, xpt, ypt in zip(labels, x, y):
        if label == "BOW":
            ax.text(xpt - 52000, ypt - 6000, label)
        elif label == "TUF":
            ax.text(xpt - 42000, ypt - 6000, label)
        else:
            ax.text(xpt + 10000, ypt + 10000, label)

    # Save map
    plt.savefig(nescac_map_filename)

    # Display map
    plt.show()


##################
# Analysis Plots #
##################

def plot_and_save_horizontal_bar_chart_min_max_mean(analysis_df, column_name, min_preferred, fig_title, nescac_mean_line, plot_path):
    """
    Save and display a horizontal bar chart with previous NESCAC
    abbreviated schedule IDs (schedule years) on the y-axis and the
    specified column values on the x-axis. If optimized schedules in
    the analysis DataFrame (where the Schedule ID column value does not
    contain "nescac"), add a vertical line at the optimized schedule
    specified column value. With min_preferred True, highlight the
    horizontal bar with the maximum value grey and the minimum value
    green. With min_preferred False, highlight the bar with the maximum
    value green and the minimum value grey. With nescac_mean_line True,
    draw a grey, dashed line at the NESCAC mean value.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    column_name : str
        The name of the column containing the values to plot on the x-
        axis.
    min_preferred : bool
        True to highlight the minimum value green and maximum value
        grey, False to highlight the minimum value grey and maximum
        value green.
    fig_title : str
        The figure title.
    nescac_mean_line : bool
        True to draw a grey, dashed line at the NESCAC mean value,
        False otherwise.
    plot_path : str
        The path for the folder to use when saving the plot.

    Returns
    -------
    plot_filename : str
        The output plot filename including the path.

    """

    # Determine schedule ID column name
    sched_id_col_name = [col for col in analysis_df.columns if "Schedule ID" in col and "Abbreviated" not in col][0]

    # Fill NaN w/ '' to avoid error when separating nescac and optimized schedules
    analysis_df[sched_id_col_name] = analysis_df[sched_id_col_name].fillna('')

    # Extract NESCAC and optimized schedules into separate DataFrames
    nescac_df = analysis_df[analysis_df[sched_id_col_name].str.contains("nescac")]
    optimized_df = analysis_df[~analysis_df[sched_id_col_name].str.contains("nescac")]

    # Sort NESCAC schedules to sort bars
    nescac_df = nescac_df.sort_values(sched_id_col_name, ascending=True)

    # Initialize subplots
    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots()

    # Add figure title
    plt.suptitle(fig_title, x=0.1, ha='left', va='top', fontsize=large_font_size, fontweight='bold')

    # Determine minimum value and maximum value colors
    min_preferred_color_min_max = {
        True: (best_value_color, worst_value_color),
        False: (worst_value_color, best_value_color),
        None: (neutral_color, neutral_color)}
    min_color = min_preferred_color_min_max[min_preferred][0]
    max_color = min_preferred_color_min_max[min_preferred][1]

    # Create horizontal bars individually
    for i, row in nescac_df.reset_index(drop=True).iterrows():

        # Extract value for bar
        col_val = row[column_name]

        # Determine schedule score column name
        sched_score_col_name = [col for col in nescac_df.columns if "Schedule Score" in col][0]

        # Determine bar pattern
        if row[sched_score_col_name] == nescac_df[sched_score_col_name].min():
            # Found best previous NESCAC schedule based on schedule score
            bar_pattern = 'x'
        else:
            bar_pattern = ''

        # Determine bar color
        if col_val == nescac_df[column_name].max():
            # Found maximum column value
            bar_color = max_color
        elif col_val == nescac_df[column_name].min():
            # Found minimum column value
            bar_color = min_color
        else:
            bar_color = neutral_color

        # Add horizontal bar with specified color and pattern to subplot
        ax.barh(i, col_val, height=0.67, color=bar_color,
                hatch=bar_pattern, edgecolor=bar_pattern_color, lw=1,
                zorder=0)

        # Add white border to the horizontal bar
        ax.barh(i, col_val, height=0.67, color='none',
                edgecolor='white', lw=2,
                zorder=1)

        # Determine bar label format
        if "Total Cost" in column_name and "Percent" not in column_name:
            label = " {:,.0f} ".format(col_val)
        else:
            label = " {:,.2f} ".format(col_val)

        # Determine bar label alignment and color
        if col_val == 0:
            label_alignment = "left"
            if min_preferred:
                label_color = best_value_color
            else:
                label_color = worst_value_color
        else:
            label_alignment = "right"
            label_color = "white"

        # Add bar label
        ax.text(col_val, i, label,
                color=label_color, fontweight='bold', fontsize=small_font_size,
                ha=label_alignment, va='center', zorder=100)

    # Add y-tick values to subplot
    plt.yticks(range(len(nescac_df)), list(nescac_df['Abbreviated Schedule ID'].values))

    # Create vertical lines for optimized schedules
    for i, row in optimized_df.reset_index(drop=True).iterrows():

        # Extract values for line
        sched_id = row["Abbreviated Schedule ID"]
        col_value = row[column_name]

        # Add vertical line for optimized schedule
        ax.axvline(col_value, c=optimized_colors[i],
                   ls="solid", lw=3,
                   zorder=10)

        # Determine line label format
        if "Total Cost" in column_name and "Percent" not in column_name:
            label = " {:,.0f} ".format(col_value)
        else:
            label = " {:,.2f} ".format(col_value)

        # Determine line label alignment
        if col_value == optimized_df[column_name].max():
            alignment = 'left'
        else:
            alignment = 'right'

        # Add line value label
        ax.text(col_value, len(nescac_df) - 0.5, label,
                color=optimized_colors[i], fontweight='bold', style='italic', fontsize=small_font_size,
                ha=alignment, va="bottom")

        # Add line schedule label
        ax.text(col_value, -0.5, "  {}  ".format(sched_id),
                color=optimized_colors[i], fontweight='bold', style='italic', fontsize=small_font_size,
                ha=alignment, va="top")

    # Create vertical NESCAC mean line
    if nescac_mean_line:

        # Calculate NESCAC mean value
        mean_value = nescac_df[column_name].mean()

        # Add vertical mean line
        ax.axvline(mean_value, c="gray",
                   ls="dashed",
                   zorder=-10)

        # Add mean line label
        ax.text(mean_value, -0.5, " mean = {:,.2f} ".format(mean_value),
                style='italic', fontsize=small_font_size,
                ha='right', va="top")

    # Hide plot outline
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Hide y-axis label
    ax.set_ylabel('')

    # Hide x-axis ticks/labels
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    # Hide y-axis ticks on left subplot
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelsize=small_font_size,
        pad=14)

    # Save plot
    plot_filename = plot_path + fig_title.lower().replace(" ", "_") + "png"
    plt.savefig(plot_filename)

    # Display plot
    plt.show()

    return(plot_filename)


def plot_and_save_columns_line_chart(analysis_df, column_names, fig_title, plot_path):
    """
    Save and display a line chart with abbreviated schedule IDs on the
    x-axis and a line for each of the specified columns on the y-axis.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    column_names : list
        A list with the name of the columns containing the values to plot
        on the y-axis.
    fig_title : str
        The figure title.
    plot_path : str
        The path for the folder to use when saving the plot.

    Returns
    -------
    plot_filename : str
        The output plot filename including the path.

    """

    # Sort the schedules by abbreviated schedule ID
    analysis_df = analysis_df.sort_values('Abbreviated Schedule ID', ascending=True)

    # Initialize subplots
    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots()

    # Add figure title
    plt.suptitle(fig_title, x=0.1, ha='left', va='top', fontsize=large_font_size, fontweight='bold')

    # Determine line colors
    line_colors = [neutral_color, muted_colors[1], muted_colors[0]]

    # Plot lines
    for i, column_name in enumerate(column_names):

        ax.plot(analysis_df['Abbreviated Schedule ID'], analysis_df[column_name], c=line_colors[i])

        # Add float mean line label
        last_x_val = list(analysis_df['Abbreviated Schedule ID'].values)[-1]
        last_y_val = list(analysis_df[column_name].values)[-1]

        ax.text(last_x_val, last_y_val, "  {}".format(column_name),
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=small_font_size)

    # Customize y-axis values
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylim(0)

    # Hide plot outline
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Hide x-axis ticks/labels
    ax.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=True,
        labelleft=True)

    # Add grid
    ax.grid(linestyle=":", axis="y")

    # Save plot
    plot_filename = plot_path + fig_title.lower().replace(" ", "_") + "png"
    plt.savefig(plot_filename)

    # Display plot
    plt.show()

    return(plot_filename)


def create_columns_from_dict_keys(analysis_df, dict_column_name, venues_mapping_filename):
    """
    Create a column from each of the specified dict column keys.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    dict_column_name : str
        The name of the column containing a dict as {venue/school
        ID: value} to plot on the y-axis.
    venues_mapping_filename: str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID

    Returns
    -------
    tmp : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results plus the additional columns from each of the specified
        dict column keys.

    """
    tmp = analysis_df.copy()

    # Add a column for each of the dict keys
    tmp = tmp.join(tmp[dict_column_name].apply(lambda x: pd.Series(x)))

    # Create a mapping dictionary with school ID: abbreviated school name
    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_filename)[1]
    id_to_abbrev_name_mapping = {venue_id: venue_name[0:3].upper() for venue_id, venue_name in id_to_name_mapping.items()}

    # Rename columns from school ID to abbreviated school name
    tmp = tmp.rename(columns=id_to_abbrev_name_mapping)

    return(tmp)


def plot_and_save_dict_column_team_swarmplot_chart(analysis_df, dict_column_name, fig_title, venues_mapping_filename, plot_path):
    """
    Save and display a swarmplot with abbreviated schedule IDs on the x-axis
    and the specified dict column team values on the y-axis.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    dict_column_name : str
        The name of the column containing a dict as {venue/school
        ID: value} to plot on the y-axis.
    fig_title : str
        The figure title.
    venues_mapping_filename: str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    plot_path : str
        The path for the folder to use when saving the plot.

    Returns
    -------
    plot_filename : str
        The output plot filename including the path.

    """

    # Sort the schedules by abbreviated schedule ID
    analysis_df = analysis_df.sort_values('Abbreviated Schedule ID', ascending=True)

    # Prepare analysis DataFrame by splitting dict column keys into columns
    analysis_df = create_columns_from_dict_keys(analysis_df, dict_column_name, venues_mapping_filename)

    # Extract column base from column name
    column_base = dict_column_name.replace(" Dict", "").replace("Team ", "")

    # Create tmp DataFrame for swarmplot containing only abbreviated schedule ID and dict column key columns
    columns = [column for column in analysis_df.columns if len(column) == 3 or column == 'Abbreviated Schedule ID']
    swarmplot_df = analysis_df[columns].copy()
    swarmplot_df = swarmplot_df.melt(id_vars=['Abbreviated Schedule ID'], var_name='Team', value_name=column_base)
    swarmplot_df = swarmplot_df.sort_values(['Abbreviated Schedule ID', 'Team'])

    # Initialize subplots
    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots()

    # Add figure title
    plt.suptitle(fig_title, x=0.1, ha='left', va='top', fontsize=large_font_size, fontweight='bold')

    # Plot swarmplot
    sns.swarmplot(x='Abbreviated Schedule ID', y=column_base, data=swarmplot_df, hue='Team', size=8)

    # Customize plot
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.ylabel("")
    plt.ylim(0)
    plt.legend(bbox_to_anchor=(1, 1), markerfirst=True, frameon=False)

    # Hide plot outline
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add grid
    ax.grid(linestyle=":", axis="y")

    # Hide x-axis ticks/labels
    ax.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=True,
        labelleft=True)

    # Save plot
    plot_filename = plot_path + "team_{}_swarmplot.png".format(column_base.lower())
    plt.savefig(plot_filename)

    # Display plot
    plt.show()

    return(plot_filename)


def plot_and_save_horizontal_bar_charts_min_max_mean(analysis_df, column_names, min_preferred_tuple, fig_title, subplot_titles, nescac_mean_line, optimized_schedule_lines, plot_path):
    """
    Save and display side by side horizontal bar charts with previous
    NESCAC abbreviated schedule IDs (schedule years) on the y-axis in
    between the charts and the specified column values on the x-axis.
    With optimized_schedule_lines True, draw a vertical line at the
    optimized schedule specified column value. With min_preferred True,
    highlight the horizontal bar with the maximum value grey and the
    minimum value green. With min_preferred False, highlight the bar
    with the maximum value green and the minimum value grey. With
    nescac_mean_line True, draw a grey, dashed line at the NESCAC mean
    value.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    column_names : tuple
        The name of the columns containing the values to plot on the x-
        axis as a tuple with (left subplot column name, right subplot
        column name)
    min_preferred_tuple : tuple
        True to highlight the minimum value green and maximum value
        grey, False to highlight the minimum value grey and maximum
        value green as a tuple with (left subplot min preferred, right
        subplot min preferred).
    fig_title : str
        The figure title.
    subplot_titles : tuple
        The subplot titles as a tuple with (left subplot title, right
        subplot title).
    nescac_mean_line : bool
        True to draw a grey, dashed line at the NESCAC mean value,
        False otherwise.
    optimized_schedule_lines : bool
        True to draw a vertical line at the specified column value for
        each of the optimized schedules in the analysis DataFrame
        (where we define an optimized schedule as a schdule with a
        Schedule ID column value not containing "nescac") on each of
        the subplots, False otherwise.
    plot_path : str
        The path for the folder to use when saving the plot.

    Returns
    -------
    plot_filename : str
        The output plot filename including the path.

    """

    # Determine schedule ID column name
    sched_id_col_name = [col for col in analysis_df.columns if "Schedule ID" in col and "Abbreviated" not in col][0]

    # Fill NaN w/ '' to avoid error when separating nescac and optimized schedules
    analysis_df[sched_id_col_name] = analysis_df[sched_id_col_name].fillna('')

    # Extract NESCAC and optimized schedules into separate DataFrames
    nescac_df = analysis_df[analysis_df[sched_id_col_name].str.contains("nescac")]
    optimized_df = analysis_df[~analysis_df[sched_id_col_name].str.contains("nescac")]

    # Sort NESCAC schedules to sort bars
    nescac_df = nescac_df.sort_values(sched_id_col_name, ascending=True)

    # Initialize subplots
    plt.style.use('seaborn-poster')
    fig, axes = plt.subplots(ncols=len(column_names), sharey=True)

    # Adjust subplot whitespace to show full nescac schedule labels
    if len(list(nescac_df['Abbreviated Schedule ID'].values)[0]) == 7:
        # Working with schedule pairs
        plt.subplots_adjust(wspace=0.3)
    elif len(list(nescac_df['Abbreviated Schedule ID'].values)[0]) == 13:
        # Working with schedule quads
        plt.subplots_adjust(wspace=0.5)

    # Add figure title
    plt.suptitle(fig_title, x=0.1, ha='left', va='top', fontsize=large_font_size, fontweight='bold')

    # Create subplots
    for idx, column_name in enumerate(column_names):

        # Add subplot title
        subplot_title = subplot_titles[idx]
        axes[idx].set_title(subplot_title, loc='center', va="bottom", fontsize=medium_font_size)

        # Determine if minimum value preferred
        min_preferred = min_preferred_tuple[idx]

        # Determine minimum value and and maximum value colors
        min_preferred_color_min_max = {
            True: (best_value_color, worst_value_color),
            False: (worst_value_color, best_value_color),
            None: (neutral_color, neutral_color)}
        min_color = min_preferred_color_min_max[min_preferred][0]
        max_color = min_preferred_color_min_max[min_preferred][1]

        # Create horizontal bars individually
        for i, row in nescac_df.reset_index(drop=True).iterrows():

            # Extract value for bar
            col_val = row[column_name]

            # Determine schedule score column name
            sched_score_col_name = [col for col in nescac_df.columns if "Schedule Score" in col][0]

            # Determine bar pattern
            if row[sched_score_col_name] == nescac_df[sched_score_col_name].min():
                # Found best previous NESCAC schedule based on schedule score
                bar_pattern = 'x'
            else:
                bar_pattern = ''

            # Determine bar color
            if col_val == nescac_df[column_name].max():
                # Found maximum column value
                bar_color = max_color
            elif col_val == nescac_df[column_name].min():
                # Found minimum column value
                bar_color = min_color
            else:
                bar_color = neutral_color

            # Ensure negative values graphed correctly
            if col_val < 0:
                positive_col_val = -1 * col_val
            else:
                positive_col_val = col_val

            # Add horizontal bar with specified color and pattern to subplot
            axes[idx].barh(i, positive_col_val, height=0.67, color=bar_color,
                           hatch=bar_pattern, edgecolor=bar_pattern_color, lw=1,
                           zorder=0)

            # Add white border to the horizontal bar
            axes[idx].barh(i, positive_col_val, height=0.67, color='none',
                           edgecolor='white', lw=2,
                           zorder=1)

            # Determine bar label format
            if "Total" in column_name and "Percent" not in column_name:
                label = " {:,.0f} ".format(col_val)
            else:
                label = " {:,.2f} ".format(col_val)

            # Determine bar label alignment and color
            if col_val == 0:
                if idx == 0:
                    label_alignment = 'right'
                else:
                    label_alignment = 'left'

                if min_preferred:
                    label_color = best_value_color
                else:
                    label_color = worst_value_color
            else:
                label_color = "white"
                if idx == 0:
                    label_alignment = 'left'
                else:
                    label_alignment = 'right'

            # Add bar label
            axes[idx].text(positive_col_val, i, label,
                           color=label_color, fontweight='bold', fontsize=small_font_size,
                           ha=label_alignment, va='center', zorder=100)

        # Add y-tick values to subplot
        plt.yticks(range(len(nescac_df)), list(nescac_df['Abbreviated Schedule ID'].values))

        if optimized_schedule_lines:

            found_max = False

            # Create vertical lines for optimized schedules
            for i, row in optimized_df.reset_index(drop=True).iterrows():

                # Extract values for line
                sched_id = row["Abbreviated Schedule ID"]
                col_value = row[column_name]

                # Add vertical line for optimized schedule
                axes[idx].axvline(col_value, c=optimized_colors[i],
                                  ls='solid', lw=3,
                                  zorder=1000)

                # Determine line label format
                if "Total Cost" in column_name and "Percent" not in column_name:
                    label = " {:,.0f} ".format(col_value)
                else:
                    label = " {:,.2f} ".format(col_value)

                # Determine line label alignment
                if col_value == optimized_df[column_name].max() and not found_max:
                    if idx == 0:
                        alignment = 'right'
                    else:
                        alignment = 'left'
                    found_max = True
                else:
                    if idx == 0:
                        alignment = 'left'
                    else:
                        alignment = 'right'

                # Add line value label
                axes[idx].text(col_value, len(nescac_df) - 0.5, label,
                               color=optimized_colors[i], fontweight='bold', style='italic', fontsize=small_font_size,
                               ha=alignment, va="bottom")

                # Add line schedule label
                axes[idx].text(col_value, -0.5, "  {}  ".format(sched_id),
                               color=optimized_colors[i], fontweight='bold', style='italic', fontsize=small_font_size,
                               ha=alignment, va="top")

        # Create vertical NESCAC mean line
        if nescac_mean_line:

            # Calculate NESCAC mean value
            mean_value = nescac_df[column_name].mean()

            # Ensure negative values graphed correctly
            if mean_value < 0:
                positive_mean_value = mean_value * -1
            else:
                positive_mean_value = mean_value

            # Add vertical mean line
            axes[idx].axvline(positive_mean_value, c="gray",
                              ls="dashed",
                              zorder=-10)

            # Determine mean line label alignment
            if idx == 0:
                alignment = 'left'
            else:
                alignment = 'right'

            # Add mean line label
            axes[idx].text(positive_mean_value, -0.5, " mean = {:,.2f} ".format(mean_value),
                           style='italic', fontsize=small_font_size,
                           ha=alignment, va="top")

        # Hide plot outline
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['bottom'].set_visible(False)
        axes[idx].spines['left'].set_visible(False)

        # Hide y-axis label
        axes[idx].set_ylabel('')

        # Hide x-axis ticks/labels
        axes[idx].tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)

    # Invert x-axis on left subplot
    axes[0].invert_xaxis()

    # Move schedule IDs in between the two subplots
    axes[0].yaxis.tick_right()

    # Hide y-axis ticks on left subplot
    axes[0].tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelsize=small_font_size,
        pad=14)

    # Hide y-axis ticks/labels on right subplot
    axes[1].tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)

    # Save plot
    plot_filename = plot_path + fig_title.lower().replace(" ", "_") + "png"
    plt.savefig(plot_filename)

    # Display plot
    plt.show()

    return(plot_filename)


######################
# DataFrame Displays #
######################

def apply_color_gradient(analysis_df, columns):
    """
    Apply color gradient to the specified DataFrame. Best values appear
    dark green and worst values appear light green.

    Parameters
    ----------
    analysis_df : pandas.core.frame.DataFrame
        A DataFrame with schedule-related information and analysis
        results.
    columns : list
        The list of columns to display in the resulting DataFrame.

    Returns
    -------
    color_gradient_df : pandas.core.frame.DataFrame
        A DataFrame with an applied color gradient containing schedule-
        related information and analysis results.

    """

    # Create cmap
    cm_min_preferred = sns.light_palette("seagreen", as_cmap=True, reverse=True)
    cm_max_preferred = sns.light_palette("seagreen", as_cmap=True, reverse=False)

    # Determine columns to color
    columns_to_color = [column for column in columns if not any(s in column for s in ['ID', 'Combo', '(Min, Max, Rng)', 'Test Success', 'Indexes'])]

    # Determine number of decimals to display for each column
    format_di = {col: ("{:.2f}" if "Total" not in col else "{:,.0f}") for col in columns_to_color}

    # Determine if min or max preferred
    max_preferred = [col for col in columns_to_color if ("Fairness Index" in col) or ("Location Average" in col)]
    min_preferred = [col for col in columns_to_color if col not in max_preferred]

    try:

        # Apply color gradient
        color_gradient_df = analysis_df[columns].sort_values('Schedule Score', ascending=True).style.background_gradient(
            cmap=cm_min_preferred, subset=min_preferred).background_gradient(
            cmap=cm_max_preferred, subset=max_preferred).format(format_di)

    except Exception:

        # Apply color gradient
        color_gradient_df = analysis_df[columns].sort_values('Pair/Quad Score', ascending=True).style.background_gradient(
            cmap=cm_min_preferred, subset=min_preferred).background_gradient(
            cmap=cm_max_preferred, subset=max_preferred).format(format_di)

    return(color_gradient_df)


####################
# Clustering Plots #
####################

def plot_clustering_with_centroids(points, clustering_information, venues_mapping_filename):
    """
    Display a plot of clustered points and centroids with each point
    colored based on cluster membership.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    clustering_information : pandas.core.series.Series
        A row from the clusterings information DataFrame that contains
        clustering source(s), information, and analysis results.
    venues_mapping_filename : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID

    """

    # Extract clustering assignments, centroid coordinates, and clustering ID from clustering information
    cluster_assignments = clustering_information['Cluster Assignments']
    centroid_coordinates = clustering_information['Centroid Coordinates']
    clustering_id = clustering_information['Clustering ID']

    # Initialize plot
    plt.style.use('seaborn-talk')
    fig, ax = plt.subplots()

    # Add subplot title
    plt.title("Clustering %s" % (clustering_id), ha='center', va='bottom',
              fontsize=large_font_size, fontweight='bold')

    # Plot points
    ax.scatter(points[:, 1], points[:, 0], c=cluster_assignments, s=120, cmap=muted_cmap)

    # Plot centroids
    ax.scatter(centroid_coordinates[:, 1], centroid_coordinates[:, 0], c='red', marker='x', s=60)

    # Remove x-axis and y-axis ticks/values
    plt.tick_params(
        which='both',
        bottom='off',
        left='off',
        labelbottom='off',
        labelleft='off')

    # Display plot
    plt.show()



def plot_and_save_clustering_on_nescac_map(points, clustering_information, venues_mapping_filename, plot_path):
    """
    Save and display a plot of clustered points on a NESCAC map with
    each point colored based on cluster membership.

    Parameters
    ----------
    points : numpy.ndarray
        An array of cartesian (x, y, z) or geographic (latitude,
        longitude) points.
    clustering_information : pandas.core.series.Series
        A row from the clusterings information DataFrame that contains
        clustering source(s), information, and analysis results.
    venues_mapping_filename : str
        The file path for a file containing venue names and unique
        integer venue IDs. The file is formatted with one venue name
        per line represented as Venue Name,ID
    plot_path : str
        The path for the folder to use when saving the plot.

    Returns
    -------
    plot_filename : str
        The output plot filename including the path.
    """

    # Create a mapping dictionary with school ID: abbreviated school name
    id_to_name_mapping = create_venue_id_mapping_dicts_from_file(venues_mapping_filename)[1]
    id_to_abbrev_name_mapping = {venue_id: venue_name[0:3].upper() for venue_id, venue_name in id_to_name_mapping.items()}

    # Extract lats and lons from the points
    lats = points[:, 0]
    lons = points[:, 1]
    lat_mean = lats.mean()
    lon_mean = lons.mean()

    # Extract clustering assignments and clustering ID from clustering information
    cluster_assignments = clustering_information['Cluster Assignments']
    clustering_id = clustering_information['Clustering ID']

    # Initialize plot
    plt.style.use('seaborn-poster')
    fig, ax = plt.subplots()

    # Add subplot title
    plt.title("Clustering %s" % (clustering_id), ha='center', va='bottom',
              fontsize=large_font_size, fontweight='bold')

    # Draw map background
    m = Basemap(resolution='i',
                projection='lcc',
                area_thresh=10000,
                lat_0=lat_mean,
                lon_0=lon_mean,
                llcrnrlon=-76.0,
                llcrnrlat=40.75,
                urcrnrlon=-66.0,
                urcrnrlat=47.75)
    m.drawmapboundary(fill_color=water_color)
    m.fillcontinents(color=land_color, lake_color=water_color)
    m.drawcoastlines(color=line_color)
    m.drawcountries(color=line_color)
    m.drawstates(color=line_color)

    # Plot school locations as abbreviated school names
    labels = list(id_to_abbrev_name_mapping.values())
    x, y = m(lons, lats)
    for label, cluster_assignment, xpt, ypt in zip(labels, cluster_assignments, x, y):
        cluster_color = muted_colors[cluster_assignment]

        ax.text(xpt, ypt, label, ha='center', va='center',
                color=land_color, fontweight='bold',
                bbox=dict(facecolor=cluster_color,
                          edgecolor=cluster_color,
                          pad=1))

    # Save plot
    plot_filename = plot_path + "clustering_%s.png" % (clustering_id)
    plt.savefig(plot_filename)

    # Display map
    plt.show()

    return(plot_filename)
