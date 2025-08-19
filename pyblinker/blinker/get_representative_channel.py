"""Channel selection utilities derived from the legacy MATLAB Blinker approach."""

def filter_blink_amplitude_ratios(df, params):
    """
    Reduce the number of candidate signals based on the blink amplitude ratios.
    Filter rows based on blink amplitude range.
    If no rows remain, set status and select the row with the max number_good_blinks.
    """
    # Filter DataFrame based on the blink amplitude ratio range
    filtered_df = df[
        (df['blink_amp_ratio'] >= params['blink_amp_range_1'])
        &
        (df['blink_amp_ratio'] <= params['blink_amp_range_2'])
        ]

    if filtered_df.empty:
        # Handle the case where no rows pass the filter
        df['status'] = "Blink amplitude too low -- may be noise"
        df['select'] = False  # Initialize the 'select' column
        max_good_blinks_idx = df['number_good_blinks'].idxmax()  # Get the index of the max value
        df.loc[max_good_blinks_idx, 'select'] = True  # Mark the row as selected
        return df
    else:
        # Add status and select columns for filtered rows
        # filtered_df['status'] = "Blink amplitude within acceptable range."
        # filtered_df['select'] = True
        return filtered_df



def filter_good_blinks(df, params):
    """
    Find the ones that meet the minimum good blink threshold.
    Filter rows based on number of good blinks.
    If no rows remain, set status and select the row with max number_good_blinks.
    """
    # Filter DataFrame based on minimum good blinks
    filtered_df = df[df['number_good_blinks'] > params['min_good_blinks']]

    if filtered_df.empty:
        # Handle the case where no rows meet the threshold
        df['status'] = "Fewer than {} minimum Good Blinks were found".format(params['min_good_blinks'])
        df['select'] = False  # Initialize 'select' column
        max_good_blinks_idx = df['number_good_blinks'].idxmax()  # Find the index of the max value
        df.loc[max_good_blinks_idx, 'select'] = True  # Mark the row as selected
        return df
    else:
        # Add status and select columns for the filtered rows
        # filtered_df['status'] = "Meets minimum good blinks threshold."
        # filtered_df['select'] = True
        return filtered_df


def filter_good_ratio(df, params):
    """
    Filter rows based on good ratio threshold.
    If no rows meet the criteria, add a status column and select the row with the maximum number of good blinks.
    """
    # Filter based on good ratio threshold
    filtered_df = df[df['good_ratio'] >= params['good_ratio_threshold']]

    if filtered_df.empty:
        # Create a status column
        df['status'] = "Good ratio too low. We will select the row with the maximum number of good blinks."

        # Find the index of the row with the maximum number of good blinks
        max_minGoodBlinks_idx = df['number_good_blinks'].idxmax()

        # Add a 'select' column to indicate the selected row
        df['select'] = False
        df.loc[max_minGoodBlinks_idx, 'select'] = True

        return df
    else:
        # # Add a status column for the filtered rows
        # filtered_df['status'] = "Good ratio meets threshold."
        #
        # # Add a 'select' column to indicate all rows that pass the filter
        # filtered_df['select'] = True

        return filtered_df



def select_max_good_blinks(df):
    """
    Ensure that the row with the maximum number_good_blinks is selected if no row is already selected.
    """
    # Check if the 'select' column exists and if any value is True
    if 'select' in df.columns and df['select'].any():
        # If there is already a selection, do nothing
        # df['status'] = "Selection already exists."
        return df
    else:
        # If 'select' column does not exist or no row is selected, proceed with the logic
        max_number_good_blinks_idx = df['number_good_blinks'].idxmax()
        df['status'] = "Complete all checking"
        df['select'] = False  # Initialize 'select' column if it doesn't exist
        df.loc[max_number_good_blinks_idx, 'select'] = True  # Select the row with the maximum value
        return df

def channel_selection(channel_blink_stats, params):
    # Apply the blink signal selection process

    channel_blink_stats = filter_blink_amplitude_ratios(channel_blink_stats, params)
    channel_blink_stats = filter_good_blinks(channel_blink_stats, params)
    channel_blink_stats = filter_good_ratio(channel_blink_stats, params)
    signal_data_output = select_max_good_blinks(channel_blink_stats)

    # Columns to ignore
    columns_to_ignore = ['status', 'select']



    # Remove `status` and `select` columns from the comparison
    signal_data_output = signal_data_output.drop(columns=columns_to_ignore, errors='ignore')

    return signal_data_output


