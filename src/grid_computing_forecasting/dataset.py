import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Dataset cleaning
def keep_useful_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and retains only the useful features from the input DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing job information.

    Returns:
    pd.DataFrame: A DataFrame containing only the selected useful features.
    """
    df = dataframe[[
        # 'JobID',
        'SubmitTime',
        # 'WaitTime',
        'RunTime',
        'ReqNProcs',
        'ReqTime',
        # 'ReqMemory',
        'Status',
        'UserID',
        'GroupID',
        'ExecutableID',
        'QueueID',
        'OrigSiteID',
        'JobStructure',
        # 'JobStructureParams',
    ]]

    return df


def remove_negative_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes rows with negative values in the specified column.

    :param df: Pandas DataFrame
    :param column: Name of the column to check for negative values
    :return: Pandas DataFrame with rows containing negative values in the specified column removed
    """
    # Filter the DataFrame to keep only rows where the specified column has non-negative values
    filtered_df = df[df[column] >= 0]

    # Reset the index of the filtered DataFrame
    return filtered_df.reset_index(drop=True)


def keep_finished_jobs(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Keeps rows with jobs finished.

    :param df: Pandas DataFrame
    :param column: Name of the column to check for negative values
    :return: Pandas DataFrame with rows containing only jobs finished
    """
    # Filter the DataFrame to keep only rows where jobs are finished
    df_filtered = df[df[column] == 1]
    # Remove column
    df_filtered = df_filtered.drop(columns=[column])

    return df_filtered


def filter_top_n_users(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filters the dataframe to retain only the rows corresponding to the top 'n' most frequent UserIDs.

    Parameters:
    df (pd.DataFrame): The input dataframe containing at least a 'UserID' column.
    n (int): The number of top most frequent UserIDs to retain.

    Returns:
    pd.DataFrame: A new dataframe filtered to include only rows with the top 'n' most frequent UserIDs.
    """
    # Count the occurrences of each UserID
    user_counts = df['ExecutableID'].value_counts()

    # Get the top 'n' most frequent UserIDs
    top_n_users = user_counts.index[:n]

    # Filter the dataframe to retain only rows with the top 'n' UserIDs
    filtered_df = df[df['ExecutableID'].isin(top_n_users)]

    return filtered_df


def filter_non_zero_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the input dataframe to retain only rows where the 'RunTime' column has a non-zero value.

    Parameters:
    df (pd.DataFrame): The input dataframe containing at least a 'RunTime' column.

    Returns:
    pd.DataFrame: A new dataframe with only rows where 'RunTime' is not zero.
    """
    # Ensure the dataframe has a 'RunTime' column
    if 'RunTime' not in df.columns:
        raise ValueError("The input dataframe must contain a 'RunTime' column.")

    # Filter the rows where 'RunTime' is different from 0
    filtered_df = df[df['RunTime'] != 0]

    return filtered_df


def plot_boxplot(df: pd.DataFrame, x_axis: str, y_axis: str):
    """
    Creates a boxplot based on the columns of the dataframe.

    :param dataframe: Pandas DataFrame containing the data
    :param x_axis: Name of the column for the x-axis
    :param y_axis: Name of the column for the y-axis
    """
    # Set the figure size for better visualization
    plt.figure(figsize=(10, 5))

    # Create the boxplot using seaborn
    sns.boxplot(x=x_axis, y=y_axis, data=df)

    # Add title and labels
    plt.title(f'Boxplot of {y_axis} by {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # Rotate x-axis labels for better readability if necessary
    plt.xticks(rotation=90)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def remove_outliers_by_app(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the 'RunTime' column for each executableID based on the IQR method.

    Parameters:
    df (pd.DataFrame): The input dataframe containing 'RunTime' and 'ExecutableID' columns.

    Returns:
    pd.DataFrame: A dataframe with outliers removed from the 'RunTime' column for each ExecutableID.
    """
    # Initialize an empty list to store filtered data
    filtered_df = pd.DataFrame()

    # Loop through each UserID group
    for executable_id, group in df.groupby('ExecutableID'):
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = group['RunTime'].quantile(0.25)
        Q3 = group['RunTime'].quantile(0.75)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the group to remove outliers
        filtered_group = group[(group['RunTime'] >= lower_bound) & (group['RunTime'] <= upper_bound)]

        # Append the filtered group to the filtered dataframe
        filtered_df = pd.concat([filtered_df, filtered_group])

    return filtered_df


# Features engineering
def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies OneHotEncoder on the specified columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of columns to apply OneHotEncoder on.

    Returns:
    pd.DataFrame: The DataFrame with the encoded columns.
    """
    # Use pandas' get_dummies method to encode the specified columns
    df_encoded = pd.get_dummies(df, columns=columns, dtype=int)

    return df_encoded


def aggregate_by_time(df: pd.DataFrame, time_column: str, period: str) -> pd.DataFrame:
    """
    Aggregates a DataFrame by the specified time period.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    time_column (str): The name of the datetime column to aggregate by.
    period (str): The period for aggregation (e.g., 'T' for minute, 'H' for hour, 'D' for day).

    Returns:
    pd.DataFrame: The aggregated DataFrame.
    """
    # Convert 'SubmitTime' column to datetime
    df_sorted = df.sort_values(by=time_column)

    # Set the time_column as the index
    df_sorted.set_index(time_column, inplace=True)

    # Resample the DataFrame by the specified period and sum the numerical columns
    df_aggregated = df_sorted.resample(period).sum()

    # Reset the index to move the time back to a column
    df_aggregated.reset_index(inplace=True)

    return df_aggregated


def split_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'SubmitTime' column in a pandas DataFrame from Unix time to
    datetime and sort the DataFrame by the 'SubmitTime' column.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing a 'SubmitTime' column with Unix time values.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the 'SubmitTime' column converted to datetime format and sorted in ascending order by 'SubmitTime'.
    """
    # Convert column "SubmitTime" to timestamp
    df["Timestamp"] = pd.to_datetime(df["SubmitTime"], unit='s')
    # Extract year, month, day, hour, minute, and second into separate columns
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute

    # Remove the 'SubmitTime' column
    df = df.drop(columns=['SubmitTime'])

    # Time-related columns
    time_columns = ['Timestamp', 'Year', 'Month', 'Day', 'Hour', 'Minute']

    # Other columns (everything except the time-related columns)
    other_columns = [col for col in df.columns if col not in time_columns]

    # Rearrange columns: put time-related columns at the beginning
    df = df[time_columns + other_columns]

    return df


def calculate_rmse_per_user(df: pd.DataFrame, medians_dict: dict) -> dict:
    """
    This function calculates the RMSE for each UserGroup based on the provided medians.

    Parameters:
    df: pandas DataFrame - The input DataFrame containing RunTime and GroupID columns.
    medians_dict: dict - A dictionary with the median values for each UserGroup.

    Returns:
    rmse_dict: dict - A dictionary containing the RMSE for each UserGroup.
    """
    rmse_dict = {}

    # Loop through each group in the median dictionary
    for group, median_value in medians_dict.items():
        if pd.isna(median_value):  # Skip if the median is NaN
            continue

        # Filter the rows where the group has value 1
        group_data = df[df[group] == 1]

        # Check if there are any data points for this group
        if not group_data.empty:
            # Calculate RMSE for this group
            rmse = np.sqrt(np.mean((group_data['RunTime'] - median_value) ** 2))
            rmse_dict[group] = rmse
        else:
            # If no data points, set RMSE to NaN
            rmse_dict[group] = np.nan

    return rmse_dict


from sklearn.metrics import mean_absolute_error

def calculate_mae_per_userid(df, medians_dict):
    """
    Calculate the Mean Absolute Error (MAE) between the 'RunTime' column in the dataframe 
    and the provided median values for each UserID using sklearn's mean_absolute_error.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing 'RunTime' and UserID columns.
    medians_dict (dict): A dictionary where keys are UserID column names and values are the medians 
                         for those UserIDs.

    Returns:
    dict: A dictionary with UserID as keys and their corresponding MAE as values.
          If a UserID has no records or the median is NaN, its value will be None.
    """
    # Create a dictionary to store MAE for each UserID
    mae_dict = {}

    # Iterate over each UserID in the medians dictionary
    for user_id, median in medians_dict.items():
        if pd.isna(median):  # Skip if the median is NaN
            continue

        # Filter rows where the current UserID is present (column has value 1)
        user_data = df[df[user_id] == 1]

        if not user_data.empty:
            # Compute MAE using sklearn's mean_absolute_error function
            mae = mean_absolute_error(user_data['RunTime'], np.full(len(user_data), median))
            mae_dict[user_id] = mae
        else:
            # If no records are found for the UserID, set MAE to None
            mae_dict[user_id] = None

    return mae_dict


def sort_dataframe_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts a pandas DataFrame based on the 'Timestamp' column in chronological order.

    This function takes a DataFrame that contains a 'Timestamp' column and sorts it 
    in ascending order based on that column. It then resets the index to ensure a 
    continuous index in the sorted DataFrame.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The sorted DataFrame with rows arranged in chronological order
                      based on the 'Timestamp' column and a reset index.
    """

    # Sort the DataFrame based on the 'Timestamp' column
    df_sorted = df.sort_values(by='Timestamp')

    # Reset the index to ensure a continuous index after sorting
    df_sorted.reset_index(drop=True, inplace=True)

    return df_sorted
