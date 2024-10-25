import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import _pickle as cPickle


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
        'SubmitTime',
        'RunTime',
        'ReqNProcs',
        'ReqTime',
        'Status',
        'UserID',
        'GroupID',
        'ExecutableID',
        'QueueID',
        'OrigSiteID',
        'JobStructure',
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


def filter_top_n_apps(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filters the dataframe to retain only the rows corresponding to the top 'n' most frequent ExecutableID.

    Parameters:
    df (pd.DataFrame): The input dataframe containing at least a 'ExecutableID' column.
    n (int): The number of top most frequent ExecutableIDs to retain.

    Returns:
    pd.DataFrame: A new dataframe filtered to include only rows with the top 'n' most frequent ExecutableIDs.
    """
    # Count the occurrences of each ExecutableID
    app_counts = df['ExecutableID'].value_counts()

    # Get the top 'n' most frequent ExecutableIDs
    top_n_apps = app_counts.index[:n]

    # Filter the dataframe to retain only rows with the top 'n' ExecutableIDs
    filtered_df = df[df['ExecutableID'].isin(top_n_apps)]

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


def plot_boxplot(df: pd.DataFrame, column: str, save_path: str, file_name: str):
    """
    Creates a boxplot for a given column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to be plotted in the boxplot.
    save_path (str): The directory path where the figure will be saved.
    file_name (str): The name of the file to save the plot as.

    Returns:
    None
    """
    # Set the figure size
    plt.figure(figsize=(6, 4))
    # Create the boxplot
    sns.boxplot(y=df[column])
    # Set the title and labels
    plt.title(f'Boxplot of {column} for all applications')
    plt.ylabel(column)

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    # Construct the full file path
    full_path = os.path.join(save_path, file_name)
    # Save the plot to the specified path
    plt.savefig(full_path)

    # Display the plot
    plt.show()


def df_remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes outliers based on the IQR method in a DataFrame

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to be processed and plotted.

    Returns:
    pd.DataFrame: A DataFrame without outliers for the specified column.
    """
    # Calculate quartiles and IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered


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


def create_dataframe_sample(df: pd.DataFrame, n: int, save_path: str, file_name: str) -> pd.DataFrame:
    """
    Selects a random sample of `n` rows from the given DataFrame
    and saves it to a specified path in pickle format.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame from which rows will be randomly sampled.
    n : int
        The number of rows to sample from the DataFrame.
    Returns:
    ----------
    None
    """
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    # Construct the full file path
    full_path = os.path.join(save_path, file_name)

    # Select a random sample of n rows
    df_sample = df.sample(n=n, random_state=42)
    # Save the sampled DataFrame to the specified path
    df_sample.to_pickle(full_path)


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = cPickle.load(file)
    return data
