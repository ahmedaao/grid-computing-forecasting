import pandas as pd


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
        # 'UserID', too much values
        'GroupID',
        # 'ExecutableID', too much values
        # 'QueueID',
        # 'OrigSiteID',
        'JobStructure'
        # 'JobStructureParams' too much values
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
    # df['Year'] = df['SubmitTime'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    # df['Second'] = df['Timestamp'].dt.second

    # Rearrange columns
    columns_order = [
        # 'JobID',
        # 'SubmitTime',
        'Timestamp',
        'Month',
        'Day',
        'Hour',
        'Minute',
        'RunTime',
        'ReqNProcs',
        'ReqTime',
        'GroupID_G0',
        'GroupID_G1',
        'GroupID_G2',
        'GroupID_G3',
        'GroupID_G4',
        'GroupID_G5',
        'GroupID_G6',
        'GroupID_G7',
        'GroupID_G8',
        'GroupID_G9',
        'GroupID_G10',
        'GroupID_G11',
        'JobStructure_BoT',
        'JobStructure_UNITARY'
    ]

    df = df[columns_order]

    return df
