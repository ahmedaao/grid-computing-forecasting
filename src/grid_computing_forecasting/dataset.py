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
        'JobID',
        'SubmitTime',
        'WaitTime',
        'RunTime',
        'NProc',
        'UsedCPUTime',
        'UsedMemory',
        'ReqNProcs',
        'Status',
        'UserID',
        'GroupID',
        # 'ExecutableID', too much values
        'QueueID',
        'OrigSiteID',
        'JobStructure'
        # 'JobStructureParams' too much values
    ]]

    return df


# Features engineering
def from_unixtime_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
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
    # Convert 'SubmitTime' column to datetime
    df_sorted = df.sort_values(by='SubmitTime')
    df_sorted['SubmitTime'] = pd.to_datetime(df_sorted['SubmitTime'], unit='s')

    # Extract year, month, day, hour, minute, and second into separate columns
    df = df_sorted.copy()
    df['Year'] = df['SubmitTime'].dt.year
    df['Month'] = df['SubmitTime'].dt.month
    df['Day'] = df['SubmitTime'].dt.day
    df['Hour'] = df['SubmitTime'].dt.hour
    # df['Minute'] = df['SubmitTime'].dt.minute
    # df['Second'] = df['SubmitTime'].dt.second

    # Rearrange columns
    columns_order = [
        'JobID',
        'SubmitTime',
        'Year',
        'Month',
        'Day',
        'Hour',
        'WaitTime',
        'RunTime',
        'NProc',
        'ReqNProcs',
        'Status',
        'UserID',
        'GroupID',
        'QueueID',
        'OrigSiteID'
    ]

    df = df[columns_order]

    return df
