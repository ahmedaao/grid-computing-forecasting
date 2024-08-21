import pandas as pd


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
        'ReqNProcs',
        'Status',
        'UserID',
        'GroupID',
        'QueueID',
        'OrigSiteID'
    ]]

    return df
