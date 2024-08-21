import os
import sqlite3
import pandas as pd


def retrieve_from_db(database_path: str) -> None:
    """
    Retrieve data from an SQLite database and save it as a Parquet file.

    This function connects to an SQLite database file located in the specified
    directory, reads all the data from a table named 'Jobs', and saves the
    resulting DataFrame to a Parquet file in the same directory.

    Args:
        database_path (str): The directory path where the SQLite database
                             (`anon_jobs.db3`) is located.

    Returns:
        None: The function does not return any value. It performs its operations
              and saves the output directly to the file system.
    """
    # Establish a connection to the database file
    database_file = os.path.join(database_path, "anon_jobs.db3")
    conn = sqlite3.connect(database_file)

    # Load data from SQLite into a DataFrame
    df = pd.read_sql_query("SELECT * FROM Jobs", conn)

    # Save the DataFrame to a parquet File
    dataset_file = os.path.join(database_path, "dataset.parquet")
    df.to_parquet(dataset_file, engine='pyarrow')

    # Close the Database Connection
    conn.close()

    return None
