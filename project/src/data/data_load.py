import os

import pandas as pd
from dotenv import load_dotenv

from src.constants import DATA_PATH_RAW, SQL_QUERY_PATH
from utils.db.connector import DatabaseConnector
from utils.log import setup_logger

logger = setup_logger(__name__)


def save_data_to_csv(data: pd.DataFrame, file_path: str) -> None:
    """Save the given DataFrame to a CSV file at the specified path."""
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")


def read_query_from_file(file_path: str) -> str:
    """Read and return the SQL query from the specified file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Query file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading query from file: {e}")
        raise


def load_data_from_query(query: str) -> pd.DataFrame:
    """Execute the SQL query and return the results as a DataFrame."""
    try:
        with DatabaseConnector() as db_connector:
            return db_connector.fetch_df(query)
    except Exception as e:
        logger.error(f"Error loading data from query: {e}")
        raise


def load_dataset(date: str, date_start: str) -> pd.DataFrame:
    """Load data for the given or current date, and save it to a CSV file."""
    logger.info(f"Loading data for date: {date} - {date_start}")

    try:
        query_path = os.path.join(SQL_QUERY_PATH, "load_dataset.sql")
        query = read_query_from_file(query_path).format(date, date_start)
        data = load_data_from_query(query)
        save_data_to_csv(data, DATA_PATH_RAW)
        return data
    except Exception as e:
        logger.error(f"Failed to load or save data: {e}")
        raise


if __name__ == "__main__":
    load_dotenv()
    data = load_dataset()
