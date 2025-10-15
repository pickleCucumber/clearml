import os

import pandas as pd
import pymssql
from dotenv import load_dotenv

from utils.log import setup_logger

logger = setup_logger(__name__)

# Load environment variables
load_dotenv()


class DatabaseConnector:
    def __init__(self, server=None, user=None, password=None, database_name=None):
        self.server = os.getenv("DB_SERVER", server)
        self.user = os.getenv("DB_USER", user)
        self.password = os.getenv("DB_PASSWORD", password)
        self.database = os.getenv("DB_NAME", database_name)
        logger.info(
            "DatabaseConnector initialized with server: %s, database: %s",
            self.server,
            self.database,
        )

    def __enter__(self):
        try:
            self.connection = pymssql.connect(
                server=self.server,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            logger.info("Connected to database.")
        except pymssql.DatabaseError as e:
            logger.error("Database connection failed: %s", e)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        logger.info("Database connection closed.")

    def fetch_data(self, query):
        logger.info("Fetching data with query: %s", query)
        with self.connection.cursor(as_dict=True) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
        logger.info("Data fetched successfully.")
        return data

    def fetch_df(self, query):
        logger.info("Fetching DataFrame with query: %s", query)
        df = pd.read_sql(query, self.connection)
        logger.info("DataFrame fetched successfully.")
        return df


if __name__ == "__main__":
    load_dotenv()

    query = "SELECT TOP 10 * FROM dataset_b_id_CL"
    with DatabaseConnector() as db:
        df = db.fetch_df(query)

        data = db.fetch_data(query)
        print("Результат выгрузки данных:")
        print(data)

        print("Результата выгрузки датафрейма:")
        print(df.head())  # For demonstration, we'll just print the first few rows

        # Save to CSV, if needed
        # df.to_csv('path_to_save_your_csv_file.csv', index=False)
