from unittest.mock import patch

import pandas as pd

from utils.db.connector import DatabaseConnector


def test_fetch_data():
    # Mock the cursor's fetchall method to return a predefined list of dictionaries
    expected_output = [{"id": 1, "name": "Test Item"}]
    with patch("pymssql.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = expected_output

        db_connector = DatabaseConnector(
            server="test_server",
            user="test_user",
            password="test_password",
            database_name="test_db",
        )
        with db_connector as db:
            result = db.fetch_data("SELECT * FROM test_table")
            assert (
                result == expected_output
            ), "fetch_data should return the expected list of dictionaries"


def test_fetch_df():
    # Mock pandas.read_sql to return a predefined DataFrame
    expected_df = pd.DataFrame({"id": [1], "name": ["Test Item"]})
    with patch("pandas.read_sql") as mock_read_sql:
        mock_read_sql.return_value = expected_df

        db_connector = DatabaseConnector(
            server="test_server",
            user="test_user",
            password="test_password",
            database_name="test_db",
        )
        with db_connector as db:
            result_df = db.fetch_df("SELECT * FROM test_table")
            (
                pd.testing.assert_frame_equal(result_df, expected_df),
                "fetch_df should return the expected DataFrame",
            )
