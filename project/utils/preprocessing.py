import re
from typing import Dict, Tuple

import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd


def prepare_time_filtered_dataset(
    data: pd.DataFrame,
    date_column: str = "dtstart",
    target_column: str = "60_6mob",
) -> pd.DataFrame:
    """
    Filters the DataFrame based on the content of the target column, determining the offset dynamically (e.g., '60_6mob' means 6 months).
    Also preprocesses the specified target column by converting '0.0' to 0 and any other value to 1.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - date_column (str): The name of the column containing date information.
    - target_column (str): The name of the target column to preprocess and to determine the offset duration.

    Returns:
    - pd.DataFrame: The filtered and preprocessed DataFrame.
    """
    # Ensure 'date_column' is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])

    # Determine the offset from the target column name
    offset_months = int(re.findall(r"\d+", target_column)[-1])

    # Find the last date in the DataFrame
    last_date = data[date_column].dt.to_period("M").dt.to_timestamp().max()

    # Calculate the offset date from the last date
    offset_date = last_date - MonthBegin(n=offset_months)

    # Filter the DataFrame based on the calculated offset date using boolean indexing
    filtered_data = data[data[date_column] < offset_date.ceil("1d")]

    # Preprocess the target column
    filtered_data[target_column] = filtered_data[target_column].apply(
        lambda x: 0 if x <= 0 else 1
    )

    # Sort the DataFrame by the date column
    filtered_data = filtered_data.sort_values(date_column)

    return filtered_data


def calculate_month_start_end(
    last_date: pd.Timestamp, n: int
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Calculate the start and end date for a given month offset from the last date."""
    month_end = last_date - MonthEnd(n=n - 1)
    month_start = month_end - MonthBegin(n=1)
    return month_start.floor("1d"), month_end.ceil("1d")


def extract_monthly_indices(
    data: pd.DataFrame,
    date_column: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> pd.Index:
    """Extract indices for data within a specified month."""
    index = data[
        (data[date_column] >= month_start) & (data[date_column] <= month_end)
    ].index
    return index


def print_default_rate_info(
    data: pd.DataFrame,
    indices: pd.Index,
    target_column: str,
    month_name: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
):
    """Calculate default rate and print summary information for a given month."""
    default_rate = (
        data.loc[indices, target_column].mean() if not indices.empty else float("nan")
    )
    print(
        f"Month: {month_name} - Samples: {len(indices)} - Default Rate: {default_rate:.2%}\
            - Dataset start date: {month_start.strftime('%Y-%m-%d')}\
                - Dataset end date: {month_end.strftime('%Y-%m-%d')}"
    )


def extract_oot_indices(
    data: pd.DataFrame, date_column: str, target_column: str, n_months: int
) -> Tuple[pd.Index, pd.Index, Dict[str, pd.Index]]:
    """
    Extracts indices for the last n full calendar months from the given DataFrame, calculates default rates,
    and prints summary information for each month and a combined OOT dataset.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - date_column (str): Column name containing date information.
    - target_column (str): Column name for calculating default rates.
    - n_months (int): Number of last full calendar months to extract.

    Returns:
    - Tuple containing a dictionary of month names with their corresponding indices and a combined index for all n months.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    last_date = data[date_column].max()

    monthly_indices = {}
    oot_indices = pd.Index([])

    for i in range(1, n_months + 1):
        month_start, month_end = calculate_month_start_end(last_date, i)
        month_name = month_start.strftime("%Y-%m")

        current_indices = extract_monthly_indices(
            data, date_column, month_start, month_end
        )
        monthly_indices[month_name] = current_indices
        oot_indices = oot_indices.union(current_indices)

        print_default_rate_info(
            data, current_indices, target_column, month_name, month_start, month_end
        )

    train_indices = data[data[date_column] < month_start].index
    monthly_indices = dict(sorted(monthly_indices.items()))

    print_default_rate_info(
        data, train_indices, target_column, month_name, month_start, month_end
    )
    print_default_rate_info(
        data, oot_indices, target_column, month_name, month_start, month_end
    )

    return train_indices, oot_indices, monthly_indices


def get_X_y(
    df: pd.DataFrame, index: pd.Index, target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features (X) and target (y) based on the provided indices and target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which to extract features and target.
    - index (pd.Index): The indices of the rows to extract from the DataFrame.
    - target_column (str): The name of the column to use as the target variable (y).

    Returns:
    - Tuple containing the features DataFrame (X) and target Series (y).
    """
    # Extract the rows for the given indices and then drop the target column to form the features DataFrame
    X = df.loc[index].drop(columns=target_column)

    # Extract the target Series
    y = df.loc[index, target_column]

    return X, y


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Generate a DataFrame with dates over the last year and random data for target columns
    np.random.seed(42)  # For reproducibility
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=365, freq="D")
    data = pd.DataFrame(
        {
            "dtstart": date_range,
            "60_6mob": np.random.choice([0, 1], size=len(date_range), p=[0.8, 0.2]),
            "30_3mob": np.random.choice([0, 1], size=len(date_range), p=[0.8, 0.2]),
        }
    )

    data = pd.read_csv("../data/raw/raw.csv")

    print("Example 1: Using the '60_6mob' Target Column")
    processed_df_6mob = prepare_time_filtered_dataset(data, target_column="60_6mob")
    print(processed_df_6mob.head(), "\n")

    print("Example 2: Using the '30_3mob' Target Column")
    processed_df_3mob = prepare_time_filtered_dataset(data, target_column="30_3mob")
    print(processed_df_3mob.head(), "\n")

    data = pd.read_csv("../data/raw/raw.csv")

    # Now you have `monthly_dfs` as a list of DataFrames for each of the last 3 months
    # and `oot_df` as a combined DataFrame of these months
    target_column = "60_6mob"
    data = prepare_time_filtered_dataset(data, target_column="60_6mob")
    train_indices, oot_indices, monthly_indices = extract_oot_indices(
        processed_df_6mob, "dtstart", target_column, 3
    )

    X, y = get_X_y(data, data.index, target_column)
    X_train, y_train = get_X_y(data, train_indices, target_column)
    X_oot, y_oot = get_X_y(data, oot_indices, target_column)

    X_monthly = {}
    y_monthly = {}
    for month, month_indices in monthly_indices.items():
        X_month, y_month = get_X_y(data, month_indices, target_column)
        X_monthly[month] = X_month
        y_monthly[month] = y_month

    # print(X.head())
    # print(X_train.tail())
    # print(X_oot.head())
