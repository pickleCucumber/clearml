import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    calculate_month_start_end,
    extract_monthly_indices,
    extract_oot_indices,
    prepare_time_filtered_dataset,
    print_default_rate_info,
)


## Тестируем то что используется внутри extract_oot_indices
def test_calculate_month_start_end():
    last_date = pd.to_datetime("2022-12-31")
    month_start, month_end = calculate_month_start_end(last_date, 1)

    assert month_start == pd.to_datetime("2022-12-01")
    assert month_end == pd.to_datetime("2022-12-31")


def test_extract_monthly_indices():
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=120, freq="D"),
            "value": range(120),
        }
    )
    date_column = "date"
    month_start = pd.to_datetime("2022-03-01")
    month_end = pd.to_datetime("2022-03-31")

    indices = extract_monthly_indices(data, date_column, month_start, month_end)

    assert len(indices) == 31  # Проверяем, что выбраны все дни марта
    assert indices[0] == 59  # Индекс первого дня марта в датафрейме
    assert indices[-1] == 89  # Индекс последнего дня марта в датафрейме


def test_print_default_rate_info(capsys):
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2022-01-01", periods=30, freq="D"),
            "target": [1, 0] * 15,
        }
    )
    indices = pd.Index(range(30))
    target_column = "target"
    month_name = "2022-01"
    month_start = pd.to_datetime("2022-01-01")
    month_end = pd.to_datetime("2022-01-30")

    print_default_rate_info(
        data, indices, target_column, month_name, month_start, month_end
    )

    captured = capsys.readouterr()
    assert "Month: 2022-01 - Samples: 30 - Default Rate: 50.00%" in captured.out


## Тут начинается тестирование основной функциональности
@pytest.fixture
def sample_data():
    np.random.seed(42)
    date_range = pd.date_range(end="2024-01-01", periods=365, freq="D")
    data = pd.DataFrame(
        {
            "dtstart": date_range,
            "60_6mob": np.random.choice([0, 1], size=len(date_range), p=[0.7, 0.3]),
            "30_3mob": np.random.choice([0, 1], size=len(date_range), p=[0.7, 0.3]),
        }
    )
    return data


def test_extract_oot_indices(sample_data):
    train_indices, oot_indices, monthly_indices = extract_oot_indices(
        sample_data, "dtstart", "60_6mob", 3
    )

    # Check that all indices are unique and there's no overlap between train and OOT
    assert len(set(train_indices).intersection(set(oot_indices))) == 0

    # Check that monthly indices cover the correct number of months
    assert len(monthly_indices) == 3

    # Check that the combined OOT indices match the individual monthly indices
    combined_indices = pd.Index([])
    for indices in monthly_indices.values():
        combined_indices = combined_indices.union(indices)
    assert combined_indices.equals(oot_indices)


def test_consistent_monthly_indices_across_years(sample_data):
    _, _, monthly_indices = extract_oot_indices(sample_data, "dtstart", "60_6mob", 3)

    # Check if the function correctly identified the last 3 months, including handling year change
    assert "2023-11" in monthly_indices.keys()
    assert "2023-12" in monthly_indices.keys()
    assert "2024-01" in monthly_indices.keys()


def test_correct_data_filtering(sample_data):
    # Using '60_6mob' which should filter the last 6 months of data
    filtered_df = prepare_time_filtered_dataset(sample_data, target_column="60_6mob")

    # Check the date range in the filtered DataFrame
    min_date = filtered_df["dtstart"].min()

    expected_min_date = sample_data["dtstart"].max() - pd.DateOffset(months=6)
    print(expected_min_date)
    assert min_date <= expected_min_date


def test_proper_target_column_preprocessing(sample_data):
    filtered_df = prepare_time_filtered_dataset(sample_data, target_column="60_6mob")

    # Check that '60_6mob' column only contains 0 or 1
    unique_values = filtered_df["60_6mob"].unique()
    assert set(unique_values).issubset({0, 1})


def test_data_sorting(sample_data):
    filtered_df = prepare_time_filtered_dataset(sample_data, target_column="60_6mob")

    # Check that data is sorted by 'dtstart' in ascending order
    assert all(filtered_df["dtstart"] == filtered_df["dtstart"].sort_values())


def test_handling_of_empty_dataframe():
    empty_data = pd.DataFrame({"dtstart": [], "60_6mob": []})
    processed_data = prepare_time_filtered_dataset(empty_data, target_column="60_6mob")

    # Expect an empty DataFrame as output
    assert processed_data.empty


def test_dynamic_offset_determination(sample_data):
    # Use '30_3mob' to check for 3 months filtering
    filtered_df_3mo = prepare_time_filtered_dataset(
        sample_data, target_column="30_3mob"
    )
    max_date_3mo = filtered_df_3mo["dtstart"].max()
    expected_min_date_3mo = sample_data["dtstart"].max() - pd.DateOffset(months=3)

    assert max_date_3mo <= expected_min_date_3mo


def test_sequential_function_usage(sample_data):
    train_indices, oot_indices, monthly_indices = extract_oot_indices(
        sample_data, "dtstart", "60_6mob", 3
    )

    # Verify preprocessing: Check if '60_6mob' has been converted to 0 and 1 correctly
    assert set(sample_data["60_6mob"].unique()) == {0, 1}

    # Verify that monthly_indices cover exactly 3 months
    assert len(monthly_indices) == 3

    # Ensure there's no overlap between train and OOT indices
    assert len(set(train_indices).intersection(set(oot_indices))) == 0

    # Validate correct indices extraction by checking if the dates match expected ranges
    last_date = sample_data["dtstart"].max()
    for i, month_name in enumerate(sorted(monthly_indices.keys(), reverse=True), 0):
        month_indices = monthly_indices[month_name]
        month_data = sample_data.loc[month_indices]
        expected_month_end = last_date - pd.offsets.MonthEnd(n=i)
        expected_month_start = expected_month_end - pd.offsets.MonthBegin(n=1)

        assert month_data["dtstart"].min() >= expected_month_start
        assert month_data["dtstart"].max() <= expected_month_end

    # Check that OOT dataset includes data only from the specified last n months
    oot_data = sample_data.loc[oot_indices]
    expected_oot_start = (
        last_date - pd.offsets.MonthBegin(n=1) - pd.offsets.MonthBegin(n=6)
    )
    assert oot_data["dtstart"].min() >= expected_oot_start
