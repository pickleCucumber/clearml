from clearml import TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=[
        "X_train",
        "y_train",
        "X_oot",
        "y_oot",
        "X_monthly",
        "y_monthly",
    ],
    cache=True,
    task_type=TaskTypes.data_processing,
)
def preproc_dataset(dataset_id, lag=3):
    import os

    import pandas as pd
    from clearml import Dataset, Task

    from src.constants import USECOLS
    from utils.preprocessing import (
        extract_oot_indices,
        get_X_y,
        prepare_time_filtered_dataset,
    )

    # Logging min and max dates for each dataset
    def log_date_range(dataset_name, indices):
        min_date = data.loc[indices, "dtstart"].min()
        max_date = data.loc[indices, "dtstart"].max()
        logger.report_text(f"{dataset_name} date range: {min_date} to {max_date}")

    target_column = "60_6mob"
    date_column = "dtstart"

    task = Task.current_task()
    logger = task.get_logger()

    # Getting dataset
    data_path = Dataset.get(dataset_id=dataset_id).get_local_copy()

    # Reading dataset
    logger.report_text("Reading dataset")
    data = pd.read_csv(
        os.path.join(data_path, "raw.csv"),
        index_col="AppId",
        parse_dates=["dtstart", "birthday"],
        usecols=USECOLS,
    )

    logger.report_text(f"Dataset shape before preprocessing: {data.shape}")
    logger.report_text(
        f"Dataset date range before preprocessing: {data['dtstart'].min()} to {data['dtstart'].max()}"
    )

    # Preparing time filtered dataset
    data = prepare_time_filtered_dataset(
        data=data, date_column=date_column, target_column=target_column
    )

    logger.report_text(f"Dataset shape after time filtering: {data.shape}")
    logger.report_text(
        f"Time-filtered dataset date range: {data['dtstart'].min()} to {data['dtstart'].max()}"
    )

    # Extracting Out of Time (OOT) indices
    train_indices, oot_indices, monthly_indices = extract_oot_indices(
        data, date_column, target_column, lag
    )

    # Splitting dataset into features (X) and target (y)
    X, y = get_X_y(data, data.index, target_column)
    log_date_range("Full dataset", data.index)

    X_train, y_train = get_X_y(data, train_indices, target_column)
    log_date_range("Training dataset", train_indices)

    X_oot, y_oot = get_X_y(data, oot_indices, target_column)
    log_date_range("OOT dataset", oot_indices)

    # Processing monthly data
    X_monthly = {}
    y_monthly = {}
    for month, month_indices in monthly_indices.items():
        X_month, y_month = get_X_y(data, month_indices, target_column)
        X_monthly[month] = X_month
        y_monthly[month] = y_month
        logger.report_text(f"Processed month: {month}, Samples: {len(month_indices)}")
        log_date_range(f"{month} dataset", month_indices)

    return X_train, y_train, X_oot, y_oot, X_monthly, y_monthly
