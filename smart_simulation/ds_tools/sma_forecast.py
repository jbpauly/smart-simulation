import logging

import pandas as pd

import daiquiri
from smart_simulation.cfg_templates import pandera_schemas as pas
from smart_simulation.ds_tools import data_eng as de

daiquiri.setup(
    level=logging.INFO,
    outputs=(
        daiquiri.output.Stream(
            formatter=daiquiri.formatter.ColorFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s.%(" "funcName)s: %(message)s"
            )
        ),
    ),
)


def create_train_test_splits(consumption_series: pd.Series, forecast_days: int) -> dict:
    """
    Create test and train splits for each date (d) to be used in a daily forecast.
    Train = day (0) to day (d - 1)
    Test = day (d) to day (d + forecast_days - 1)
    Args:
        consumption_series: Consumption as a Pandas Series.
        forecast_days: Number of days expected to forecast on each day.

    Returns: A dictionary of train and test dataset splits.
            Keys are timestamps.
            Values are a dictionary:
                Keys: train, test
                Values: train consumption series, test consumption series
    """
    consumption_schema = pas.consumption_series
    de.validate_data(consumption_series, consumption_schema)

    forecast_days = int(forecast_days)

    if consumption_series.index.inferred_freq != "1D":
        consumption_series = de.consumption_daily(consumption_series)

    max_forecast_size = len(consumption_series) - 1
    if forecast_days > max_forecast_size:
        logging.error(
            f"forecast_days: {forecast_days}, must be at least 1 day less than number of days in "
            f"consumption series: {max_forecast_size}"
        )
        raise ValueError

    forecast_dates = consumption_series.index[1 : -forecast_days + 1]
    splits = {}
    for d_idx, day in enumerate(forecast_dates):
        forecast_d_idx = d_idx + 1
        train = consumption_series[0:forecast_d_idx]
        test = consumption_series[forecast_d_idx : forecast_d_idx + forecast_days]
        train_test = {"train": train, "test": test}
        splits[day] = train_test
    return splits


def predict(
    averaging_window: int, x_consumption: pd.Series, y_dates: pd.DatetimeIndex
) -> pd.Series:
    """
    Forecast consumption with a simple moving average (sma).
    Forecast is the sma of size (averaging_window) and held constant for all days in y_dates.
    Args:
        averaging_window: Size of averaging window in days as an integer.
        x_consumption: Daily consumption preceding the day of forecast as a Pandas Series.
        y_dates: Dates to forecast as a Pandas DatetimeIndex.

    Returns: Forecast of daily consumption for y_dates as a Pandas Series.
    """
    consumption_schema = pas.consumption_series
    de.validate_data(x_consumption, consumption_schema)
    averaging_window = int(averaging_window)

    averaging_window = str(averaging_window) + "D"
    sma = x_consumption.rolling(averaging_window, min_periods=1).mean()[-1]
    predictions = pd.Series(sma, index=y_dates, dtype=float, name="consumption")
    return predictions


def predict_all(train_test_splits: dict, averaging_window: int) -> dict:
    """
    Predict consumption for all splits in train_test_splits.
    Args:
        train_test_splits: Dictionary of training and testing splits for a given dataset.
        averaging_window: Size of the averaging window to use for forecasting.

    Returns: A dictionary of predictions.
            Keys: Timestamps (date of prediction).
            Values: Forecasts a Pandas Series.
    """
    if not isinstance(train_test_splits, dict):
        logging.error(
            f"train_test_splits must be a dictionary, received a {type(train_test_splits)}"
        )
        raise TypeError
    all_predictions = {}
    for day, split in train_test_splits.items():
        x = split["train"]
        y_true = split["test"]
        y_dates = y_true.index
        y_pred = predict(averaging_window, x, y_dates)
        all_predictions[day] = y_pred
    return all_predictions


def calculate_test_weights(
    start_weight: float, consumption_series: pd.Series
) -> pd.Series:
    """
    Calculate theoretical test weights based on starting weight and consumption (true or predicted)
    Args:
        start_weight: Starting weight at time of forecast.
        consumption_series: Daily consumption (true or predicted) as a Pandas Series.

    Returns: Theoretical weights series.
    """
    consumption_schema = pas.consumption_series
    de.validate_data(consumption_series, consumption_schema)
    start_weight = float(start_weight)

    weights = consumption_series.copy() * -1
    weights.iloc[0] += start_weight
    weights = weights.cumsum().rename("weight")
    return weights


def test_weight_binary(start_weight: float, consumption_series: pd.Series) -> pd.Series:
    """
    Calculate theoretical binary weight of scale based on starting weight and consumption (true or predicted).
    0 ~ scale weight <= 0
    1 ~ scale weight > 0
    Args:
        start_weight: Starting weight at time of forecast.
        consumption_series: Daily consumption (true or predicted) as a Pandas Series.

    Returns: Theoretical binary weights series.
    """
    scale_weight_positive = calculate_test_weights(start_weight, consumption_series)
    scale_weight_positive.loc[scale_weight_positive <= 0] = 0
    scale_weight_positive.loc[scale_weight_positive > 0] = 1
    return scale_weight_positive


def train_weights(weight_series: pd.Series, train_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Get the weight_series subset for training dates and return as a Pandas Series.
    Args:
        weight_series: Weights as a Pandas Series.
        train_dates: Training dates as a Pandas DataTimeIndex.

    Returns: Subset of weights for training dates.
    """
    weight_schema = pas.weight_series
    de.validate_data(weight_series, weight_schema)

    weights = weight_series.copy().loc[train_dates]
    return weights


def single_test(
    training_weights: pd.Series,
    consumption_series_train: pd.Series,
    consumption_series_test: pd.Series,
    sma_window: int,
) -> dict:
    """
    Forecast consumption for a single train/test dataset.
    Args:
        training_weights: Weights for the training dates.
        consumption_series_train: Consumption of training dates.
        consumption_series_test: True consumption values for test dates.
        sma_window: Size of the averaging window to use for forecasting.

    Returns: Dictionary of train, test, and prediction datasets.
            Datasets in dictionary: train_weight, train_consumption, test_weight, test_consumption, test_binary,
                                    pred_weight, pred_consumption, pred_binary

    """
    weight_schema = pas.weight_series
    de.validate_data(training_weights, weight_schema)

    datasets = {}
    test_dates = consumption_series_test.index
    start_weight = training_weights[-1]
    test_theoretical_weight = calculate_test_weights(
        start_weight, consumption_series_test
    )
    test_theoretical_binary = test_weight_binary(start_weight, consumption_series_test)
    pred_consumption = predict(sma_window, consumption_series_train, test_dates)
    pred_weight = calculate_test_weights(start_weight, pred_consumption)
    pred_binary = test_weight_binary(start_weight, pred_consumption)

    datasets["train_weight"] = training_weights
    datasets["train_consumption"] = consumption_series_train
    datasets["test_weight"] = test_theoretical_weight
    datasets["test_consumption"] = consumption_series_test
    datasets["test_binary"] = test_theoretical_binary
    datasets["pred_weight"] = pred_weight
    datasets["pred_consumption"] = pred_consumption
    datasets["pred_binary"] = pred_binary

    return datasets


def multi_test(
    weight_series: pd.Series, train_test_splits: dict, sma_window: int
) -> dict:
    """
    Forecast consumption for a all train/test datasets in train_test_splits.
    Args:
        weight_series: Full weight series associated with the train_test_splits.
        train_test_splits: Dictionary of train/test datasets.
        sma_window: Size of the averaging window to use for forecasting.

    Returns: A dictionary of sma_single_test dictionaries.
    """
    tests = {}
    for day, split in train_test_splits.items():
        consumption_series_train = split["train"]
        consumption_series_test = split["test"]
        weights_series_train = weight_series[consumption_series_train.index].copy()
        test_ouput = single_test(
            weights_series_train,
            consumption_series_train,
            consumption_series_test,
            sma_window,
        )
        tests[day] = test_ouput

    return tests


def create_test_df(test_output: dict, prediction_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create a Pandas DataFrame of test data from test dictionary. Drop training data before returning DataFrame.
    Args:
        prediction_date: Date of prediction.
        test_output: Test dict of forecast results.
    Returns: Test data DataFrame.
    """
    if not isinstance(test_output, dict):
        logging.error(
            f"test_output must be a dictionary. Received type: {type(test_output)}"
        )
        raise TypeError
    if not isinstance(prediction_date, pd.Timestamp):
        logging.error(
            f"prediction_date must be of type pandas.Timestamp. Received type: {type(prediction_date)}"
        )
        raise TypeError

    test = pd.DataFrame.from_dict(test_output, orient="columns")
    test = test.drop(columns=["train_weight", "train_consumption"]).dropna(
        axis=0, how="any"
    )
    test.loc[:, "date_of_prediction"] = prediction_date
    test = test.reset_index().rename(columns={"index": "date"})
    return test


def create_all_tests_df(tests_by_day: dict) -> pd.DataFrame:
    """
    Create a Pandas DataFrame of test data from all test dictionaries in multi_test_outputs.
    Args:
        tests_by_day: Dictionary of test outputs for all dates with predictions.
    Returns: Test data DataFrame.
    """
    tests = {}
    for day, test in tests_by_day.items():
        test_df = create_test_df(test, day)
        tests[day] = test_df
    all_tests = pd.concat(tests.values(), ignore_index=True)
    return all_tests
