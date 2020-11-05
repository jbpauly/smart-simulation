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
    Create test and train datasets for each date (d) to be used in a daily forecast.
    Train = day (0) to day (d - 1)
    Test = day (d) to day (d + forecast_days - 1)
    Args:
        consumption_series: Consumption as a Pandas Series.
        forecast_days: Number of days expected to forecast on each day.

    Returns: A dictionary of train and test datasets.
            Keys are timestamps.
            Values are a dictionary:
                Keys: train, test
                Values: train consumption series, test consumption series
    """
    consumption_schema = pas.consumption_series
    try:
        de.validate_data(consumption_series, consumption_schema)
    except Exception as ex:
        raise ex

    if consumption_series.index.inferred_freq != "D":
        consumption_series = de.consumption_daily(consumption_series)

    if not isinstance(forecast_days, int):
        try:
            forecast_days = int(forecast_days)
        except Exception as ex:
            raise ex

    max_forecast_size = len(consumption_series) - 1
    if forecast_days > max_forecast_size:
        logging.exception(
            f"forecast_days: {forecast_days}, must be atleast 1 day less than number of days in "
            f"consumption series: {max_forecast_size}"
        )
        raise ValueError

    forecast_dates = consumption_series.index[1 : -forecast_days + 1]
    datasets = {}
    for d_idx, day in enumerate(forecast_dates):
        forecast_d_idx = d_idx + 1
        train = consumption_series[0:forecast_d_idx]
        test = consumption_series[forecast_d_idx : forecast_d_idx + forecast_days]
        train_test = {"train": train, "test": test}
        datasets[day] = train_test
    return datasets


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
    try:
        de.validate_data(x_consumption, consumption_schema)
    except Exception as ex:
        raise ex

    if not isinstance(averaging_window, int):
        try:
            averaging_window = int(averaging_window)
        except Exception as ex:
            raise ex

    averaging_window = str(averaging_window) + "D"
    sma = x_consumption.rolling(averaging_window, min_periods=1).mean()[-1]
    predictions = pd.Series(sma, index=y_dates, dtype=float, name="pred")
    return predictions


def predict_all(train_test_splits_dict: dict, averaging_window: int) -> dict:
    """
    Predict consumption for all datasets in a train_test_splits_dict.
    Args:
        train_test_splits_dict: Dictionary of training and testing data for a given dataset.
        averaging_window: Size of the averaging window to use for forecasting.

    Returns: A dictionary of predictions.
            Keys: Timestamps (date of prediction).
            Values: Forecasts a Pandas Series.
    """
    if not isinstance(train_test_splits_dict, dict):
        logging.exception(
            f"train_test_splits_dict must be a dictionary, received a {type(train_test_splits_dict)}"
        )
        raise TypeError
    predict_dates_list = list(train_test_splits_dict.keys())
    all_predictions = {}
    for date in predict_dates_list:
        x = train_test_splits_dict[date]["train"]
        y_true = train_test_splits_dict[date]["test"]
        y_dates = y_true.index
        y_pred = predict(averaging_window, x, y_dates)
        all_predictions[date] = y_pred
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
    try:
        de.validate_data(consumption_series, consumption_schema)
    except Exception as ex:
        raise ex

    if not isinstance(start_weight, float):
        try:
            averaging_window = float(start_weight)
        except Exception as ex:
            raise ex

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
    try:
        de.validate_data(weight_series, weight_schema)
    except Exception as ex:
        raise ex

    t_weights = weight_series.copy().loc[train_dates]
    return t_weights


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
    try:
        de.validate_data(training_weights, weight_schema)
    except Exception as ex:
        raise ex

    data_dict = {}
    test_dates = consumption_series_test.index
    start_weight = training_weights[-1]
    test_theoretical_weight = calculate_test_weights(
        start_weight, consumption_series_test
    )
    test_theoretical_binary = test_weight_binary(start_weight, consumption_series_test)
    pred_consumption = predict(sma_window, consumption_series_train, test_dates)
    pred_weight = calculate_test_weights(start_weight, pred_consumption)
    pred_binary = test_weight_binary(start_weight, pred_consumption)

    data_dict["train_weight"] = training_weights
    data_dict["train_consumption"] = consumption_series_train
    data_dict["test_weight"] = test_theoretical_weight
    data_dict["test_consumption"] = consumption_series_test
    data_dict["test_binary"] = test_theoretical_binary
    data_dict["pred_weight"] = pred_weight
    data_dict["pred_consumption"] = pred_consumption
    data_dict["pred_binary"] = pred_binary

    return data_dict


def multi_test(
    weight_series: pd.Series, train_test_dict: dict, sma_window: int
) -> dict:
    """
    Forecast consumption for a all train/test datasets in train_test_dict.
    Args:
        weight_series: Full weight series associated with the train_test_dict.
        train_test_dict: Dictionary of train/test datasets.
        sma_window: Size of the averaging window to use for forecasting.

    Returns: A dictionary of sma_single_test dictionaries.
    """
    all_tests_dict = {}
    pred_dates = list(train_test_dict.keys())
    for day in pred_dates:
        day_train_test = train_test_dict[day]
        consumption_series_train = day_train_test["train"]
        consumption_series_test = day_train_test["test"]
        weights_series_train = weight_series[consumption_series_train.index].copy()
        day_pred_dict = single_test(
            weights_series_train,
            consumption_series_train,
            consumption_series_test,
            sma_window,
        )
        all_tests_dict[day] = day_pred_dict

    return all_tests_dict


def create_test_df(test_dict: dict, prediction_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create a Pandas DataFrame of test data from test_dict. Drop training data before returning DataFrame.
    Args:
        prediction_date: Date of prediction.
        test_dict: Test dict of forecast results.
    Returns: Test data DataFrame.
    """
    copy_dict = test_dict.copy()
    del copy_dict["train_weight"]
    del copy_dict["train_consumption"]
    test_df = pd.DataFrame.from_dict(copy_dict, orient="columns")
    test_df.loc[:, "date_of_prediction"] = prediction_date
    test_df.index.name = "date"
    test_df = test_df.reset_index()
    return test_df


def create_all_tests_df(multi_test_dict: dict) -> pd.DataFrame:
    """
    Create a Pandas DataFrame of test data from all test_dicts in a multi_test_dict.
    Args:
        multi_test_dict: Dictionary of test_dicts for all dates with predictions.
    Returns: Test data DataFrame.
    """
    test_dfs = {}
    pred_dates = list(multi_test_dict.keys())
    for day in pred_dates:
        test_dict = multi_test_dict[day]
        test_df = create_test_df(test_dict, day)
        test_dfs[day] = test_df
    all_test_df = pd.concat(test_dfs.values(), ignore_index=True)
    return all_test_df
