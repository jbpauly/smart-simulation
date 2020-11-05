import pathlib
import statistics

import pandas as pd

import pytest
import pytest_mock
from smart_simulation.cfg_templates.config import package_dir
from smart_simulation.ds_tools import data_eng as de
from smart_simulation.ds_tools import sma_forecast as sma

PACKAGE_PATH = pathlib.Path(package_dir)
TEST_COMPONENTS_PATH = PACKAGE_PATH / "tests/test_components"

# lists and pandas series for reuse in tests
WEIGHTS = [10.0, 9.0, 9.0, 8.0, 7.0, 6.0, 6.0, 6.0, 5.0, 4.0]
CONSUMPTION = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
DATES = pd.date_range(start="2020-01-01", periods=10, freq="1D")
CONSUMPTION_SERIES = pd.Series(
    CONSUMPTION, index=DATES, dtype=float, name="consumption"
)
WEIGHT_SERIES = pd.Series(WEIGHTS, index=DATES, dtype=float, name="weight")


def test_create_train_test_splits(mocker):
    """
    Test create_train_test_splits function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    forecast_size = 2
    num_pred_dates = len(CONSUMPTION_SERIES) - forecast_size
    mocker.patch.object(
        de, "validate_data", return_value=True
    )  # assume that dataset is valid

    # call on function here
    splits = sma.create_train_test_splits(CONSUMPTION_SERIES, forecast_size)
    keys = list(splits.keys())
    first_pred_date = keys[0]
    first_split = splits[first_pred_date]
    train_series = first_split["train"]
    test_series = first_split["test"]

    # Positive testings
    assert len(splits) == num_pred_dates
    assert type(first_pred_date) == pd.Timestamp
    assert test_series.index[0] == first_pred_date
    assert len(test_series) == forecast_size
    assert train_series.index[-1] + pd.Timedelta("1D") == test_series.index[0]
    assert train_series.index == CONSUMPTION_SERIES[:first_pred_date].index[:-1]
    assert sma.create_train_test_splits(CONSUMPTION_SERIES, "7")

    # Negative testing

    # expect forecast days to be an integer
    with pytest.raises(Exception):
        assert sma.create_train_test_splits(CONSUMPTION_SERIES, forecast_days="7D")

    # forecast_days needs to be at least 1 day less than the number of days in the consumption series
    # CONSUMPTION_SERIES is 10 days long
    with pytest.raises(ValueError):
        assert sma.create_train_test_splits(CONSUMPTION_SERIES, forecast_days=20)


def test_predict(mocker):
    """
    Test predict function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=True
    )  # assume dataset is true
    averaging_window = 2
    avg = statistics.mean(CONSUMPTION[4 - averaging_window : 4])
    x_consumption = CONSUMPTION_SERIES[:4]
    y_dates = CONSUMPTION_SERIES.index[4:]

    # call on function here
    predictions = sma.predict(averaging_window, x_consumption, y_dates)

    # Positive testing
    assert predictions.index.equals(y_dates)
    # sma forecast the the sma average on the day before prediction and held constant through all forecasted days
    assert predictions[0] == avg

    # Negative testing
    # expect an integer for averaging_window
    with pytest.raises(Exception):
        assert sma.predict(
            averaging_window="2D", x_consumption=x_consumption, y_dates=y_dates
        )


def test_predict_all(mocker):
    """
    Test predict_all function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        sma, "predict", return_value=["a", "b"]
    )  # ensure an expected value is returned
    series = pd.Series([0, 1])
    dummy_split = {"train": series, "test": series}
    train_test_splits = {"day1": dummy_split, "day2": dummy_split}

    # call on function here
    all_predictions = sma.predict_all(train_test_splits, 1)

    # Positive testing
    assert (
        train_test_splits.keys() == all_predictions.keys()
    )  # keys (prediction timestamp) should be reused

    # Negative testing
    # expect a dictionary for train_test_splits
    with pytest.raises(TypeError):
        assert sma.predict_all(train_test_splits="not a dictionary", averaging_window=1)


def test_calculate_test_weights(mocker):
    """
    Test calculate_test_weights function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=True
    )  # assume dataset is valid
    start_weight = float(10)
    theoretical_weights = [10, 9, 9, 8, 7, 6, 6, 6, 5, 4]
    valid_test_weights = pd.Series(
        theoretical_weights, index=DATES, dtype=float, name="weight"
    )

    # call on function here
    test_weights = sma.calculate_test_weights(start_weight, CONSUMPTION_SERIES)

    # Positive testing
    assert test_weights.equals(valid_test_weights)

    # TODO fix assert with non float
    # assert sma.calculate_test_weights(int(10), CONSUMPTION_SERIES)  # int should be successfully converted to integer

    # Negative testing
    # expect a float object for start_weight
    with pytest.raises(Exception):
        assert sma.calculate_test_weights(
            start_weight="not a float", consumption_series=CONSUMPTION_SERIES
        )


def test_weights_binary(mocker):
    """
    Test weights_binary function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    theoretical_weights = [5, 4, 4, 3, 3, 1, 1, 1, 0, -1]
    weight_series = pd.Series(
        theoretical_weights, index=DATES, dtype=float, name="weight"
    )
    theoretical_binary = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    valid_binary_weights = pd.Series(
        theoretical_binary, index=DATES, dtype=float, name="weight"
    )
    start_weight = float(5)

    # ensure an expected value is returned
    mocker.patch.object(sma, "calculate_test_weights", return_value=weight_series)

    # call on function here
    test_binary_weights = sma.test_weight_binary(start_weight, CONSUMPTION_SERIES)

    # Positive testing

    # All theoretical weights > 0 should have a binary value of 1
    # All theoretical weights <= 0 should have a binary value of 0
    assert test_binary_weights.equals(valid_binary_weights)


def test_train_weights(mocker):
    """
    Test train_weights function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=True
    )  # assume dataset is valid
    weight_series = WEIGHT_SERIES
    dates = DATES[2:]

    # call on function here
    train_weights = sma.train_weights(weight_series, dates)

    # Positive testing
    assert train_weights.index.equals(dates)
    assert train_weights.iloc[-1] == weight_series.iloc[-1]


def test_single_test(mocker):
    """
    Test single_test function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    size_train = 2
    training_weights = WEIGHT_SERIES[:size_train]
    consumption_series_train = CONSUMPTION_SERIES[:size_train]
    consumption_series_test = CONSUMPTION_SERIES[size_train:]
    mocker.patch.object(
        de, "validate_data", return_value=True
    )  # assume dataset is valid
    mocker.patch.object(
        sma, "calculate_test_weights", return_value="a"
    )  # ensure expected value is returned
    mocker.patch.object(
        sma, "test_weight_binary", return_value="b"
    )  # ensure expected value is returned
    mocker.patch.object(
        sma, "predict", return_value="c"
    )  # ensure expected value is returned

    # call on function here
    test_output = sma.single_test(
        training_weights,
        consumption_series_train,
        consumption_series_test,
        sma_window=2,
    )

    # Positive testing

    # function should return a dictionary with 8 keys:
    #   train_weight, train_consumption, test_weight, test_consumption,
    #   test_binary, pred_weight, pred_consumption, pred_binary
    assert len(test_output) == 8
    # test the objects passed to function are properly returned as a dictionary item
    assert test_output["train_weight"].equals(training_weights)
    # test the outputs from other called on functions are properly returned as a dictionary item
    assert test_output["pred_binary"] == "b"


def test_multi_test(mocker):
    """
    Test multi_test function from sma_forecast module
    Args:
        mocker: object used to patch outside function calls.
    """
    series = pd.Series([0, 1])
    mocker.patch.object(
        sma, "single_test", return_value=series
    )  # ensure an expected value is returned
    dummy_split = {"train": series, "test": series}
    train_test_splits = {"day1": dummy_split, "day2": dummy_split}

    # call on function here
    tests = sma.multi_test(series, train_test_splits, 2)

    # Positive testing
    # function returns a dictionary of dictionaries
    # keys should be the same from the train_test_splits
    # values should be the return of sma.single_test(), which is patched in this test
    assert list(tests.items()) == [("day1", series), ("day2", series)]


def test_create_test_df():
    """
    Test create_test_df function from sma_forecast module
    """
    prediction_date = pd.Timestamp("2020-01-05")
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06")])
    series = pd.Series(["a", "b"], index=dates)
    output = {
        "pred": series,
        "test": series,
        "train_weight": series,
        "train_consumption": series,
    }

    # call on function here
    test_df = sma.create_test_df(output, prediction_date)

    # Positive testing
    # returned dataframe should have the columns as follows:
    #   - keys from the output dictionary, except 'train_weight' and 'train_consumption'
    #   - two additional columns: 'date' and 'date_of_prediction'
    assert list(test_df.columns) == ["date", "pred", "test", "date_of_prediction"]
    # index should be sequential integers
    assert list(test_df.index.values) == [0, 1]
    # date column should have values of the dates of datesets in the output dictionary
    assert list(test_df.date.values) == list(dates.values)

    # Negative testing
    # prediction_date should be a pandas.Timestamp
    with pytest.raises(TypeError):
        assert sma.create_test_df(output, prediction_date="not a timestamp")
    # test_output should be a dictionary
    with pytest.raises(TypeError):
        assert sma.create_test_df(
            test_output="not a dictionary", prediction_date=prediction_date
        )
