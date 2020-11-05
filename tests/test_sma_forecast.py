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

WEIGHTS = [10.0, 9.0, 9.0, 8.0, 7.0, 6.0, 6.0, 6.0, 5.0, 4.0]
CONSUMPTION = [0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
DATES = pd.date_range(start="2020-01-01", periods=10, freq="1D")
CONSUMPTION_SERIES = pd.Series(
    CONSUMPTION, index=DATES, dtype=float, name="consumption"
)
WEIGHT_SERIES = pd.Series(WEIGHTS, index=DATES, dtype=float, name="weight")


def test_create_train_test_splits(mocker):
    forecast_size = 2
    num_pred_dates = len(CONSUMPTION_SERIES) - forecast_size
    mocker.patch.object(de, "validate_data", return_value=True)
    tts_dicts = sma.create_train_test_splits(CONSUMPTION_SERIES, forecast_size)
    keys = list(tts_dicts.keys())
    first_pred_date = keys[0]
    tts_first_date = tts_dicts[first_pred_date]
    train_series = tts_first_date["train"]
    test_series = tts_first_date["test"]

    # Positive testings
    assert len(tts_dicts) == num_pred_dates
    assert type(first_pred_date) == pd.Timestamp
    assert test_series.index[0] == first_pred_date
    assert len(test_series) == forecast_size
    assert train_series.index[-1] + pd.Timedelta("1D") == test_series.index[0]
    assert train_series.index == CONSUMPTION_SERIES[:first_pred_date].index[:-1]
    assert sma.create_train_test_splits(CONSUMPTION_SERIES, "7")

    # Negative testing
    with pytest.raises(Exception):
        assert sma.create_train_test_splits(CONSUMPTION_SERIES, "7D")

    with pytest.raises(ValueError):
        assert sma.create_train_test_splits(CONSUMPTION_SERIES, 20)


def test_predict(mocker):
    mocker.patch.object(de, "validate_data", return_value=True)
    averaging_window = 2
    avg = statistics.mean(CONSUMPTION[4 - averaging_window : 4])
    x_consumption = CONSUMPTION_SERIES[:4]
    y_dates = CONSUMPTION_SERIES.index[4:]
    predictions = sma.predict(averaging_window, x_consumption, y_dates)

    # Positive testing
    assert predictions.index.equals(y_dates)
    assert predictions[0] == avg

    # Negative testing
    with pytest.raises(Exception):
        assert sma.predict("2D", x_consumption, y_dates)


def test_predict_all(mocker):
    mocker.patch.object(sma, "predict", return_value=["a", "b"])
    series = pd.Series([0, 1])
    dummy_dict = {"train": series, "test": series}
    train_test_splits = {"day1": dummy_dict, "day2": dummy_dict}
    all_predictions = sma.predict_all(train_test_splits, 1)

    # Positive testing
    assert train_test_splits.keys() == all_predictions.keys()

    # Negative testing
    with pytest.raises(TypeError):
        assert sma.predict_all("not a dictionary", 1)


def test_calculate_test_weights(mocker):
    mocker.patch.object(de, "validate_data", return_value=True)
    start_weight = float(10)
    theoretical_weights = [10, 9, 9, 8, 7, 6, 6, 6, 5, 4]
    valid_test_weights = pd.Series(
        theoretical_weights, index=DATES, dtype=float, name="weight"
    )
    test_weights = sma.calculate_test_weights(start_weight, CONSUMPTION_SERIES)

    # Positive testing
    assert test_weights.equals(valid_test_weights)

    # TODO fix assert with non float
    # assert sma.calculate_test_weights(int(10), CONSUMPTION_SERIES)

    # Negative testing
    with pytest.raises(Exception):
        assert sma.calculate_test_weights("not a float", CONSUMPTION_SERIES)


def test_weights_binary(mocker):
    theoretical_weights = [10, 9, 9, 8, 7, 6, 6, 6, 5, 4]
    weight_series = pd.Series(
        theoretical_weights, index=DATES, dtype=float, name="weight"
    )
    theoretical_binary = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    valid_binary_weights = pd.Series(
        theoretical_binary, index=DATES, dtype=float, name="weight"
    )
    start_weight = float(10)
    mocker.patch.object(sma, "calculate_test_weights", return_value=weight_series)

    test_binary_weights = sma.test_weight_binary(start_weight, CONSUMPTION_SERIES)

    # Positive testing
    assert test_binary_weights.equals(valid_binary_weights)


def test_train_weights(mocker):
    mocker.patch.object(de, "validate_data", return_value=True)
    weight_series = WEIGHT_SERIES
    dates = DATES[2:]
    train_weights = sma.train_weights(weight_series, dates)

    # Positive testing
    assert train_weights.index.equals(dates)
    assert train_weights.iloc[-1] == weight_series.iloc[-1]


def test_single_test(mocker):
    size_train = 2
    training_weights = WEIGHT_SERIES[:size_train]
    consumption_series_train = CONSUMPTION_SERIES[:size_train]
    consumption_series_test = CONSUMPTION_SERIES[size_train:]
    mocker.patch.object(de, "validate_data", return_value=True)
    mocker.patch.object(sma, "calculate_test_weights", return_value="a")
    mocker.patch.object(sma, "test_weight_binary", return_value="b")
    mocker.patch.object(sma, "predict", return_value="c")

    test_dict = sma.single_test(
        training_weights, consumption_series_train, consumption_series_test, 2
    )

    # Positive testing
    assert len(test_dict) == 8
    assert test_dict["train_weight"].equals(training_weights)
    assert test_dict["pred_binary"] == "b"
