import pathlib

import numpy as np
import pandas as pd

import pandera as pa
import pytest
import pytest_mock
from smart_simulation.cfg_templates import pandera_schemas as ps
from smart_simulation.cfg_templates.config import package_dir
from smart_simulation.ds_tools import data_eng as de

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


def test_load_sim_data():
    """
    Test load_sim_data function from data_eng module
    """
    valid_file = TEST_COMPONENTS_PATH / "test_weights.csv"
    valid_columns = ["weight"]
    invalid_columns = ["not", "in", "file"]
    test_output = de.load_sim_data(valid_file, valid_columns)

    # Positive testing
    assert list(test_output.columns) == valid_columns
    assert test_output.index.dtype == np.dtype("datetime64[ns]")

    # Negative testing
    with pytest.raises(Exception):
        assert de.load_sim_data(valid_file, invalid_columns)


def test_validate_data():
    """
    Test validate_data function from data_eng module
    """
    values = [0, 1]
    dates = pd.DatetimeIndex([pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06")])
    series = pd.Series(values, index=dates, dtype=float, name="consumption")
    matching_schema = ps.consumption_series
    mismatched_schema = ps.weight_series

    # Positive testing
    assert de.validate_data(series, matching_schema) is None

    # Negative testing
    with pytest.raises(pa.errors.SchemaError):
        assert de.validate_data(series, mismatched_schema)
    with pytest.raises(TypeError):
        assert de.validate_data(series, "not a schema") is not None


def test_calculate_consumption(mocker):
    """
    Test calculate_consumption function from the data_eng module.
        Args:
            mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=None
    )  # assume dataset is valid
    dates = pd.date_range(start="2020-01-01", periods=4, freq="1D")
    weights = [3, 2, 1, 3]
    consumption = [0, 1, 1, 1]
    adjustments = [3]
    adjustment_dates = [pd.Timestamp("2020-01-04")]

    test_weights = pd.Series(weights, index=dates, dtype=float, name="weight")
    test_adjustments = pd.Series(
        adjustments, index=adjustment_dates, dtype=float, name="weight",
    )

    # Positive testing
    test_consumption = de.calculate_consumption(
        weight_series=test_weights, adjustments=test_adjustments
    )
    # Consumption should be the difference between the weights + any specfied adjustments
    valid_consumption = pd.Series(
        consumption, index=dates, dtype=float, name="consumption",
    )
    assert test_consumption.equals(valid_consumption)


def test_consumption_daily(mocker):
    """
    Test consumption_daily function from data_eng module
        Args:
            mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=None
    )  # assume dataset is valid
    dates_12h = pd.date_range(start="2020-01-01", end="2020-01-02", freq="12h")
    dates_1d = pd.date_range(start="2020-01-01", end="2020-01-02", freq="1D")
    consumption_12h = [1, 1, 1]
    consumption_1d = [2, 1]

    consumption_series_12h = pd.Series(
        consumption_12h, index=dates_12h, dtype=float, name="consumption"
    )  # input has two values (1,1) with date 2020-01-01
    consumption_series_1d = pd.Series(
        consumption_1d, index=dates_1d, dtype=float, name="consumption"
    )  # expected output value of 2020-01-01 should be 2 (1+1)

    assert de.consumption_daily(consumption_series_12h).equals(consumption_series_1d)


def test_eod_weights(mocker):
    """
    Test eod_weights function from data_eng module
        Args:
            mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=None
    )  # assume dataset is valid
    dates_12h = pd.date_range(start="2020-01-01", end="2020-01-02", freq="12h")
    dates_1d = pd.date_range(start="2020-01-01", end="2020-01-02", freq="1d")
    weights = [1, 2, 3]
    weights_at_eod = [2, 3]

    # input has two values (1, 2) with date of 2020-01-01
    weight_series_12h = pd.Series(weights, index=dates_12h, dtype=float, name="weight")
    # expected output should only return the last value (2) of 2020-01-01
    weight_series_eod = pd.Series(
        weights_at_eod, index=dates_1d, dtype=float, name="weight"
    )

    assert de.eod_weights(weight_series_12h).equals(weight_series_eod)


def test_calculate_theoretical_weights(mocker):
    """
    Test calculate_test_weights function from data_eng module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=None
    )  # assume dataset is valid
    start_weight = float(10)
    theoretical_weights = [10, 9, 9, 8, 7, 6, 6, 6, 5, 4]
    valid_test_weights = pd.Series(
        theoretical_weights, index=DATES, dtype=float, name="weight"
    )

    # call on function here
    test_weights = de.calculate_theoretical_weights(start_weight, CONSUMPTION_SERIES)

    # Positive testing
    assert test_weights.equals(valid_test_weights)
    assert (
        de.calculate_theoretical_weights(int(10), CONSUMPTION_SERIES) is not None
    )  # int should be successfully converted to integer

    # Negative testing
    # expect a float object for start_weight
    with pytest.raises(ValueError):
        assert de.calculate_theoretical_weights(
            start_weight="not a float", consumption_series=CONSUMPTION_SERIES
        )


def test_weights_binary(mocker):
    """
    Test weights_binary function from data_eng module
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
    mocker.patch.object(de, "calculate_theoretical_weights", return_value=weight_series)

    # call on function here
    test_binary_weights = de.test_weight_binary(start_weight, CONSUMPTION_SERIES)

    # Positive testing

    # All theoretical weights > 0 should have a binary value of 1
    # All theoretical weights <= 0 should have a binary value of 0
    assert test_binary_weights.equals(valid_binary_weights)


def test_train_weights(mocker):
    """
    Test train_weights function from data_eng module
    Args:
        mocker: object used to patch outside function calls.
    """
    mocker.patch.object(
        de, "validate_data", return_value=None
    )  # assume dataset is valid
    weight_series = WEIGHT_SERIES
    dates = DATES[2:]

    # call on function here
    train_weights = de.train_weights(weight_series, dates)

    # Positive testing
    assert train_weights.index.equals(dates)
    assert train_weights.iloc[-1] == weight_series.iloc[-1]
