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
    assert de.validate_data(series, matching_schema)
    with pytest.raises(pa.errors.SchemaError):
        assert de.validate_data(series, mismatched_schema)

    # Negative testing
    with pytest.raises(TypeError):
        assert de.validate_data(series, "not a schema")


def test_calculate_consumption():
    """
    Test calculate_consumption function from the data_eng module.
    """
    dates = pd.date_range(start="2020-01-01", periods=4, freq="1D")
    weights = [3, 2, 1, 3]
    consumption = [np.NAN, 1, 1, 1]
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

    # Negative testing
    invalid_weights = pd.Series(
        weights, index=dates, dtype=int, name="weight"
    )  # schema expects dtype=float for weight values
    with pytest.raises(pa.errors.SchemaError):
        assert de.calculate_consumption(
            weight_series=invalid_weights, adjustments=test_adjustments
        )


def test_consumption_daily():
    """
    Test consumption_daily function from data_eng module
    """
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


def test_eod_weights():
    """
    Test eod_weights function from data_eng module
    """
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
