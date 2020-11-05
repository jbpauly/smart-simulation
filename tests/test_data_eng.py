import pathlib

import numpy as np
import pandas as pd

import pandera as pa
import pytest
import pytest_mock
from smart_simulation.cfg_templates.config import package_dir
from smart_simulation.ds_tools import data_eng as de

PACKAGE_PATH = pathlib.Path(package_dir)
TEST_COMPONENTS_PATH = PACKAGE_PATH / "tests/test_components"


def test_load_sim_data():
    """
    Test the load_sim_data function from data_eng module
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


def test_calculate_consumption():
    """
    Test the calculate_consumption function from the data_eng module.
    """
    weights = {
        "2020-01-01 00:00:00": 3,
        "2020-01-02 00:00:00": 2,
        "2020-01-03 00:00:00": 1,
        "2020-01-04 00:00:00": 3,
    }
    adjustments = {"2020-01-04 00:00:00": 3}
    consumption = {
        "2020-01-01 00:00:00": np.NaN,
        "2020-01-02 00:00:00": 1,
        "2020-01-03 00:00:00": 1,
        "2020-01-04 00:00:00": 1,
    }
    test_weights = pd.Series(
        weights, index=pd.to_datetime(list(weights.keys())), dtype=float, name="weight"
    )
    test_adjustments = pd.Series(
        adjustments,
        index=pd.to_datetime(list(adjustments.keys())),
        dtype=float,
        name="weight",
    )

    # Positive testing
    test_consumption = de.calculate_consumption(
        weight_series=test_weights, adjustments=test_adjustments
    )
    valid_consumption = pd.Series(
        consumption,
        index=pd.to_datetime(list(consumption.keys())),
        dtype=float,
        name="consumption",
    )
    assert test_consumption.equals(valid_consumption)

    # Negative testing
    invalid_weights = pd.Series(
        weights, index=pd.to_datetime(list(weights.keys())), dtype=int, name="weight"
    )  # schema expects dtype=float for weight values
    with pytest.raises(pa.errors.SchemaError):
        assert de.calculate_consumption(
            weight_series=invalid_weights, adjustments=test_adjustments
        )
