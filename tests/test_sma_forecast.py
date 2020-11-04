import pathlib

import numpy as np
import pandas as pd

import pandera as pa
# import pytest
import pytest_mock
from smart_simulation.cfg_templates.config import package_dir
from smart_simulation.ds_tools import data_eng as de
from smart_simulation.ds_tools import sma_forecast as sma

PACKAGE_PATH = pathlib.Path(package_dir)
TEST_COMPONENTS_PATH = PACKAGE_PATH / "tests/test_components"


CONSUMPTION_SERIES = pd.Series()


def test_create_train_test_splits(mocker):
    mocker.patch("de.validate_data", return_value=True)
