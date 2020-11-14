import logging
import pathlib

import numpy as np
import pandas as pd

import daiquiri
import pandera as pa
from smart_simulation.cfg_templates import pandera_schemas as pas

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


def load_sim_data(file_path: pathlib.Path, columns: list) -> pd.DataFrame:
    """
    Read and return simulation file from csv to a Pandas DataFrame.
    Args:
        file_path: Path to file.
        columns: Columns to include in DataFrame.

    Returns: The DataFrame.
    """
    sim_df = pd.read_csv(
        file_path, names=columns, header=0, parse_dates=True, index_col=0
    )
    return sim_df


def validate_data(
    dataset: pd.Series or pd.DataFrame, schema: pa.DataFrameSchema or pa.SeriesSchema
) -> bool:
    """
    Validate the schema of a dataset against an expected schema.
    Args:
        dataset: Pandas object (Series or DataFrame) to validate.
        schema: Expected schema of the dataset.
    """

    if not isinstance(schema, pa.SeriesSchema or pa.DataFrameSchema):
        logging.error(f"schema must be a pandera schema. received a {type(schema)}")
        raise TypeError
    schema(dataset)
    return


def calculate_consumption(
    weight_series: pd.Series, adjustments: pd.Series = None
) -> pd.Series:
    """
    Calculate the consumption from scale weight measurements and return as a Pandas Series.
    Args:
        weight_series: Measured weights.
        adjustments: Adjustments to raw consumption calculation (i.e. account for new product arrival on scale).

    Returns: Consumption as a Pandas Series.
    """
    weight_schema = pas.weight_series
    validate_data(weight_series, weight_schema)

    consumption = -1 * weight_series.diff().rename("consumption").fillna(0)
    consumption.loc[consumption == -0] = 0  # prior transform converts 0 to -0
    if adjustments is not None:
        validate_data(adjustments, weight_schema)
        consumption = consumption.add(adjustments, fill_value=0).rename("consumption")
    return consumption


def consumption_daily(consumption_series: pd.Series) -> pd.Series:
    """
    Upsample a consumption series to a daily value and return series.
    Args:
        consumption_series: Consumption values as a Pandas Series.

    Returns: Upsampled Pandas Series.
    """
    consumption_schema = pas.consumption_series
    validate_data(consumption_series, consumption_schema)
    daily_consumption = consumption_series.resample("1D").sum()
    return daily_consumption


def eod_weights(weight_series: pd.Series) -> pd.Series:
    """
    Get the daily end of day weights for a weight series.
    Args:
        weight_series: Measured weights.

    Returns: last measured weight of each day in the weight_series.
    """
    weight_schema = pas.weight_series
    validate_data(weight_series, weight_schema)
    eod_weight_series = weight_series.resample("1D").last()
    return eod_weight_series


def residual_days(consumption_series: pd.Series, residual_weight: float = 0.0):
    start_timestamp = consumption_series.index[0]
    consumption_cumsum = (
        consumption_series.cumsum() - consumption_series.values[0]
    )  # remove consumption incurred before timestamp 0
    threshold_crossed = consumption_cumsum.ge(
        residual_weight
    )  # boolean array, True where cumsum >= residual weight
    if True in threshold_crossed.values:
        residual_end_timestamp = consumption_cumsum[threshold_crossed].index[
            0
        ]  # first True case
        days_remaining = pd.Timedelta(
            residual_end_timestamp - start_timestamp
        ) / pd.Timedelta("1D")
    else:
        days_remaining = np.nan
    return days_remaining


def all_residual_days(weights_consumption: pd.DataFrame, threshold: float = 0):
    dates_index = weights_consumption.index
    remaining_days = pd.Series(
        data=0, index=dates_index, dtype=float, name="residual_days"
    )
    for day in dates_index:
        consumption = weights_consumption.consumption[day:]
        residual_weight = weights_consumption.weight[day] - threshold
        remaining_days[day] = residual_days(consumption, residual_weight)
    return remaining_days
