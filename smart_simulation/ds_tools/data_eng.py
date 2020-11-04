import logging
import pathlib

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
    try:
        sim_df = pd.read_csv(
            file_path, names=columns, header=0, parse_dates=True, index_col=0
        )
    except Exception:
        raise
    return sim_df


def validate_data(
    dataset: pd.Series or pd.DataFrame, schema: pa.DataFrameSchema or pa.SeriesSchema
) -> bool:
    """
    Validate the schema of a dataset against an expected schema.
    Args:
        dataset: Pandas object (Series or DataFrame) to validate.
        schema: Expected schema of the dataset.

    Returns: True if dataset schema matches expected schema.
    """
    try:
        schema(dataset)
    except pa.errors.SchemaErrors:
        raise
    return True


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
    try:
        validate_data(weight_series, weight_schema)
    except Exception as ec:
        raise ec

    consumption = -1 * weight_series.diff().rename("consumption")
    consumption.loc[consumption == -0] = 0  # prior transform converts 0 to -0
    if adjustments is not None:
        try:
            validate_data(adjustments, weight_schema)
        except Exception as ec:
            raise ec
        consumption = consumption.add(adjustments, fill_value=0).rename("consumption")
    return consumption


def consumption_daily(consumption_series: pd.Series) -> pd.Series:
    """
    Upsample a consumption series to a daily value and return series.
    Args:
        consumption_series: Consumption values as a Pandas Series.

    Returns: Upsampled Pandas Series.
    """
    daily_consumption = consumption_series.resample("1D").sum()
    return daily_consumption


def eod_weights(weight_series: pd.Series) -> pd.Series:
    """
    Get the daily end of day weights for a weight series.
    Args:
        weight_series: Measured weights.

    Returns: last measured weight of each day in the weight_series.
    """
    eod_weight_series = weight_series.groupby([weight_series.index.date]).last()
    return eod_weight_series
