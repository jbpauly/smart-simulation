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


# TODO Complete validate_data function
# def validate_data(dataset: pd.Object, schema: pa.Schema) -> bool:
#     try:
#         schema(dataset)
#     except pa.errors.SchemaErrors:
#         raise
#     return True


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
        weight_schema(weight_series)
    except pa.errors.SchemaErrors:
        raise

    consumption = -1 * weight_series.diff().rename("consumption")
    consumption.loc[consumption == -0] = 0  # prior transform converts 0 to -0
    if adjustments is not None:
        try:
            weight_schema(adjustments)
        except pa.errors.SchemaErrors:
            raise
        consumption = consumption.add(adjustments, fill_value=0)
    return consumption
