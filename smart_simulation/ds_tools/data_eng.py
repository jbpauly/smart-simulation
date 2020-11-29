import logging
import pathlib

import numpy as np
import pandas as pd

import daiquiri
import pandera as pa
from smart_simulation.cfg_templates import pandera_schemas as pas
from smart_simulation.ds_tools import eda

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


def create_consumption_adjustments(
    weight_series: pd.Series, adjustment_weight: float
) -> pd.Series:
    """
    Create consumption adjustments to correct for new bag arrivals in consumption calculation.
    Args:
        weight_series: Scale weights as a pandas Series.
        adjustment_weight: Weight (ounces) of new bags.

    Returns: Consumption adjustments as a pandas Series.

    """
    estimated_peaks = eda.find_weight_peaks(weight_series=weight_series)
    segments = eda.create_consumption_segments(
        weight_series=weight_series, peaks=estimated_peaks
    )
    segments_data = eda.create_segments_data(
        consumption_segments=segments, weight_series=weight_series
    )
    segments_misaligned = eda.segments_misaligned(segments_data=segments_data)
    if segments_misaligned:
        adj_segments = eda.adjust_segments(
            consumption_segments=segments, weight_series=weight_series
        )
        adj_segments_data = eda.create_segments_data(
            consumption_segments=adj_segments, weight_series=weight_series
        )
        adj_segments_misaligned = eda.segments_misaligned(
            segments_data=adj_segments_data
        )
        if adj_segments_misaligned:
            raise Exception(
                "Exact segments cannot be identified, check the weight series."
            )
        else:
            peaks = adj_segments_data.start_time[1:].values
    else:
        peaks = estimated_peaks

    adjustments = pd.Series(adjustment_weight, index=peaks, dtype=float, name="weight")
    return adjustments


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


def calculate_theoretical_weights(
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
    validate_data(consumption_series, consumption_schema)
    start_weight = float(start_weight)

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
    scale_weight_positive = calculate_theoretical_weights(
        start_weight, consumption_series
    )
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
    validate_data(weight_series, weight_schema)

    weights = weight_series.copy().loc[train_dates]
    return weights


def residual_days(consumption_series: pd.Series, residual_weight: float) -> int:
    """
    Calculate and return the remaining days of consumption until all residual product is consumed.
    Args:
        consumption_series: Consumption values as a pandas Series.
        residual_weight: Residual weight remaining.

    Returns: Remaining days or numpy nan value if remaining days is beyond the last day in consumption series.

    """
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


def all_residual_days(
    weights_consumption: pd.DataFrame, threshold: float = 0.0
) -> pd.Series:
    """
    Calculate the residual days at multiple timestamps given weights and consumption values.
    Args:
        weights_consumption: DataFrame of weight and consumption values.
        threshold: Lowest acceptable weight, used to calculate residual weight.
                   residual weight = starting weight - threshold.
    Returns: A series of residual days for all days in the weight_consumption DataFrame.
    """
    dates_index = weights_consumption.index
    remaining_days = pd.Series(
        data=0, index=dates_index, dtype=float, name="residual_days"
    )
    for day in dates_index:
        consumption = weights_consumption.consumption[day:]
        residual_weight = weights_consumption.weight[day] - threshold
        remaining_days[day] = residual_days(consumption, residual_weight)
    return remaining_days


def calcuate_consumption_avg(
    consumption_series: pd.Series, all_timesteps: bool = False
) -> float:
    """
    Calculate and return the average consumption of a consumption series.
    Args:
        consumption_series: Consumption values as a pandas Series.
        all_timesteps: Calculate the raw average if True.
                       Calculate the average of timesteps with consumption > 0 if false.

    Returns: The average consumption as a float.

    """
    avg_consumption = 0.0
    if all_timesteps:
        avg_consumption = consumption_series.mean()
    else:
        avg_consumption = consumption_series[consumption_series > 0].mean()
    return avg_consumption


def prep_daily_forecast(
    weight_series: pd.Series, threshold: float = 0.0, new_bag_weight: float = 14.0
) -> pd.DataFrame:
    """
    Create a dataset of end of day weights, daily consumption, and residual days of consumption.
    Args:
        weight_series: Scale weights as a pandas Series.
        threshold: Threshold (ounces) for last acceptable weight before 0 days of consumption.
                   Typically 0 or the average daily consumption amount.
        new_bag_weight: Weight of new bags to create a clean consumption series.
    Returns:

    """
    weights = eod_weights(weight_series=weight_series)
    consumption_adjustments = create_consumption_adjustments(
        weight_series=weights, adjustment_weight=new_bag_weight
    )
    consumption = calculate_consumption(
        weight_series=weights, adjustments=consumption_adjustments
    )
    daily_data = pd.concat([weights, consumption], axis=1)
    remaining_consumption_days = all_residual_days(
        weights_consumption=daily_data, threshold=threshold
    )
    daily_data = pd.concat([daily_data, remaining_consumption_days], axis=1)
    return daily_data
