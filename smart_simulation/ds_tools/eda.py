import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from time_window import TimeWindow

import daiquiri
import pandera as pa
import ruptures as rpt
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


def find_break_points(
    weight_series: pd.Series, estimated_breaks: int, window_width=int, model: str = "l1"
) -> list:
    """
    Find break points in a weight measurement series using Ruptures Windowing.
    Args:
        weight_series: Weight measurements as a Pandas Series.
        estimated_breaks: Number of estimated breakpoints (i.e. number of product arrivals to customer)
        window_width: Minimum width to be used for Ruptures Windowing function.
        model: Model selected to be used with the Window Sliding Segmentation search method.

    Returns: List of break point timestamps.

    """
    weight_schema = pas.weight_series
    acceptable_models = ["l2", "l1", "rbf", "linear", "normal", "ar"]

    try:
        weight_schema(weight_series)
    except pa.errors.SchemaErrors:
        raise

    if not isinstance(estimated_breaks, int):
        logging.exception("estimated_breaks must be an integer.")
        raise TypeError
    if not isinstance(window_width, int):
        logging.exception("window_width must be an integer.")
        raise TypeError
    if not isinstance(model, str):
        logging.exception("model must be a string.")
        raise TypeError
    if model not in acceptable_models:
        logging.exception(
            f"model must an acceptable model type for Ruptures Windowing: {acceptable_models}"
        )
        raise ValueError

    algorithm = rpt.Window(width=window_width, model=model).fit(weight_series.values)
    break_points = algorithm.predict(n_bkps=estimated_breaks)
    if break_points[-1] >= len(weight_series):
        del break_points[-1]  # ruptures adds a breakpoint at end of series
    break_point_time_stamps = list(weight_series.index[break_points])
    return break_point_time_stamps


def find_weight_peaks(weight_series: pd.Series) -> list:
    """
    Find the peaks in weight data using Scipy Signal method(s).
    Args:
        weight_series: Weight measurements as a Pandas Series.
    Returns: List of break point timestamps.
    """
    weight_schema = pas.weight_series
    try:
        weight_schema(weight_series)
    except pa.errors.SchemaErrors:
        raise

    peak_data = signal.find_peaks(weight_series, plateau_size=1)
    plateau_data_dict = peak_data[1]
    left_edges = plateau_data_dict["left_edges"]
    left_edges_ts = list(weight_series.index[left_edges])

    return left_edges_ts


def create_consumption_segments(weight_series: pd.Series, peaks: list) -> dict:
    """
    Create and return a dictionary of time windows representing consumption segments of individual products.
    Args:
        weight_series: Weight measurements as a Pandas Series.
        peaks: Peaks in data as a list of time stamps.

    Returns: Dictionary of time_window.TimeWindows 'windows'.
    """
    weight_schema = pas.weight_series
    try:
        weight_schema(weight_series)
    except pa.errors.SchemaErrors:
        raise

    time_stamps = weight_series.index
    consumption_segments = {}

    start_time_stamp = time_stamps[0]
    end_index = (
        time_stamps.get_loc(peaks[0]) - 1
    )  # end the segment 1 timestamp before peak
    end_time_stamp = time_stamps[end_index]
    segment_0 = TimeWindow(start_time_stamp, end_time_stamp)
    consumption_segments[0] = segment_0

    for index, break_point in enumerate(peaks):
        start_time_stamp = break_point
        if index < len(peaks) - 1:
            end_index = (
                time_stamps.get_loc(peaks[index + 1]) - 1
            )  # end the segment 1 timestamp before peak
            end_time_stamp = time_stamps[end_index]
        else:
            end_time_stamp = time_stamps[-1]
        consumption_segments[index + 1] = TimeWindow(start_time_stamp, end_time_stamp)

    return consumption_segments


def create_segment_data(segment: TimeWindow, weight_series: pd.Series) -> dict:
    """
    Create a dictionary of segment summary information: time delta, start timestamp, end timestamp, start weight,
    end weight, max weight, min weight, max weight timestamp, min weight time.
    Args:
        segment: TimeWindow of the single segment.
        weight_series: Weight measurements as a Pandas Series.
    Returns: Summary information as a dictionary.
    """
    segment_data = {}
    time_delta = segment.delta
    start_time = segment.since
    end_time = segment.until
    start_weight = weight_series.loc[start_time]
    end_weight = weight_series.loc[end_time]
    max_weight = weight_series[start_time:end_time].max()
    min_weight = weight_series[start_time:end_time].min()
    max_weight_time = (
        weight_series[start_time:end_time].loc[
            weight_series[start_time:end_time] == max_weight
        ]
    ).index[0]
    min_weight_time = (
        weight_series[start_time:end_time].loc[
            weight_series[start_time:end_time] == min_weight
        ]
    ).index[-1]

    segment_data["time_delta"] = time_delta
    segment_data["start_time"] = start_time
    segment_data["end_time"] = end_time

    segment_data["start_weight"] = start_weight
    segment_data["end_weight"] = end_weight

    segment_data["max_weight"] = max_weight
    segment_data["min_weight"] = min_weight

    segment_data["max_weight_time"] = max_weight_time
    segment_data["min_weight_time"] = min_weight_time

    return segment_data


def create_segments_data(
    consumption_segments: dict, weight_series: pd.Series
) -> pd.DataFrame:
    """
    Create and return a Pandas DataFrame of segments summary information: time delta, start timestamp, end timestamp,
    start weight, end weight, max weight, min weight, max weight timestamp, min weight time.
    Args:
        consumption_segments: Dictionary of TimeWindow segments.
        weight_series: Weight measurements as a Pandas Series.
    Returns: Summary information as a Pandas DataFrame.
    """
    segments_data = pd.DataFrame.from_dict(
        consumption_segments, orient="index", columns=["time_window"]
    )
    segments_data["time_delta"] = [
        time_window.delta for time_window in segments_data["time_window"]
    ]
    segments_data["start_time"] = [
        time_window.since for time_window in segments_data["time_window"]
    ]
    segments_data["end_time"] = [
        time_window.until for time_window in segments_data["time_window"]
    ]

    segments_data["start_weight"] = [
        weight_series.loc[start_time] for start_time in segments_data["start_time"]
    ]
    segments_data["end_weight"] = [
        weight_series.loc[end_time] for end_time in segments_data["end_time"]
    ]

    segments_data["max_weight"] = [
        weight_series[row.start_time : row.end_time].max()
        for row in segments_data.itertuples()
    ]
    segments_data["min_weight"] = [
        weight_series[row.start_time : row.end_time].min()
        for row in segments_data.itertuples()
    ]

    segments_data["max_weight_time"] = [
        (
            (
                weight_series[row.start_time : row.end_time].loc[
                    weight_series[row.start_time : row.end_time] == row.max_weight
                ]
            ).index[0]
        )
        for row in segments_data.itertuples()
    ]
    segments_data["min_weight_time"] = [
        (
            (
                weight_series[row.start_time : row.end_time].loc[
                    weight_series[row.start_time : row.end_time] == row.min_weight
                ]
            ).index[-1]
        )
        for row in segments_data.itertuples()
    ]

    segments_data = segments_data[
        [
            "time_delta",
            "start_time",
            "max_weight_time",
            "end_time",
            "min_weight_time",
            "start_weight",
            "max_weight",
            "end_weight",
            "min_weight",
        ]
    ]

    return segments_data


def segments_misaligned(segments_data: pd.DataFrame) -> bool:
    """
    Check for misalignment in segments based off summary information. Segments should end right before a new product
    arrival.
    Args:
        segments_data: Pandas DataFrame of segments data.
    Returns: Boolean value of misalignment.
    """
    misaligned = False
    for segment_index in segments_data.index:
        if (
            segments_data.at[segment_index, "start_time"]
            != segments_data.at[segment_index, "max_weight_time"]
        ):
            misaligned = True
        if (
            segments_data.at[segment_index, "end_time"]
            != segments_data.at[segment_index, "min_weight_time"]
        ):
            misaligned = True
        if segment_index < segments_data.index[-1]:
            if (
                segments_data.at[segment_index, "end_weight"]
                >= segments_data.at[segment_index + 1, "start_weight"]
            ):
                misaligned = True
    return misaligned


def adjust_segments(consumption_segments: dict, weight_series: pd.Series) -> dict:
    """
    Realign consumption segments to end one timestamp before a new product arrives on a scale and start on timestamp
    of product arrival.
    Args:
        consumption_segments: Dictionary of TimeWindow segments.
        weight_series: Weight measurements as a Pandas Series.

    Returns: A dictionary of adjusted TimeWindow segments.
    """
    segments = consumption_segments.copy()

    for segment_index in range(len(segments) - 1):
        current_segment_data = create_segment_data(
            segments[segment_index], weight_series
        )
        next_segment_data = create_segment_data(
            segments[segment_index + 1], weight_series
        )

        current_end_weight = current_segment_data["end_weight"]
        next_start_weight = next_segment_data["start_weight"]

        if current_end_weight >= next_start_weight:
            weight_sub_range = weight_series[
                current_segment_data["min_weight_time"] : next_segment_data["end_time"]
            ]
            sub_range_values = np.array(weight_sub_range.values)
            sub_range_times = weight_sub_range.index
            peaks, plateaus = signal.find_peaks(sub_range_values, plateau_size=1)
            first_maxima_index = peaks[0]
            if plateaus["left_edges"]:
                plateaus_left_edge = plateaus["left_edges"]
                first_maxima_index = np.minimum(peaks, plateaus_left_edge)[0]

            adjusted_current_end_time = sub_range_times[first_maxima_index - 1]
            adjusted_next_start_time = sub_range_times[first_maxima_index]

            adjusted_current_segment = TimeWindow(
                current_segment_data["start_time"], adjusted_current_end_time
            )
            adjusted_next_segment = TimeWindow(
                adjusted_next_start_time, next_segment_data["end_time"]
            )

            segments[segment_index] = adjusted_current_segment
            segments[segment_index + 1] = adjusted_next_segment
        else:
            continue
    return segments


def segment_consumption_stats(
    consumption_segments: dict, consumption_series: pd.Series
) -> pd.DataFrame:
    """
    Create and return a Pandas DataFrame of  consumption segments summary information.
    Args:
        consumption_segments: Dictionary of TimeWindow segments.
        consumption_series: Consumption as a Pandas Series.
    Returns: Summary statistics for consumption segments as a Pandas DataFrame.
    """
    segments_data = pd.DataFrame.from_dict(
        consumption_segments, orient="index", columns=["time_window"]
    )
    segments_data["time_delta"] = [
        time_window.delta for time_window in segments_data["time_window"]
    ]
    segments_data["start_time"] = [
        time_window.since for time_window in segments_data["time_window"]
    ]
    segments_data["end_time"] = [
        time_window.until for time_window in segments_data["time_window"]
    ]
    segments_stats_dict = {}
    for segment in segments_data.itertuples():
        segment_consumption = consumption_series[segment.start_time : segment.end_time]
        segment_stats = segment_consumption.describe()
        segment_stats["cumulative"] = segment_consumption.values.sum()
        segments_stats_dict[segment.Index] = segment_stats.to_dict()
    stats_df = pd.DataFrame.from_dict(segments_stats_dict).T
    segments_stats = pd.concat([segments_data, stats_df], axis=1)
    segments_stats = segments_stats.drop(columns=["count", "time_window"])

    return segments_stats


def create_desired_consumption(
    servings_series: pd.Series, quantity_to_weight: tuple
) -> pd.Series:
    """
    Create theoretical desired consumption in ounces for a consumer based on their desired servings.
    Args:
        servings_series: Desired servings as a Pandas Series.
        quantity_to_weight: Conversion function and inputs for servings to ounces.

    Returns: Pandas series of desired consumption in ounces.
    """
    conversion_fn = quantity_to_weight[0]
    fn_inputs = quantity_to_weight[1]
    consumption = map(
        lambda serving: serving * (conversion_fn(*fn_inputs)), servings_series
    )
    desired_consumption = pd.Series(
        consumption,
        index=servings_series.index,
        dtype=float,
        name="desired_consumption",
    )
    return desired_consumption


def linear_weights(
    desired_consumption: pd.Series, delivery_frequency: int, bag_weight: int
) -> pd.Series:
    """
    Create theoretical weight series for a linear subscription model.
    Args:
        desired_consumption: Desired consumption in ounces as a Pandas Series.
        delivery_frequency: Frequency of delivery in days to represent subscription frequency.
        bag_weight: Weight of a new bag.

    Returns: A Pandas Series of the scale weights.
    """
    index = desired_consumption.index
    weights = pd.Series(0, index=index, dtype=float, name="weight")
    delivery_frequency = str(delivery_frequency) + "D"
    first_day = index[0]
    last_day = index[-1]
    delivery_dates = pd.date_range(first_day, last_day, freq=delivery_frequency)
    weights.loc[delivery_dates] = bag_weight

    yesterday_weight = 0
    min_weight = 0.0
    for day in index:
        end_weight = yesterday_weight + weights.at[day] - desired_consumption.at[day]
        end_weight = max(end_weight, min_weight)
        weights.loc[day] = end_weight
        yesterday_weight = end_weight
    weights = weights.round(2)
    return weights


def classify_dates(weight_series: pd.Series, bag_weight: int) -> pd.Series:
    """
    Classify the availability of product based on scale weight and expected bag weight.
    Args:
        weight_series: Scale weights as a Pandas Series.
        bag_weight: Weight of a single product.

    Returns: Classification of availability: (available, excess, unavailable) as a Pandas Series.

    """
    index = weight_series.index
    available = weight_series.index[(weight_series <= bag_weight) & (weight_series > 0)]
    excess = weight_series.index[(weight_series > bag_weight)]
    unavailable = weight_series.index[(weight_series == 0)]
    classifications = pd.Series(
        "unavailable", index=index, dtype=str, name="classification"
    )
    classifications.loc[available] = "available"
    classifications.loc[excess] = "excess"
    classifications.loc[unavailable] = "unavailable"
    return classifications


def plot_weights(
    weight_series: pd.Series, title: str, sample_label: str, bar: bool = False
):
    if not isinstance(weight_series, pd.Series):
        logging.exception("weight_series must be a Pandas Series.")
        raise TypeError

    dates = weight_series.index
    weights = weight_series.values

    fig, ax = plt.subplots(figsize=(18, 6))

    if bar:
        ax.bar(dates, weights, label=sample_label)
    else:
        ax.plot(dates, weights, label=sample_label)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set(xlabel="Date", ylabel="Weight (oz)", title=title)
    ax.grid(True)
    ax.legend()
