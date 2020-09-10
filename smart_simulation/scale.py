import logging
import pathlib
import random

import numpy as np
import pandas as pd

import daiquiri
import smart_simulation.consumer as consumer
from smart_simulation.cfg_templates import auxiliary

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


def upsample(dataset, consumption_window_template: str) -> pd.DataFrame:
    """
    Upsample a dataset of daily servings to a specified frequency and distribute servings based on configured
    probabilities

    Args:
        dataset: Dataset of time stamps and servings (consumed) by consumer
        consumption_window_template: Upsampled time windows and probabilities

    Returns:
        Upsampled data as a Pandas DataFrame

    """

    consumption_window = auxiliary.consumption_windows[consumption_window_template]
    time_stamps = list(consumption_window.keys())
    weights = list(consumption_window.values())
    frequency = pd.infer_freq(time_stamps)

    upsampled_time_stamps = pd.date_range(
        start=dataset.index[0],
        end=(dataset.index[-1] + pd.Timedelta("1 days") - pd.Timedelta(frequency)),
        freq=frequency,
    )

    upsampled_data = pd.DataFrame(index=upsampled_time_stamps).assign(servings=0)

    for day in dataset.index:
        for serving in range(dataset.loc[day]["servings"]):
            period = random.choices(population=time_stamps, weights=weights, k=1)[0]
            time_stamp = day + pd.to_timedelta(period)
            upsampled_data.at[time_stamp, "servings"] += 1

    return upsampled_data


def create_weight_data(dataset: pd.DataFrame, arrival_template: str) -> pd.DataFrame:
    """
    Create weight data based on consumption and shipment arrivals
    Args:
        dataset: Dataset of time stamps and servings (consumed) by consumer
        arrival_template: Arrival windows and probabilities used for random selection of new product arrival on scale

    Returns:
        Pandas DataFrame of weight measurements

    """
    arrival_options = auxiliary.arrival_options[arrival_template]
    arrival_options_weights = tuple(
        auxiliary.arrival_options_weights[arrival_template].values()
    )
    stock_weight = auxiliary.stock_weight
    quantity_to_weight = auxiliary.quantity_to_weight
    data = dataset.copy()
    data = data.assign(weight=0.0)
    weight_col = data.columns.get_loc("weight")
    servings_col = data.columns.get_loc("servings")

    data.iat[0, weight_col] = stock_weight

    time_step = 1
    while time_step < len(data.index):
        previous_step_weight = data.iat[time_step - 1, weight_col]
        consumption_servings = data.iat[time_step, servings_col]

        if previous_step_weight == 0.0:
            resupply_arrival = random.choices(
                population=[*arrival_options], weights=arrival_options_weights, k=1
            )[0]
            arrival_timing = arrival_options[resupply_arrival][0](
                *arrival_options[resupply_arrival][1]
            )
            time_step = time_step + arrival_timing
            if time_step > len(data.index) + 1:
                continue
            if time_step < 1:
                time_step = time_step - arrival_timing
                continue

            data.iat[time_step, weight_col] += stock_weight
            time_step += 1
            previous_step_weight = data.iat[time_step - 1, weight_col]
            consumption_servings = data.iat[time_step, servings_col]

        if consumption_servings == 0:
            data.iat[time_step, weight_col] = data.iat[time_step - 1, weight_col]

        serving_weight = quantity_to_weight[0](*quantity_to_weight[1])
        consumption_weight = serving_weight * consumption_servings

        if consumption_weight > previous_step_weight:
            data.iat[time_step, weight_col] = 0.0
        else:
            data.iat[time_step, weight_col] = previous_step_weight - consumption_weight

        time_step += 1

    return data


def calibration_error(
    dataset: pd.DataFrame, start_time_index: int, count_time_steps: int
):
    """
    Create calibration error type edits to a range of weight data
    Args:
        dataset: Dataset of weight data
        start_time_index: Index in the dataset to start the calibration error
        count_time_steps: Number of consecutive time steps to hold the calibration error
    """
    weight_range = auxiliary.scale_calibration_error_range["Standard"]
    weight_delta = random.uniform(*weight_range)
    end_time_index = start_time_index + count_time_steps
    weight_col = dataset.columns.get_loc("weight")
    dataset.iloc[start_time_index:end_time_index, weight_col] += weight_delta


def signal_removal(dataset: pd.DataFrame, start_time_index: int, count_time_steps: int):
    """
    Remove weight data from a dataset to mimic expected issues like scale batteries dying, network disconnection, etc.
    Args:
        dataset: Dataset of weight data
        start_time_index: Index in the dataset to start the signal removal
        count_time_steps: Number of consecutive time steps to hold the signal removal
    """
    end_time_index = start_time_index + count_time_steps
    weight_col = dataset.columns.get_loc("weight")
    dataset.iloc[start_time_index:end_time_index, weight_col] = np.nan


def full_change(
    dataset: pd.DataFrame, percent_to_change: int, days_max: int, change_function
) -> pd.DataFrame:
    """
    Change a dataset of weight data up to a percent of instances
    Args:
        dataset: Dataset of weight measurements to change
        percent_to_change: Percent of dataset to change
        days_max: Maximum number of consecutive days allowable for signal calibration or removal changes
        change_function: Function to change data (calibration or signal removal)

    Returns:
        Pandas DataFrame of changed dataset
    """
    changed_data = dataset.copy()

    frequency = pd.infer_freq(changed_data.index)
    one_day = pd.Timedelta("1 days")
    instances_per_day = one_day / frequency
    instance_max = instances_per_day * days_max

    instances = changed_data.index.copy()
    available_instances = pd.Series(data=[True] * len(instances), index=instances)
    available_instances[0] = False
    buffer = 1
    percent_changed = 0

    while percent_changed < percent_to_change:

        gbs = available_instances[available_instances == True].groupby(
            (available_instances != True).cumsum()
        )

        gbs_max = gbs.size().max()  # get max group length

        if (
            gbs_max < instance_max
        ):  # if max length < instance_max .. instances_max = max length
            instance_max = gbs_max

        time_span = random.randint(1, instance_max)  # randomly select delta time length

        group_availability = gbs.size() >= time_span
        available_increments = group_availability[group_availability == True]

        group_index = available_increments.sample(1).index[0]

        group = gbs.get_group(group_index)

        if time_span == len(group):
            random_start = group
        else:
            random_start = group[0:-time_span].sample(1)

        start_time_step = random_start.index[0]
        start_time_index = available_instances.index.get_loc(start_time_step)

        change_function(*(changed_data, start_time_index, time_span))

        available_instances[
            start_time_index - buffer : start_time_index + time_span + buffer
        ] = False

        percent_changed = (available_instances.value_counts(normalize=True) * 100)[
            False
        ]

    return changed_data


def main():
    path = pathlib.Path.cwd() / "outputs"
    daily_data = pd.read_csv(path / "daily_servings.csv", parse_dates=True, index_col=0)
    upsampled_data = upsample(daily_data, "Standard")
    weight_data = create_weight_data(upsampled_data, "Standard")
    with_calibration_error = full_change(weight_data, 10, 5, calibration_error)
    with_signal_removal = full_change(with_calibration_error, 5, 5, signal_removal)
    consumer.write_output(with_signal_removal, path, "full_transform")


if __name__ == "__main__":
    main()
