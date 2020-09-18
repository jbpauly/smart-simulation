import logging
import pathlib
import random

import numpy as np
import pandas as pd

import daiquiri
import smart_simulation.consumer as consumer
from smart_simulation.cfg_templates import products

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

    consumption_window = products.consumption_windows[consumption_window_template]
    consumption_times = list(consumption_window.keys())
    consumption_times_probabilities = list(consumption_window.values())
    frequency = pd.infer_freq(consumption_times)

    upsampled_time_stamps = pd.date_range(
        start=dataset.index.min(),
        end=(dataset.index.max() + pd.Timedelta("1 days") - pd.Timedelta(frequency)),
        freq=frequency,
    )  # To account a full day of data on the last day, extend end date by 1 day and remove 1 time stamp (midnight)

    upsampled_data = pd.DataFrame({"servings": 0}, index=upsampled_time_stamps)

    for day in dataset.index:
        for serving in range(dataset.loc[day]["servings"]):
            period = random.choices(
                population=consumption_times,
                weights=consumption_times_probabilities,
                k=1,
            )[0]
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
    arrival_options = products.arrival_options[arrival_template]
    arrival_classifications = [*arrival_options]
    arrival_classifications_weight = tuple(
        products.arrival_options_weights[arrival_template].values()
    )
    stock_weight = products.stock_weight
    quantity_to_weight = products.quantity_to_weight
    data = dataset.copy()
    servings = data.servings
    num_time_stamps = len(data.index)
    weights = pd.Series(data=num_time_stamps * [0], index=data.index, dtype=float)
    weights.iat[0] = stock_weight

    time_step = 1
    while time_step < weights.size:
        previous_step_weight = weights.iat[time_step - 1]
        consumption_servings = servings.iat[time_step]

        if previous_step_weight == 0.0:
            # random.choices() returns a list of 'k' number selections, just need string value
            resupply_classification = random.choices(
                population=arrival_classifications,
                weights=arrival_classifications_weight,
                k=1,
            )[0]

            arrival_option = arrival_options[resupply_classification]
            rand_number_function = arrival_option[0]
            function_inputs = arrival_option[1]
            resupply_timing = rand_number_function(*function_inputs)

            time_step = time_step + resupply_timing
            if (
                time_step > weights.size - 1
            ):  # last delivery after end of time range in dataset
                break

            # Todo change to check 'previous delivery time step'
            if time_step < 1:  # delivery must be within dataset time range
                time_step = (
                    time_step - resupply_timing
                )  # reset to previous time_step and try again
                continue

            weights.iat[time_step] += stock_weight
            time_step += 1
            previous_step_weight = weights.iat[time_step - 1]
            consumption_servings = servings.iat[time_step]

        if consumption_servings == 0:
            weights.iat[time_step] = weights.iat[time_step - 1]

        serving_weight = quantity_to_weight[0](*quantity_to_weight[1])
        consumption_weight = serving_weight * consumption_servings

        if consumption_weight > previous_step_weight:
            weights.iat[time_step] = 0.0
        else:
            weights.iat[time_step] = previous_step_weight - consumption_weight

        time_step += 1

    return weights.to_frame(name="weight")


def calibration_error():
    """
    Generate random number to represent delta of calibration error of weight scale

    Returns:
        Random value within range of preset calibration errors
    """
    weight_range = products.scale_calibration_error_range["Standard"]
    weight_delta = random.uniform(*weight_range)
    return weight_delta


def signal_removal():
    """
    Return a value to mimic removal of weight signal scenarios like network disconnection, dead battery of scale, etc.

    Returns:
        Numpy nan value
    """
    return np.nan


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
        New Pandas Dataframe of weight data after changes
    """
    changed_data = dataset.copy()
    new_weights = pd.Series(data=dataset["weight"], index=dataset.index)
    frequency = pd.infer_freq(new_weights.index)
    one_day = pd.Timedelta("1 days")
    instances_per_day = one_day / frequency
    instance_max = instances_per_day * days_max

    instances = changed_data.index.copy()
    available_instances = pd.Series(data=[True] * len(instances), index=instances)
    available_instances[0] = False
    buffer = 1
    percent_changed = 0

    while percent_changed < percent_to_change:

        groups = available_instances[available_instances == True].groupby(
            (available_instances != True).cumsum()
        )

        groups_max = groups.size().max()  # get max group length
        instance_max = max(instance_max, groups_max)

        time_span = random.randint(1, instance_max)  # randomly select delta time length
        group_availability = groups.size() >= time_span
        available_increments = group_availability[group_availability == True]
        if len(available_increments) < 1:
            break

        group_index = available_increments.sample(1).index[0]

        group = groups.get_group(group_index)

        if time_span == len(group):
            random_start = group
        else:
            random_start = group[0:-time_span].sample(1)

        start_time_step = random_start.index[0]
        start_time_index = available_instances.index.get_loc(start_time_step)
        end_time_index = start_time_index + time_span

        new_weights[start_time_index:end_time_index] += change_function()

        available_instances[
            start_time_index - buffer : start_time_index + time_span + buffer
        ] = False

        percent_changed = (available_instances.value_counts(normalize=True) * 100)[
            False
        ]

    return new_weights.to_frame(name="weight")


def main():
    path = pathlib.Path.cwd() / "outputs"
    daily_data = pd.read_csv(path / "daily_servings.csv", parse_dates=True, index_col=0)
    upsampled_data = upsample(daily_data, "Standard")
    weight_data = create_weight_data(upsampled_data, "Standard")
    with_calibration_error = full_change(weight_data, 10, 5, calibration_error)
    with_signal_removal = full_change(with_calibration_error, 5, 5, signal_removal)
    consumer.write_output(weight_data, path, "full_transform")


if __name__ == "__main__":
    main()
