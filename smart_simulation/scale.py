import datetime
import logging
import pathlib
import random
from typing import Union

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


def calculate_arrival_windows(delivery_days_max: dict, freq: str) -> dict:
    """
    Calculate the arrival windows in number of time stamps for early, on-time, and late deliveries
    Args:
        delivery_days_max: Maximum days in early, on-time, late delivery window
        freq: Frequency of the dataset utilized to calculate number of time stamps per day

    Returns:
        A dictionary of tuples as the range of time stamps for early, on-time, and late delivery windows
    """
    time_stamps_per_day = pd.Timedelta("1 day") / freq

    max_early = delivery_days_max["early"]
    early_low = max_early * time_stamps_per_day * -1
    early_high = time_stamps_per_day * -1
    early_range = (early_low, early_high)

    max_on_time = int(delivery_days_max["on_time"] * time_stamps_per_day / 2)
    on_time_low = max_on_time * -1
    on_time_high = max_on_time
    on_time_range = (on_time_low, on_time_high)

    max_late = delivery_days_max["late"]
    late_low = time_stamps_per_day
    late_high = max_late * time_stamps_per_day
    late_range = (late_low, late_high)

    arrival_windows = {
        "early": early_range,
        "on_time": on_time_range,
        "late": late_range,
    }
    return arrival_windows


def deliver_product(delivery_skew_probabilities: dict, arrival_windows: dict) -> int:
    """
    Calculate and return the delta, in number of timestamps, of delivery from a true 'scale weight == zero' timestamp
    Args:
        delivery_skew_probabilities: A dictionary of of probabilities for early, on-time, and late delivery
        arrival_windows: A dictionary of acceptable range in timestamps for early, on-time, and late delivery

    Returns: A int representing the delta of timestamps away from true 'scale weight == zero' that a delivery arrives

    """
    arrival_time_categories = [*delivery_skew_probabilities]
    arrival_time_probabilities = tuple(delivery_skew_probabilities.values())
    # random.choices() returns a list of 'k' number selections, just need string value
    arrival_time_category = random.choices(
        population=arrival_time_categories, weights=arrival_time_probabilities, k=1,
    )[0]
    delivery_time_stamp_range = arrival_windows[arrival_time_category]
    delivery_time_delta = random.randint(*delivery_time_stamp_range)

    return delivery_time_delta


def calculate_current_weight(
    consumption_servings: int, previous_step_weight: float, quantity_to_weight: tuple
) -> float:
    """
    Calculate and return the current weight at a timestamp given serving consumed and previous timestamp weight
    Args:
        consumption_servings: Number of servings consumed in current time window
        previous_step_weight: Weight at previous timestamp
        quantity_to_weight: Tuple of a random function and function arguments to calculate the weight per serving

    Returns: A float of the current weight after consumption

    """
    current_weight = previous_step_weight

    if consumption_servings > 0:
        random_function = quantity_to_weight[0]
        function_args = quantity_to_weight[1]
        serving_weight = random_function(*function_args)
        consumption_weight = serving_weight * consumption_servings

        if consumption_weight > previous_step_weight:
            current_weight = 0
        else:
            current_weight = previous_step_weight - consumption_weight

    return current_weight


def create_weight_data(
    dataset: pd.DataFrame,
    delivery_days_template: str,
    delivery_skew_template: str,
    weight_template: str,
    quantity_weight_template: str,
) -> pd.DataFrame:
    """
    Create weight data based on consumption and shipment arrivals
    Args:
        delivery_skew_template: Probabilities of a delivery skewing arriving early, on-time, or late
        delivery_days_template: Maximum number of days in early, on-time, and late ranges
        quantity_weight_template: Conversion from a serving to weight in ounces
        weight_template: Stock weight for item resupply
        dataset: Dataset of time stamps and servings (consumed) by consumer

    Returns:
        Pandas DataFrame of weight measurements

    """
    data = dataset.copy()
    delivery_max_days = products.delivery_max_days[delivery_days_template]
    delivery_skew_probabilities = products.delivery_skew_probabilities[
        delivery_skew_template
    ]
    stock_weight = products.stock_weight[weight_template]
    quantity_to_weight = products.quantity_to_weight[quantity_weight_template]
    frequency = data.index.freq
    arrival_windows = calculate_arrival_windows(delivery_max_days, frequency)

    servings = data.servings
    num_time_stamps = len(data.index)
    weights = pd.Series(data=num_time_stamps * [0], index=data.index, dtype=float)
    weights.iat[0] = stock_weight

    time_step = 1
    previous_delivery_index = 0

    while time_step < weights.size:
        previous_step_weight = weights.iat[time_step - 1]
        consumption_servings = servings.iat[time_step]
        if previous_step_weight == 0.0:
            # get time step index of current delivery
            delivery_time_delta = deliver_product(
                delivery_skew_probabilities, arrival_windows
            )
            delivery_index = time_step + delivery_time_delta

            if delivery_index > weights.size - 1:
                break  # last delivery after end of time range in dataset
            if (
                delivery_index < previous_delivery_index
            ):  # delivery must be within dataset time range
                continue  # try again

            # add weight of delivery to scale on delivery time step
            time_step = delivery_index
            previous_delivery_index = delivery_index
            weights.iat[time_step] += stock_weight

            # reset to 1 time step after delivery
            time_step += 1
            previous_step_weight = weights.iat[time_step - 1]
            consumption_servings = servings.iat[time_step]

        # Calculate current weight after consumption
        current_weight = calculate_current_weight(
            consumption_servings, previous_step_weight, quantity_to_weight
        )
        weights.iat[time_step] = current_weight

        time_step += 1

    return weights.to_frame(name="weight")


def calibration_error(calibration_error_template: str) -> float:
    """
    Generate random number to represent delta of calibration error of weight scale
    Args:
        calibration_error_template: Acceptable range of weight for calibration error
    Returns:
        Random value within range of preset calibration errors
    """
    weight_range = products.scale_calibration_error_range[calibration_error_template]
    weight_delta = random.uniform(*weight_range)
    return weight_delta


def signal_removal() -> np.nan:
    """
    Return a value to mimic removal of weight signal scenarios like network disconnection, dead battery of scale, etc.

    Returns:
        Numpy nan value
    """
    return np.nan


def change_start(
    available_instances: pd.Series, instance_max: int
) -> Union[datetime.datetime, int]:
    """
    Create a starting datetime and timespan of a weight transformation
    Args:
        available_instances: A series boolean values for datetime availablity for change
        instance_max: Maximum number of time steps that can be changed in one timespan

    Returns:
        random_start - the datetime of the change timespan start
        time_span - number of time steps to change

    """
    groups = available_instances[available_instances == True].groupby(
        (available_instances != True).cumsum()
    )

    groups_max = groups.size().max()  # get max group length
    instance_max = min(instance_max, groups_max)

    time_span = random.randint(1, instance_max)  # randomly select delta time length
    group_availability = groups.size() >= time_span
    available_increments = group_availability[group_availability == True]

    random_start = None
    if len(available_increments) > 0:
        group_index = available_increments.sample(1).index[0]
        group = groups.get_group(group_index)
        group_time_stamps = group.index.tolist()

        if time_span == len(group_time_stamps):
            random_start = group_time_stamps[0]
        else:
            random_start = random.choice(group_time_stamps[0:-time_span])

    return random_start, time_span


def full_change(
    dataset: pd.DataFrame,
    percent_to_change: int,
    days_max: int,
    change_function,
    function_argument=None,
) -> pd.DataFrame:
    """
    Change a dataset of weight data up to a percent of instances
    Args:
        function_argument: Optional argument for the change change_function
        dataset: Dataset of weight measurements to change
        percent_to_change: Percent of dataset to change
        days_max: Maximum number of consecutive days allowable for signal calibration or removal changes
        change_function: Function to change data (calibration or signal removal)

    Returns:
        New Pandas Dataframe of weight data after changes
    """
    changed_data = dataset.copy()
    new_weights = pd.Series(data=changed_data["weight"], index=changed_data.index)
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

        random_start, time_span = change_start(available_instances, instance_max)

        # break while loop if there are no more available time windows to change
        if random_start is None:
            break

        start_time_index = available_instances.index.get_loc(random_start)
        end_time_index = start_time_index + time_span

        if function_argument is not None:
            weight_delta = change_function(function_argument)
        else:
            weight_delta = change_function()
        new_weights[start_time_index:end_time_index] += weight_delta

        available_instances[
            start_time_index - buffer : start_time_index + time_span + buffer
        ] = False
        percent_changed = (available_instances.value_counts(normalize=True) * 100)[
            False
        ]

    weights = new_weights.to_frame(name="weight")
    return weights


def main():
    path = pathlib.Path.cwd() / "outputs"
    daily_data = pd.read_csv(path / "daily_servings.csv", parse_dates=True, index_col=0)
    upsampled_data = upsample(daily_data, "Standard")
    weight_data = create_weight_data(
        upsampled_data, "Standard", "skew_late", "Standard", "Standard"
    )
    with_calibration_error = full_change(
        weight_data, 30, 5, calibration_error, "Standard"
    )
    with_signal_removal = full_change(with_calibration_error, 5, 5, signal_removal,)
    consumer.write_output(with_signal_removal, path, "full_transform")


if __name__ == "__main__":
    main()
