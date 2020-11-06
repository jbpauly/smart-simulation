import logging
import pathlib
from uuid import uuid4

import pandas as pd

import daiquiri
from smart_simulation import consumer, scale
from smart_simulation.cfg_templates import config, products
from smart_simulation.cfg_templates.customers import \
    customers as customer_templates

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)
package_path = pathlib.Path(config.package_dir)
outputs_path = package_path / "smart_simulation" / "outputs"
cust_templates = list(customer_templates.keys())


def write_output(data: pd.DataFrame, directory_path: pathlib.Path, file_name: str):
    """
    Write a Pandas DataFrame to a csv given a path and file name
    Args:
        directory_path: Path to write out the file
        file_name: Name of the file
        data: Pandas DataFrame
    """
    file_name = file_name + ".csv"
    if not isinstance(data, pd.DataFrame):
        logging.error("data must be a Pandas DataFrame.")
        raise TypeError
    data.to_csv(directory_path / file_name)


def batch_simulation(
    number_instances: int,
    customer_template: str,
    upsample_template: str,
    delivery_days_template: str,
    delivery_skew_template: str,
    weight_template: str,
    quantity_weight_template: str,
    start_date: str,
    num_days: str,
    output_path: pathlib.Path = outputs_path,
):
    """
    Create a batch of consumer servings and scale weight data generation and save to an simulation output directory.
    Args:
        number_instances: Number of instances to simulate with the configuration selections.
        customer_template: Configuration choice for customer template.
        upsample_template: Configuration choice for upsample template.
        delivery_days_template: Configuration choice for delivery days template.
        delivery_skew_template: Configuration choice for delivery skew template.
        weight_template: Configuration choice for weight template.
        quantity_weight_template: Configuration choice for quantity to weight template.
        start_date: Start date of the data generation in YYYY-MM-DD format.
        num_days: Number of days to generate data.
        output_path: Output path for the simulations directory and generated data csv files.
    """
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    simulations_directory = output_path / "simulations"
    servings_directory = outputs_path / simulations_directory / "servings"
    weights_directory = outputs_path / simulations_directory / "weights"
    directories = [simulations_directory, servings_directory, weights_directory]
    simulations_log_path = simulations_directory / "simulations_log.csv"
    full_simulations_log_path = outputs_path / simulations_log_path
    for directory in directories:
        directory.mkdir(parents=False, exist_ok=True)

    sim_config = {
        "customer_template": customer_template,
        "upsample_template": upsample_template,
        "delivery_days_template": delivery_days_template,
        "delivery_skew_template": delivery_skew_template,
        "weight_template": weight_template,
        "quantity_weight_template": quantity_weight_template,
    }
    instances_id = []
    for instance in range(number_instances):
        uid = str(uuid4())
        instances_id.append(uid)
        daily_servings = consumer.multi_day(
            customer_number=customer_template, days=num_days, start_date=start_date
        )
        upsampled_servings = scale.upsample(daily_servings, upsample_template)
        weights = scale.create_weight_data(
            upsampled_servings,
            delivery_days_template,
            delivery_skew_template,
            weight_template,
            quantity_weight_template,
        )
        daily_servings_file_name = uid + "_daily"
        upsampled_servings_file_name = uid + "_upsampled"

        weights_file_name = (
            uid + "_weights_" + delivery_days_template + "_" + delivery_skew_template
        )

        write_output(daily_servings, servings_directory, daily_servings_file_name)
        write_output(
            upsampled_servings, servings_directory, upsampled_servings_file_name
        )
        write_output(weights, weights_directory, weights_file_name)

    instances_configs = {instance: sim_config for instance in instances_id}
    configs_df = pd.DataFrame(instances_configs).T

    if full_simulations_log_path.is_file():
        configs_df.to_csv(full_simulations_log_path, header=False, mode="a")
    else:
        configs_df.to_csv(full_simulations_log_path, mode="w")


def main():
    batch_simulation(
        number_instances=3,
        customer_template="0",
        upsample_template="Standard",
        delivery_days_template="Standard",
        delivery_skew_template="skew_on_time",
        weight_template="Standard",
        quantity_weight_template="Standard",
        start_date="2020-01-01",
        num_days=365,
    )


if __name__ == "__main__":
    main()
