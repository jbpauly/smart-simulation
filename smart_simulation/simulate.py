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
        logging.exception("data must be a Pandas DataFrame.")
        raise TypeError
    data.to_csv(directory_path / file_name)


def batch_simulation(
    number_instances,
    customer_template,
    upsample_template,
    resupply_template,
    start_date,
    num_days,
    output_path,
):

    simulations_directory = outputs_path / "simulations"
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
        "resupply_template": resupply_template,
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
            upsampled_servings, "Standard", "skew_early", "Standard", "Standard"
        )
        daily_servings_file_name = uid + "_daily"
        upsampled_servings_file_name = uid + "_upsampled"
        weights_file_name = uid + "_weights_" + resupply_template

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
    batch_simulation(3, "0", "Standard", "Standard", "2020-01-01", 365, "null")


if __name__ == "__main__":
    main()
