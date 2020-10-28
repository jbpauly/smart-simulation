"""Console script for smart_simulation."""
import logging
import pathlib
import random

import click

import daiquiri
from smart_simulation import consumer, simulate
from smart_simulation.cfg_templates import config, products
from smart_simulation.cfg_templates.customers import \
    customers as customer_templates

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)
package_path = pathlib.Path(config.package_dir)
outputs_path = package_path / "smart_simulation" / "outputs"
customer_template_keys = list(customer_templates.keys())
upsample_template_keys = list(products.consumption_windows.keys())
delivery_days_template_keys = list(products.delivery_max_days.keys())
delivery_skew_template_keys = list(products.delivery_skew_probabilities.keys())
weight_template_keys = list(products.stock_weight.keys())
quantity_weight_templates = list(products.quantity_to_weight.keys())


@click.group()
def main():
    pass


@main.command()
@click.option(
    "-c", "--customer_number", prompt=True, type=click.Choice(customer_template_keys)
)
@click.option("-d", "--days", prompt=True, type=int)
@click.option("-s", "--start_date", default="2020-01-01", prompt=True, type=str)
@click.option(
    "-f", "--file_name", prompt="File name (without file type extension)", type=str
)
@click.option(
    "-p", "--output_path", default=outputs_path, prompt=True, type=click.Path()
)
def create_consumption(customer_number, days, start_date, file_name, output_path):
    """Create consumer consumption data (servings) and associated scale data (weight)"""
    customer_behavior = consumer.multi_day(
        customer_number=customer_number, days=days, start_date=start_date
    )
    customer_consumption = consumer.perfect_scale(
        data=customer_behavior,
        quantity_to_weight=(random.normalvariate, (0.38, 0.05)),
        stock_weight=13,
    )
    consumer.write_output(customer_consumption, output_path, file_name)


@main.command()
@click.option("-n", "--number_instances", prompt=True, type=int)
@click.option(
    "-c", "--customer_template", prompt=True, type=click.Choice(customer_template_keys)
)
@click.option(
    "-u", "--upsample_template", prompt=True, type=click.Choice(upsample_template_keys)
)
@click.option(
    "-dd",
    "--delivery_days_template",
    prompt=True,
    type=click.Choice(delivery_days_template_keys),
)
@click.option(
    "-ds",
    "--delivery_skew_template",
    prompt=True,
    type=click.Choice(delivery_skew_template_keys),
)
@click.option(
    "-w", "--weight_template", prompt=True, type=click.Choice(weight_template_keys)
)
@click.option(
    "-qw",
    "--quantity_weight_template",
    prompt=True,
    type=click.Choice(quantity_weight_templates),
)
@click.option("-s", "--start_date", default="2020-01-01", prompt=True, type=str)
@click.option("-d", "--num_days", prompt=True, type=int)
@click.option(
    "-p", "--output_path", default=outputs_path, prompt=True, type=click.Path()
)
def batch_simulation(
    number_instances,
    customer_template,
    upsample_template,
    delivery_days_template,
    delivery_skew_template,
    weight_template,
    quantity_weight_template,
    start_date,
    num_days,
    output_path,
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
    simulate.batch_simulation(
        number_instances,
        customer_template,
        upsample_template,
        delivery_days_template,
        delivery_skew_template,
        weight_template,
        quantity_weight_template,
        start_date,
        num_days,
        output_path,
    )


if __name__ == "__main__":
    main()
