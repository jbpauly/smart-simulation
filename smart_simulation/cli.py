"""Console script for smart_simulation."""
import logging
import pathlib
import random

import click

import daiquiri
from smart_simulation import consumer
from smart_simulation.cfg_templates import config
from smart_simulation.cfg_templates.customers import \
    customers as customer_templates

daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)
package_path = pathlib.Path(config.package_dir)
outputs_path = package_path / "smart_simulation" / "outputs"
cust_templates = list(customer_templates.keys())


@click.group()
def main():
    pass


@main.command()
@click.option("-c", "--customer_number", prompt=True, type=click.Choice(cust_templates))
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


if __name__ == "__main__":
    main()
