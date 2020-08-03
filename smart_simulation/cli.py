"""Console script for smart_simulation."""
import importlib
import logging
import sys

import click

import daiquiri

consumer = importlib.import_module("consumer")


daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger(__name__)


def sim_customer(customer, days):
    return consumer.multi_day(customer, days)


@click.command()
@click.argument("customer")
@click.argument("days")
def main(customer, days):
    data = sim_customer(customer, int(days))
    print(data)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
