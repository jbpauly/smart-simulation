import importlib
import logging
import random
from collections import namedtuple

import daiquiri

daiquiri.setup(level=logging.INFO)
CUSTOMERS = importlib.import_module("cfg_templates.customers")


def decide(probability: float) -> bool:
    """
    Randomly make a binary decision given the probability of decision outcome

    Args:
        probability: That a customer will consume

    Returns:
        Customer's decision

    """
    if probability < 0 or probability > 1:
        logging.exception("Probability is an invalid number. Check configuration.")
        raise ValueError
    return random.random() < probability


def consume(random_function: random.random, function_params: tuple) -> int:
    """
    Consume a quantity of servings based on a number generated by a function of the Random library

    Args:
        random_function: Function for quantity number generation
        function_params: Input parameters for the function

    Returns:
        The quantity of servings consumes
    """
    try:
        quantity = random_function(*function_params)
    except TypeError:
        logging.exception(
            "function_params are invalid inputs for random_function. Check configurations."
        )
        raise
    except Exception:
        raise
    return quantity


def single_day(customer_config: namedtuple, day: int):
    """
    Generate consumption data for a customer on a single day

    Args:
        customer_config: Configuration namedtuple
        day: Day to generate a single day of data

    Returns:
        Quantity of product consumed
    """
    day_profile = customer_config[day]
    decision_probability = day_profile.probability
    consumption_generator = day_profile.consumption
    quantity = 0
    if decide(decision_probability):
        quantity = consume(
            consumption_generator.function, consumption_generator.parameters
        )
    return quantity


def multi_day(customer_number: str, days: int):
    """
    Generate consumption data for a customer for a given range of days

    Args:
        customer_number: ID number of customer, which should exist in the customers config file
        days: Number of days to generate data

    Returns:
        A list of product consumption in single quantity units
    """
    customer_config = get_customer(customer_number)
    consumption = []
    days_in_week = 7
    for day in range(0, days):
        day_of_week = day % days_in_week
        consumption.append(single_day(customer_config, day_of_week))
    return consumption


def get_customer(customer_number):
    """
        Get a customer from the customers configuration file specified by customer number
    Args:
        customer_number: Customer number in the configuration file

    Returns:
        Customer configuration namedtuple
    """
    if customer_number in CUSTOMERS.customers:
        return CUSTOMERS.customers[customer_number]
    else:
        raise Exception(
            "Customer: "
            + customer_number
            + ", does not exist in the configuration file."
        )


def main():

    customer_consumption = multi_day(customer_number="0", days=7)
    print(customer_consumption)


if __name__ == "__main__":
    main()
