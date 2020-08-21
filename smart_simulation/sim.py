import logging
import random

import daiquiri
from smart_simulation.cfg_templates import customers

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


class Customer:
    def __init__(self, name, consumer_profile):
        self.name = name
        self.consumer_profile = consumer_profile

    def get_name(self):
        return self.name

    def get_consumer_profile(self):
        return self.consumer_profile

    def get_day_profile(self, day):
        return self.consumer_profile[day]

    def get_day_probability(self, day):
        return self.consumer_profile[day].probability

    def get_day_consumption(self, day):
        return self.consumer_profile[day].consumption

    def decide(self, day):
        probability = self.get_day_probability(day)
        chance = random.random()
        return chance < probability

    def consume(self, day):
        consumption_profile = self.get_day_consumption(day)
        function = consumption_profile.function
        parameters = consumption_profile.parameters

        try:
            quantity = function(*parameters)
        except TypeError:
            logging.exception(
                "function_params are invalid inputs for random_function. Check configurations."
            )
            raise
        except Exception:
            raise
        return quantity

    def single_day(self, day):
        if self.consume(day):
            return self.consume(day)


class Scale:
    def __init__(self, weight):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight


def main():
    joe = Customer("joe", customers.customers["0"])

    print(joe.decide(6))


if __name__ == "__main__":
    main()
