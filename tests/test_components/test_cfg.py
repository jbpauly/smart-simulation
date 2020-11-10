import random
from collections import namedtuple

CUSTOMER = namedtuple(
    "Customer", "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday"
)
DAY_PROFILE = namedtuple("Day_Profile", "probability, consumption")
CONSUMPTION = namedtuple("Consumption", "function, parameters")

consumption_types = {
    "low": CONSUMPTION(random.randint, [1, 2]),
    "high": CONSUMPTION(random.randint, [5, 8]),
}
probabilities = {"low": 0.1, "high": 0.9}

customers = {
    "test": CUSTOMER(
        Monday=DAY_PROFILE("high", "low"),
        Tuesday=DAY_PROFILE("high", "low"),
        Wednesday=DAY_PROFILE("high", "low"),
        Thursday=DAY_PROFILE("high", "low"),
        Friday=DAY_PROFILE("high", "low"),
        Saturday=DAY_PROFILE("low", "high"),
        Sunday=DAY_PROFILE("low", "low"),
    ),
}
