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
        Monday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Tuesday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Wednesday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Thursday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Saturday=DAY_PROFILE(probabilities["low"], consumption_types["high"]),
        Sunday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
    )
}
