import random
from collections import namedtuple

# CUSTOMER is used to save customer templates of a given consumption profile.
# Consumption profiles are broken down into daily consumption habits: probability to consume and serving consumed.
CUSTOMER = namedtuple(
    "Customer", "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday"
)

# DAY_PROFILE is used to save templates of daily consumer habits: probability to consume and serving consumed.
DAY_PROFILE = namedtuple("Day_Profile", "probability, consumption")

# CONSUMPTION is used to save templates of consumption habits calculated by a random number generator.
CONSUMPTION = namedtuple("Consumption", "function, parameters")

# templates used to select the servings consumed on a specific day within a given range of acceptable values.
consumption_types = {
    "low": CONSUMPTION(random.randint, [1, 2]),
    "average": CONSUMPTION(random.randint, [2, 5]),
    "high": CONSUMPTION(random.randint, [5, 8]),
}

# templates used to represent categories of probability
# probabilities must be in range [0,1]
probabilities = {"low": 0.1, "average": 0.5, "high": 0.9}

customers = {
    "0": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Tuesday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Wednesday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Thursday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Saturday=DAY_PROFILE(probabilities["low"], consumption_types["high"]),
        Sunday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
    ),
    "1": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Tuesday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Wednesday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Thursday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Saturday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Sunday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
    ),
    "2": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Tuesday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Wednesday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Thursday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["average"], consumption_types["low"]),
        Saturday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Sunday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
    ),
    "3": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Tuesday=DAY_PROFILE(probabilities["low"], consumption_types["average"]),
        Wednesday=DAY_PROFILE(probabilities["average"], consumption_types["low"]),
        Thursday=DAY_PROFILE(probabilities["average"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["high"], consumption_types["low"]),
        Saturday=DAY_PROFILE(probabilities["low"], consumption_types["average"]),
        Sunday=DAY_PROFILE(probabilities["low"], consumption_types["high"]),
    ),
    "4": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Tuesday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Wednesday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Thursday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Friday=DAY_PROFILE(probabilities["high"], consumption_types["average"]),
        Saturday=DAY_PROFILE(probabilities["average"], consumption_types["low"]),
        Sunday=DAY_PROFILE(probabilities["average"], consumption_types["low"]),
    ),
    "5": CUSTOMER(
        Monday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Tuesday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Wednesday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Thursday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Friday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Saturday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
        Sunday=DAY_PROFILE(probabilities["low"], consumption_types["low"]),
    ),
}
