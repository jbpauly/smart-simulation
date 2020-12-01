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
    "Michael": CUSTOMER(
        Monday=DAY_PROFILE("high", "low"),
        Tuesday=DAY_PROFILE("high", "low"),
        Wednesday=DAY_PROFILE("high", "low"),
        Thursday=DAY_PROFILE("high", "low"),
        Friday=DAY_PROFILE("high", "low"),
        Saturday=DAY_PROFILE("low", "high"),
        Sunday=DAY_PROFILE("low", "low"),
    ),
    "Joe": CUSTOMER(
        Monday=DAY_PROFILE("high", "average"),
        Tuesday=DAY_PROFILE("low", "average"),
        Wednesday=DAY_PROFILE("average", "low"),
        Thursday=DAY_PROFILE("average", "low"),
        Friday=DAY_PROFILE("high", "low"),
        Saturday=DAY_PROFILE("low", "average"),
        Sunday=DAY_PROFILE("low", "high"),
    ),
    "Liana": CUSTOMER(
        Monday=DAY_PROFILE("high", "average"),
        Tuesday=DAY_PROFILE("high", "average"),
        Wednesday=DAY_PROFILE("high", "average"),
        Thursday=DAY_PROFILE("high", "average"),
        Friday=DAY_PROFILE("high", "average"),
        Saturday=DAY_PROFILE("average", "low"),
        Sunday=DAY_PROFILE("average", "low"),
    ),
}


# customers = {
#     "0": CUSTOMER(
#         Monday=DAY_PROFILE("high", "low"),
#         Tuesday=DAY_PROFILE("high", "low"),
#         Wednesday=DAY_PROFILE("high", "low"),
#         Thursday=DAY_PROFILE("high", "low"),
#         Friday=DAY_PROFILE("high", "low"),
#         Saturday=DAY_PROFILE("low", "high"),
#         Sunday=DAY_PROFILE("low", "low"),
#     ),
#     "1": CUSTOMER(
#         Monday=DAY_PROFILE("high", "average"),
#         Tuesday=DAY_PROFILE("low", "low"),
#         Wednesday=DAY_PROFILE("high", "average"),
#         Thursday=DAY_PROFILE("low", "low"),
#         Friday=DAY_PROFILE("high", "average"),
#         Saturday=DAY_PROFILE("low", "low"),
#         Sunday=DAY_PROFILE("high", "average"),
#     ),
#     "2": CUSTOMER(
#         Monday=DAY_PROFILE("low", "low"),
#         Tuesday=DAY_PROFILE("low", "low"),
#         Wednesday=DAY_PROFILE("low", "low"),
#         Thursday=DAY_PROFILE("low", "low"),
#         Friday=DAY_PROFILE("average", "low"),
#         Saturday=DAY_PROFILE("high", "average"),
#         Sunday=DAY_PROFILE("high", "average"),
#     ),
#     "3": CUSTOMER(
#         Monday=DAY_PROFILE("high", "average"),
#         Tuesday=DAY_PROFILE("low", "average"),
#         Wednesday=DAY_PROFILE("average", "low"),
#         Thursday=DAY_PROFILE("average", "low"),
#         Friday=DAY_PROFILE("high", "low"),
#         Saturday=DAY_PROFILE("low", "average"),
#         Sunday=DAY_PROFILE("low", "high"),
#     ),
#     "4": CUSTOMER(
#         Monday=DAY_PROFILE("high", "average"),
#         Tuesday=DAY_PROFILE("high", "average"),
#         Wednesday=DAY_PROFILE("high", "average"),
#         Thursday=DAY_PROFILE("high", "average"),
#         Friday=DAY_PROFILE("high", "average"),
#         Saturday=DAY_PROFILE("average", "low"),
#         Sunday=DAY_PROFILE("average", "low"),
#     ),
#     "5": CUSTOMER(
#         Monday=DAY_PROFILE("low", "low"),
#         Tuesday=DAY_PROFILE("low", "low"),
#         Wednesday=DAY_PROFILE("low", "low"),
#         Thursday=DAY_PROFILE("low", "low"),
#         Friday=DAY_PROFILE("low", "low"),
#         Saturday=DAY_PROFILE("low", "low"),
#         Sunday=DAY_PROFILE("low", "low"),
#     ),
# }
