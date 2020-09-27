import random

# deliver_max_days templates must be in the form: early, on_time, late
# OR match the template structure of delivery_skew_probabilities.
# The integer value of each 'delivery category' represents the maximum number of days away
# from a perfect, just-in-time delivery
delivery_max_days = {"Standard": {"early": 5, "on_time": 1, "late": 4,}}

# delivery_skew_probabilities templates must be in the form: early, on_time, late
# OR match the template structure of deliver_max_days.
# The integer value of each 'delivery category' represents the probability of a delivery being of that category.
# The sum of all probabilities must == 100
delivery_skew_probabilities = {
    "skew_early": {"early": 50, "on_time": 30, "late": 20,},
    "skew_on_time": {"early": 10, "on_time": 80, "late": 10,},
    "skew_late": {"early": 20, "on_time": 30, "late": 50,},
    "perfect": {"early": 0, "on_time": 100, "late": 0,},
}

# the stock weights of products to be delivered
# weight unit is not assumed, but coffee is typically measured in ounces
stock_weight = {"Standard": 14.0}

# templates consist of a tuple of (function, function arguments) to be used to calculate the weight of a 'serving'
quantity_to_weight = {"Standard": (random.normalvariate, (0.38, 0.05))}

# consumption_windows templates must be in the for 'hh:mm:ss' and in equal increments
consumption_windows = {
    "Standard": {"00:00:00": 20, "06:00:00": 50, "12:00:00": 20, "18:00:00": 10}
}

# scale_calibration_error_range templates must be a tuple of the (min, max) error range
# assume the same unit as the stock weight
scale_calibration_error_range = {"Standard": (-15, 15)}
