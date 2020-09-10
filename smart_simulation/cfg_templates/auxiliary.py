import random

arrival_options = {
    "Standard": {
        "early": (random.randint, (-16, -3)),
        "perfect": (random.randint, (-2, 2)),
        "late": (random.randint, (3, 16)),
    },
    "Ideal": {"perfect": (random.randint, (-2, 2)),},
}

arrival_options_weights = {
    "Standard": {"early": 50, "perfect": 30, "late": 20,},
    "Ideal": {"perfect": 100},
}

stock_weight = 14.0
quantity_to_weight = (random.normalvariate, (0.38, 0.05))


consumption_windows = {
    "Standard": {"00:00:00": 20, "06:00:00": 50, "12:00:00": 20, "18:00:00": 10}
}


scale_calibration_error_range = {"Standard": (-15, 15)}
