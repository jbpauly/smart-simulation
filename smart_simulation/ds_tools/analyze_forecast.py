import logging

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import daiquiri

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


def plot_forecast_results(
    pred_date: pd.Timestamp,
    train_consumption: pd.Series,
    test_consumption: pd.Series,
    pred_consumption: pd.Series,
    train_weight: pd.Series,
    test_weight: pd.Series,
    pred_weight: pd.Series,
    test_binary: pd.Series,
    pred_binary: pd.Series,
    model_name: str,
    forecast_days: int,
):

    date = pred_date.strftime("%B %d, %Y")

    fig, ax = plt.subplots(3, 1, figsize=(18, 12))

    ax[0].plot(
        train_consumption.index, train_consumption.values, label="Train", color="blue"
    )
    ax[0].plot(
        test_consumption.index,
        test_consumption.values,
        label="Test: True",
        color="green",
    )
    ax[0].plot(
        pred_consumption.index,
        pred_consumption.values,
        label=f"Test: {model_name} Forecast",
        color="red",
    )
    ax[0].set(
        ylabel="Consumption (oz)",
        title=f"{forecast_days} Day Consumption Forecast of [Sample] Customer on {date}",
    )
    ax[0].legend()

    ax[1].plot(train_weight.index, train_weight.values, label="Train", color="blue")
    ax[1].plot(test_weight.index, test_weight.values, label="Test: True", color="green")
    ax[1].plot(
        pred_weight.index,
        pred_weight.values,
        label=f"Test: {model_name} Forecast",
        color="red",
    )
    ax[1].set(
        ylabel="Scale Weight (oz)",
        title=f"{forecast_days} Day Scale Weight Forecast of [Sample] Customer on {date}",
    )

    ax[2].plot(test_binary.index, test_binary.values, label="Test: True", color="green")
    ax[2].plot(
        pred_binary.index,
        pred_binary.values,
        label=f"Test: {model_name} Forecast",
        color="red",
    )
    ax[2].set(
        xlabel="Date",
        ylabel="Scale Weight Above (0) oz",
        title=f"{forecast_days} Day Positive Scale Weight Forecast of [Sample] Customer on {date}",
    )
    ax[2].set_yticks([0, 1])

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax[0].xaxis.set_major_formatter(formatter)
    ax[1].xaxis.set_major_formatter(formatter)
    ax[2].xaxis.set_major_formatter(formatter)
