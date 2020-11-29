import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.naive import NaiveForecaster


def create_prediction_dates(
    dates: pd.DatetimeIndex, min_train: int = 50, max_forecast: int = 14
) -> pd.DatetimeIndex:
    """
    Create a range of dates for predictions.
    Args:
        dates: All dates in a weight dataset.
        min_train: Minimum days in a training dataset
        max_forecast: Maximum number of days in a forecast.

    Returns: Available dates for predictions.

    """
    unique_dates = list(set(dates.date))
    unique_dates.sort()
    min_date = unique_dates[min_train]
    max_date = unique_dates[-max_forecast]
    available_dates = pd.date_range(min_date, max_date, freq="1D")
    return available_dates


def sma_forecast(y_train: pd.Series, forecast_horizon: np.array) -> pd.Series:
    """

    Args:
        y_train:

    Returns:

    """
    forecaster = NaiveForecaster(strategy="mean", window_length=7)
    forecaster.fit(y_train)
    forecast = forecaster.predict(forecast_horizon)
    return forecast


def sarima_forecast(y_train: pd.Series, forecast_size: int) -> pd.Series:
    """

    Args:
        y_train:

    Returns:

    """
    model = sm.tsa.SARIMAX(endog=y_train, order=(0, 0, 0), seasonal_order=(1, 0, 1, 7))
    res = model.fit()
    # start = y_train.index[-1] + pd.Timedelta("1D")
    # end = start + pd.Timedelta(str(forecast_size) + "D")
    # forecast = fit.predict(start=start, end=end)
    forecast = res.forecast(steps=forecast_size)

    return forecast


def rf_forecast(y_train: pd.Series, forecast_horizon: np.array) -> pd.Series:
    """

    Args:
        y_train:

    Returns:

    """
    regressor = RandomForestRegressor(n_estimators=100)
    forecaster = ReducedRegressionForecaster(
        regressor=regressor, window_length=10, strategy="recursive"
    )
    forecaster.fit(y_train)
    forecast = forecaster.predict(forecast_horizon)
    return forecast


def create_forecast_horizon(forecast_size: int) -> np.array:
    """

    Args:
        forecast_size:

    Returns:

    """
    fh = np.arange(1, forecast_size + 1)
    return fh


def forecast_consumption(
    forecast_size: int, y_train: pd.Series,
) -> (pd.Series, pd.Series, pd.Series):
    forecast_horizon = create_forecast_horizon(forecast_size=forecast_size)
    sma_predictions = sma_forecast(y_train=y_train, forecast_horizon=forecast_horizon)
    sarima_predictions = sarima_forecast(y_train=y_train, forecast_size=forecast_size)
    rf_predictions = rf_forecast(y_train=y_train, forecast_horizon=forecast_horizon)
    return sma_predictions, sarima_predictions, rf_predictions
