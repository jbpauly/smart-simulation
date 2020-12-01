import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import streamlit as st
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.naive import NaiveForecaster
from smart_simulation.ds_tools import data_eng as de


@st.cache(suppress_st_warning=True)
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


@st.cache(suppress_st_warning=True)
def sma_forecast(y_train: pd.Series, forecast_horizon: np.array) -> pd.Series:
    """
    Fit a simple moving average model with training data and forecast for a given horizon.
    Args:
        y_train: Historic dataset to fit model.
        forecast_horizon: Array of forecast periods [1, ... , n] n being number of desired periods to forecast.

    Returns: A pandas series of consumption forecast with a datetimeindex.

    """
    forecaster = NaiveForecaster(strategy="mean", window_length=7)
    forecaster.fit(y_train)
    forecast = forecaster.predict(forecast_horizon).rename("consumption")
    return forecast


@st.cache(suppress_st_warning=True)
def sarima_forecast(y_train: pd.Series, forecast_size: int) -> pd.Series:
    """
    Fit a SARIMA model with training data and forecast for a given horizon.
    Args:
        y_train: Historic dataset to fit model.
        forecast_size: Number of timesteps to forecast consumption.

    Returns: A pandas series of consumption forecast with a datetimeindex.

    """
    model = sm.tsa.SARIMAX(endog=y_train, order=(0, 0, 0), seasonal_order=(1, 0, 1, 7))
    res = model.fit()
    end = y_train.index[-1] + pd.Timedelta(str(forecast_size) + "D")
    forecast = res.forecast(steps=end).rename("consumption")
    return forecast


@st.cache(suppress_st_warning=True)
def rf_forecast(y_train: pd.Series, forecast_horizon: np.array) -> pd.Series:
    """
    Fit a random forest model with training data and forecast for a given horizon.
    Args:
        y_train: Historic dataset to fit model.
        forecast_horizon: Array of forecast periods [1, ... , n] n being number of desired periods to forecast.

    Returns: A pandas series of consumption forecast with a datetimeindex.

    """
    regressor = RandomForestRegressor(n_estimators=100)
    forecaster = ReducedRegressionForecaster(
        regressor=regressor, window_length=15, strategy="recursive"
    )
    forecaster.fit(y_train)
    forecast = forecaster.predict(forecast_horizon).rename("consumption")
    return forecast


@st.cache(suppress_st_warning=True)
def create_forecast_horizon(forecast_size: int) -> np.array:
    """
    Create a forecast horizon for use in sktime forecasting.
    Horizon should be an array of integers representing the timesteps since end of training data to forecast.
    For Example [1,2,3] represents a 3 timestep horizon with a prediction at each timestep.
    Args:
        forecast_size: Size of the desired forecast.

    Returns: An array [1 ... forecast_size]
    """
    fh = np.arange(1, forecast_size + 1)
    return fh


@st.cache(suppress_st_warning=True)
def forecast_consumption(
    forecast_size: int, y_train: pd.Series,
) -> (pd.Series, pd.Series, pd.Series):
    """
    Forecast consumption for a specified forecast size given a training dataset using 3 different models:
        1. Simple Moving Average
        2. SARIMA
        3. Random Forest
    Args:
        forecast_size: Size of the forecast (days)
        y_train: Training dataset to fit the models.

    Returns: A pandas Series of consumption forecasts for each model, with a datetimeindex.

    """
    forecast_horizon = create_forecast_horizon(forecast_size=forecast_size)
    sma_predictions = sma_forecast(y_train=y_train, forecast_horizon=forecast_horizon)
    sarima_predictions = sarima_forecast(y_train=y_train, forecast_size=forecast_size)
    rf_predictions = rf_forecast(y_train=y_train, forecast_horizon=forecast_horizon)
    return sma_predictions, sarima_predictions, rf_predictions


@st.cache(suppress_st_warning=True)
def calculate_rmse(
    y_true: pd.Series, y_pred: pd.Series, squared: bool = False
) -> float:
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=squared)
    return rmse


@st.cache(suppress_st_warning=True)
def rmse_table(
    y_true: pd.Series, sma_pred: pd.Series, sarima_pred: pd.Series, rf_pred: pd.Series
) -> pd.DataFrame:
    """
    Create a table of RMSE values given a true value and predicted value(s) from SMA, ARIMA, RF forecasts.
    Args:
        y_true: True value(s) of prediction.
        sma_pred: Prediction values of SMA forecast.
        sarima_pred: Prediction values of ARIMA forecast.
        rf_pred: Prediction values of Random Forest forecast.

    Returns: RMSE values as a pandas DataFrame

    """
    sma_rmse = calculate_rmse(y_true=y_true, y_pred=sma_pred)
    sarima_rmse = calculate_rmse(y_true=y_true, y_pred=sarima_pred)
    rf_rmse = calculate_rmse(y_true=y_true, y_pred=rf_pred)
    all_rmse = pd.DataFrame(
        [sma_rmse, sarima_rmse, rf_rmse],
        columns=["RMSE (oz.)"],
        index=["Moving Average", "SARIMA", "Random Forest"],
    )
    return all_rmse


@st.cache(suppress_st_warning=True)
def residuals_table(
    residual_weight: int,
    threshold: int,
    forecast_size: int,
    y_true: pd.Series,
    sma_pred: pd.Series,
    sarima_pred: pd.Series,
    rf_pred: pd.Series,
) -> pd.DataFrame:
    """
    Create a table of residual days of consumption values for SMA, ARIMA, RF forecasts.

    Args:
        residual_weight: Weight at time of forecast
        threshold: Threshold of 'empty' stock
        forecast_size: Number of days forecasted
        y_true: True value(s) of prediction.
        sma_pred: Prediction values of SMA forecast.
        sarima_pred: Prediction values of ARIMA forecast.
        rf_pred: Prediction values of Random Forest forecast.

    Returns: Residual days of consumption values as a pandas DataFrame
    """
    y_true_residual = de.residual_days(
        consumption_series=y_true,
        residual_weight=residual_weight,
        threshold=threshold,
        forecast_size=forecast_size,
    )
    sma_residual = de.residual_days(
        consumption_series=sma_pred,
        residual_weight=residual_weight,
        threshold=threshold,
        forecast_size=forecast_size,
    )
    arima_residual = de.residual_days(
        consumption_series=sarima_pred,
        residual_weight=residual_weight,
        threshold=threshold,
        forecast_size=forecast_size,
    )
    rf_residual = de.residual_days(
        consumption_series=rf_pred,
        residual_weight=residual_weight,
        threshold=threshold,
        forecast_size=forecast_size,
    )
    all_residuals = pd.DataFrame(
        [y_true_residual, sma_residual, arima_residual, rf_residual],
        columns=["Remaining Consumption (Days)"],
        index=["True", "Moving Average", "SARIMA", "Random Forest"],
    )
    return all_residuals
