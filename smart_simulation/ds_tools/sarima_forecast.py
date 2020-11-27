import numpy as np
import pandas as pd

import pmdarima as pm


def grid_search_sarima(
    consumption: pd.Series,
    start_p: int = 0,
    start_q: int = 0,
    max_p: int = 7,
    max_q: int = 7,
    d: int = None,
    start_p_seasonal: int = 0,
    start_q_seasonal: int = 0,
    d_seasonal: int = None,
    m: int = 7,
) -> pm.arima.ARIMA:
    """
    Find and return an optimal SARIMA model using pmdarima auto_arima() grid search.
    Args:
        consumption: consumption series to fit the model
        start_p: starting non-seasonal p term
        start_q: starting non-seasonal q term
        max_p: maximum non-seasonal p term
        max_q: maximum non-seasonal q term
        d: degree of differencing of non-seasonal model component
        start_p_seasonal: starting seasonal p term
        start_q_seasonal: starting seasonal q term
        d_seasonal: degree of differencing of seasonal model component
        m: number of timesteps per season

    Returns: The pmdarima ARIMA model.
    """
    model = pm.auto_arima(
        consumption,
        start_p=start_p,
        start_q=start_q,
        test="adf",
        max_p=max_p,
        max_q=max_q,
        m=m,
        start_P=start_p_seasonal,
        start_Q=start_q_seasonal,
        seasonal=True,
        d=d,
        D=d_seasonal,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def update_sarima_model(
    model: pm.arima.ARIMA, new_consumption: pd.Series
) -> pm.arima.ARIMA:
    """
    Update fitted model's maximum likelihood estimation using with new consumption values.
    Args:
        model: Previously fitted pmdarima arima model.
        new_consumption: New consumption values previously unused to fit old model.

    Returns: Newly updated model.
    """
    consumption = new_consumption.values()
    model.update(consumption)
    return model


def predict_consumption(
    model: pm.arima.ARIMA,
    n_periods: int = 7,
    return_conf_int: bool = True,
    alpha: float = 0.05,
) -> np.array or (np.array, np.array):
    """
    Predict, out of place, consumption for a given number of periods.
    Args:
        alpha: The confidence intervals for the forecasts are (1 - alpha) %
        model: pmdarima fitted arima model
        n_periods: Number of periods to predict
        return_conf_int: Option to return confidence interval of predicted values.

    Returns: Predicted values and optionally the confidence intervals of predicted values.
    """
    forecast_output = model.predict(
        n_periods=n_periods, return_conf_int=return_conf_int, alpha=alpha
    )
    return forecast_output
