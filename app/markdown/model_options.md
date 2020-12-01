#### Data Assessment
Before we build any models,
it's important to remember our simplified consumer framework.
Holidays, travel, non-traditional schedules were not taken into account.

#### Forecast Horizon
For our example, the forecasts should give proper lead time for ordering and just-in-time delivery.
Our solution will be flexible and allow for a range of forecast horizons: 7-14 days.

#### Naive Model: Simple Moving Average
Naive time series model use basic approaches and can work with just few data points.

Our naive model will take an average over the previous 7 days, from day of prediction, and hold a constant forecast
 for the full forecast horizon.

#### Statistical Model: Seasonal Autoregressive Integrated Moving Average (SARIMA)
Yes, a mouthful, but fairly simple once broken down into individual components.
Statistical models require stationary data or a model that can account for non-stationarity.
Trending and seasonality are examples of non-stationarity,
and a SARIMA model is just an ARMA (autogressive moving average) model which can handle trending and seasonality.
- Integrated component differences the data to removing trending
- Seasonal component breaks out the seasonality for it's own model
    - Essentially end up with two models
        - ARIMA, non-seasonal
        - ARIMA, seasonal

Our SARIMA model will use 1 seasonal autoregressive and 1 seasonal moving average term with a specified seasonality
 of 7 days.

#### Machine Learning Model: Random Forest Regression
Traditional, and deep learning, machine learning models can be applied for time series forecasting.
To do so, the time series dataset is tabularized at a specified window size.
This is advantageous, because it's then easy to add data features for each timestep:
- Day of week, month, year
- Holidays
- Customer status or identifiers

Our machine learning model of choice will be a random forest with 100 trees.
- Random forest is an ensemble of decision trees
- Final regression prediction is the average across all trees


