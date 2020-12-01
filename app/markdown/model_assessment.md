#### Result Comparison
You likely picked up a few behaviors of our models and results.
1. The SARIMA model almost always has the lowest RMSE score
2. Forecasted residual days, across all models, appear to be more accurate on a longer horizon

These behaviors can be tied back to our consumer framework and data generation process.
The desired consumption and probability of consumption are constant week to week (not day to day).

With that context, the higher performance of the SARIMA model should be expected.
The model had 1 _seasonal_  autoregressive and 1 _seasonal_  moving average component, with a 7 day seasonality.
- Put another way, every Monday, the model only weighs the relationship to consumption the previous Monday(s),
not any other day the previous week.
- This fit our data very well, because there's no dependence day to day in the consumption,
but it is highly correlated week to week (set in the consumer templates).

Since the data was built by random number generators within our framework,
it holds a fairly steady mean over a long time period.
This is why the final _residual days of consumption_ are fairly accurate on a longer horizon.
Day to day, there's inaccuracy in the consumption forecasts, but they balance out over time.

#### Model Selection
A naive model a suitable choice for forecasting new customer consumption.
Just a few consumption datapoints and a naive model can operationalize can the smart subscription.
It's also justifiable to continue using a naive approach longer than one might expect to use it,
_as long as it's meeting the needs of the subscription service_.

Once a consumer starts to outrun the naive model, the next step up is a statistical

#### Machine Learning over Traditional Statistical Approaches
For time series forecasting, a standard regression model is more scalable than a stats based model like SARIMA.
With proper data labeling features,
a single machine learning model could be used for all consumption forecasts, scale to scale.
The traditional statistical approaches are sensitive to non-stationarity,
and unique models should be separately maintained customer to customer or cohort to cohort.

For example, we specified a 7 day seasonality, but will everyone's consumption follow a 7 day pattern?
Medical professionals, factory and refinery workers, travelling business personnel, etc. all have unique schedule.

Lastly, enriching the models with more data features is also easier to accomplish with machine learning models.

This doesn't mean abandon all statistical time series analysis for machine learning!
Traditional time series analysis helps to contextualize your data and could even lead to feature generation.

#### Improving the Usability of Consumption and Residual Days Forecasts
To actually use these forecasts for smart subscriptions would require additional development.

Reasonable next steps:
1. Generate forecast confidence intervals
2. Build a forecasting / purchasing dashboard
3. Serve forecasts to a _human in the loop system_
    - Set a model confidence threshold
    - Set a residual days alert range
    - Send forecasts within the residuals range to system
        - Automate high confidence forecasts
        - Require human decision on low confidence forecast

#### Consumption Calculations
It can't be stated enough that the success of any these models is reliant on the consumption calculations.
Further development of this project or your own interest in this space should be around these calculations.
- What are possible edge cases?
- When should and shouldn't missing data be estimated?
- What correction methods should be used?
