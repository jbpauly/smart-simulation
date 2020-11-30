#### What are we trying to answer with our forecast?
When should a new order be triggered for customer (X) for just-in-time delivery of product (a)?

This can be broken down further:

1. Estimate of shipping duration
    - How long will shipping take for product (a) to reach customer (X)?
2. Estimate of remaining days of customer's current stock
    - How many days of consumption of product (a) does customer (X) have left in stock?

**_We'll focus on estimating remaining days of customer stock._**

#### Brief review of a few constraints
1. The weight/stock is directly influenced by the customer's consumption
2. Scale weight measurement error does occur
3. New bags come in a variety of weights (12oz, 32oz, etc.)
4. Scale weight should fall within an expected range
    - minimum = 0
    - maximum = weight of new shipment + leftover weight from previous shipment

#### Forecasting Framework
We could get to remaining days of stock by directly forecasting scale weight
 and identifying when weight crosses an **empty stock** threshold.
However, the scale weight measurement is dependent on consumer behavior, **consumption**.
Framing the model around consumption simplifies the problem and offers more flexibility when problems arise.

#### Forecasting Process
Reframe our question:
**Given the current weight and historic consumption, how many remaining days of available consumption remain?**

1. Calculate consumption from historic weight
2. Forecast consumption
3. Calculate theoretical weight based on forecasted consumption and current scale weight
3. Set an empty stock threshold, likely 0 or daily average consumption (oz.)
4. Find the timestamp where the forecast crosses the threshold
