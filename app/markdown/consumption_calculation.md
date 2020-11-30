In theory, calculating consumption from measured weight is simple.

$$consumption_{t} = weight_{t-1} - weight_{t} + correction_{t}$$

The **correction** value is a catchall new product arrivals, null measurements, scale miscalibration, and unexpected
 behaviors (user error).

For now, we're still assuming **clean** consumption, but keep in mind there are many correction edge cases.
Calculating accurate consumption one of the most critical components of weight based smart subscriptions.
**It does not work without _legible_ consumption information.**
