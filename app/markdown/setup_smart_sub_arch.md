#### Smart Ordering

With a basic understanding of how a smart subscription works for the consumer,
we can piece together how it might work behind the scenes.

At a basic level, a smart subscription can be broken down into a few components and processes:

1. Scales, associated to a customer and product
2. Shipments, to restock a specific scale
3. Stock forecasts, of a customers' diminishing local stock of a product
4. Shipment duration estimate, of the next shipment scheduled for each scale
5. Supply side stock of each product

The diagram below represents the relationship between these components and processes.
There is no supply side stock in this diagram, and we'll continue to assume supply is readily available.
Of course, this will not always be the case in real life.
It's also worth nothing ownership of supply vary depending on business model.
For example, Bottomless is a marketplace and subscription service tied together.
Products associated with a scale can change, order to order, based on user selection.
It is then up to the the marketplace suppliers to ensure their products are available to Bottomless customers.

