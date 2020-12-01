# smart-simulation

Use this tool learn about weight based smart subscriptions.
[Launch the app!](https://share.streamlit.io/jbpauly/smart-simulation/main.py)

![](app/figures/example.gif)

How does a weight based subscription work? As an example, the Bottomless
service will be described. Bottomless provides customers with a wifi
scale linked to their account. Customers select a delivery preference
from three choices: Never Run Out, Just-In-Time, As Fresh As Possible.
This preference provides additional context for the ordering algorithm.
Customers then store their subscribed product, like coffee beans, on the
scale. Consumption data is captured by weight measurements and monitored
by Bottomless' ordering algorithm. Orders are then triggered to arrive
when expected by the customer based off their resupply preference.

Data and machine learning is at the core of smart subscriptions.
Starting or advancing a smart subscription business requires a quality
dataset. There are multiple agents or factors involved with the working
of smart subscriptions: *Customers, Scales, Product Marketplace,
Warehouses, Shipping*.

## Generate data for yourself by following the steps below

Simulator for Smart Subscription Services.

This package generates data, simulations, and counter-factual scenarios
for weight-based smart subscriptions. Weight-based smart subscriptions
utilize hardware (wifi weight scales) and software (machine learning) to
provide just-in-time delivery of repeatably purchased goods. Bottomless
is the first of its kind.

## Dependencies
Python >= 3.7

## Run Locally in Virtual Environment

Documentation example uses `bash commands` and works for Mac and Linux
OS.

### Clone Repository to Local Directory

Detailed instructions on cloning:
<https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>

`git clone https://github.com/jbpauly/smart-simulation.git`

### Move to The Project Directory

`cd smart-simulation`

### Create the Virtual Environment

`python -m venv venv`

### Activate the Virtual Environment

`source venv/bin/activate`

### Install Requirements

`pip install -r requirements.txt`

### Install the Package

`pip install -e .`

### Run the Package Locally

**Explore CLI Options**

`smart_simulation --help`

`smart_simulation create-consumption --help` :

Usage: smart_simulation batch-simulation [OPTIONS]

```
Usage: smart_simulation batch-simulation [OPTIONS]

  Create a batch of consumer servings and scale weight data generation and
  save to an simulation output directory.
  Args:
        number_instances: Number of instances to simulate with the configuration selections.
        customer_template: Configuration choice for customer template.
        upsample_template: Configuration choice for upsample template.
        delivery_days_template: Configuration choice for delivery days template.
        delivery_skew_template: Configuration choice for delivery skew template.
        weight_template: Configuration choice for weight template.
        quantity_weight_template: Configuration choice for quantity to weight template.
        start_date: Start date of the data generation in YYYY-MM-DD format.
        num_days: Number of days to generate data.
        output_path: Output path for the simulations directory and generated data csv files.

Options:
  -n, --number_instances INTEGER
  -c, --customer_template [Michael|Joe|Liana]
  -u, --upsample_template [Standard]
  -dd, --delivery_days_template [Standard]
  -ds, --delivery_skew_template [skew_early|skew_on_time|skew_late|perfect]
  -w, --weight_template [Standard]
  -qw, --quantity_weight_template [Standard]
  -s, --start_date TEXT
  -d, --num_days INTEGER
  -p, --output_path PATH
  --help                          Show this message and exit.

```

### Example Use

`smart_simulation batch-simulation`

**Check output path for simulations directory**

```
simulations
│   simulations_log.csv
│
└───servings
│   │   01_daily.csv
│   │   01_upsampled.csv
│   │   ...
│
└───weight
    │   01_arrival.csv
    │   ...
```


Credits
-------

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
