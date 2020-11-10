import pathlib
from os.path import abspath, dirname

import pandas as pd

from smart_simulation import consumer as con
from smart_simulation.cfg_templates import customers as cm_templates
from smart_simulation.cfg_templates import products
from smart_simulation.ds_tools import eda

package_dir = dirname(dirname(abspath(__file__)))
app_path = pathlib.Path(package_dir) / "app"
writing_path = app_path / "markdown"

quantity_to_weight = products.quantity_to_weight["Standard"]
consumption_type_templates = cm_templates.consumption_types
consumption_probability_templates = cm_templates.probabilities
customer_templates = cm_templates.customers


def read_markdown_file(file):
    return (writing_path / file).read_text()


def create_consumption_types_df(
    template: dict = consumption_type_templates,
) -> pd.DataFrame:
    types = {}
    for category, detail in template.items():
        low_high = detail.parameters
        types[category] = low_high
    types_df = pd.DataFrame.from_dict(
        types, orient="index", columns=["Minimum Servings", "Maximum Servings"]
    )
    return types_df


def create_probabilities_df(
    template: dict = consumption_probability_templates,
) -> pd.DataFrame:
    probabilities = pd.DataFrame.from_dict(
        template, orient="index", columns=["Probability"]
    )
    return probabilities


def create_customers_df(template: dict = customer_templates) -> pd.DataFrame:
    customers_dict = {}
    for customer, week in template.items():
        weekdays = {
            day: {
                "probability": week[i].probability,
                "consumption": week[i].consumption,
            }
            for i, day in enumerate(week._fields)
        }
        customers_dict[customer] = pd.DataFrame.from_dict(weekdays)
        customers_df = pd.concat(customers_dict)
        customers_df.index = customers_df.index.rename(["Customer", "Attribute"])
    return customers_df


# @st.cache
def create_desired_servings_df(
    customer_numbers: list = list(customer_templates.keys()),
    days: int = 365,
    start_date: str = "2020-01-01",
) -> dict:
    all_customers = {}
    for template_id in customer_numbers:
        desired_consumption = con.multi_day(template_id, days, start_date)
        all_customers[template_id] = desired_consumption
    customers_df = pd.concat(all_customers, axis=1)
    customers_df.columns = customers_df.columns.droplevel(1)
    return customers_df


# @st.cache
def create_linear_subscription_data(desired_servings: pd.DataFrame, durations: list):
    bag_weight = float(12)
    all_subs_all_consumers = {}
    for customer in list(desired_servings.columns):
        all_subs = {}
        servings = desired_servings.loc[:, customer].squeeze()
        consumption = eda.create_desired_consumption(servings, quantity_to_weight)
        for duration in durations:
            weights = eda.linear_weights(consumption, duration, bag_weight)
            classifications = eda.classify_dates(weights, bag_weight)
            duration_data = (
                pd.concat([weights, classifications], axis=1)
                .reset_index()
                .rename(columns={"index": "date"})
                .assign(duration=duration, customer=customer)
            )
            all_subs[duration] = duration_data
        customer_data = pd.concat(all_subs.values(), ignore_index=True)
        all_subs_all_consumers[customer] = customer_data
    all_data = pd.concat(all_subs_all_consumers.values(), ignore_index=True)
    return all_data
