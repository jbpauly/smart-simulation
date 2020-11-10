import streamlit as st
from app import plotting as pl
from app import utilities as util
from smart_simulation.cfg_templates import customers as cm_templates

consumption_type_templates = cm_templates.consumption_types
consumption_probability_templates = cm_templates.probabilities
customer_templates = cm_templates.customers

intro_file = util.read_markdown_file("introduction.md")
st.markdown(intro_file, unsafe_allow_html=True)

st.markdown("## Standard Subscription Analysis")

setup_file = util.read_markdown_file("setup_scenario.md")
st.markdown(setup_file, unsafe_allow_html=True)


probabilities = util.create_probabilities_df()
st.table(probabilities.style.format("{:.2}"))
consumption_types = util.create_consumption_types_df()
st.write(consumption_types)

st.write("Now that we have our framework, we can build the consumer profiles.")
sub_group = list(cm_templates.customers.keys())
customers = util.create_customers_df()

st.markdown("### Consumer Profile: Micheal")
michael_file = util.read_markdown_file("michael.md")
st.markdown(michael_file, unsafe_allow_html=True)
st.write(customers.loc["Michael"])

st.markdown("### Consumer Profile: Liana")
liana_file = util.read_markdown_file("liana.md")
st.markdown(liana_file, unsafe_allow_html=True)
st.write(customers.loc["Liana"])

st.markdown("### Consumer Profile: Joe")
joe_file = util.read_markdown_file("joe.md")
st.markdown(joe_file, unsafe_allow_html=True)
st.write(customers.loc["Joe"])

desired_servings = util.create_desired_servings_df()
st.text("")
if st.checkbox("Create Desired Servings"):
    st.subheader("Desired Servings")
    if st.checkbox("View as Table"):
        st.write(desired_servings)
    else:
        chart = pl.create_servings_fig(desired_servings)
        st.plotly_chart(chart)

linear_subscription_data = util.create_linear_subscription_data(
    desired_servings, [14, 21, 28, 35, 42]
)
selected_customer = st.radio("Select Customer:", sub_group)
selected_duration = st.select_slider(
    "Select Subscription Period (days):", [14, 21, 28, 35, 42]
)
sub_subscription_data = linear_subscription_data.loc[
    (linear_subscription_data["customer"] == selected_customer)
    & (linear_subscription_data["duration"] == selected_duration)
]
subscription_fig = pl.create_single_subscription_fig(sub_subscription_data)
st.plotly_chart(subscription_fig)
