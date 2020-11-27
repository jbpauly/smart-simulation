from datetime import datetime

from PIL import Image

import streamlit as st
from app import plotting as pl
from app import utilities as util
from smart_simulation.cfg_templates import customers as cm_templates

consumption_type_templates = cm_templates.consumption_types
consumption_probability_templates = cm_templates.probabilities
customer_templates = cm_templates.customers


st.sidebar.header("Building a Smart Subscription")
st.sidebar.subheader("Chose a section:")
sb_problem_introduction_checkbox = st.sidebar.checkbox(
    "Problem Introduction", value=True
)
sb_standard_subscription_analysis_checkbox = st.sidebar.checkbox(
    "Standard Subscription Analysis"
)
sb_on_demand_consumption_checkbox = st.sidebar.checkbox("On-Demand Consumption")
sb_smart_subscription_architecture_checkbox = st.sidebar.checkbox(
    "Smart Subscription Architecture"
)
sb_consumption_forecasting_checkbox = st.sidebar.checkbox("Consumption Forecasting")

st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
with st.sidebar.beta_expander("Try out a Smart Subscription"):
    st.markdown(
        """
    If you enjoy high quality coffee, I recommend Bottomless!
    Your second bag is on the house with a [referral](https://www.bottomless.com/referral/9qcbu1og).
    """,
        unsafe_allow_html=True,
    )
    st.markdown("![Alt Text](https://media.giphy.com/media/dGhlifOCTtSdW/giphy.gif)")


if sb_problem_introduction_checkbox:
    intro_file = util.read_markdown_file("introduction.md")
    st.markdown(intro_file, unsafe_allow_html=True)

if sb_standard_subscription_analysis_checkbox:
    st.markdown(
        """
                ## Standard Subscription Analysis
                """
    )
    with st.beta_expander("Framework", expanded=True):
        setup_file = util.read_markdown_file("setup_scenario.md")
        st.markdown(setup_file, unsafe_allow_html=True)

    with st.beta_expander("Consumer Profiles"):
        profiles_file = util.read_markdown_file("setup_profiles.md")
        st.markdown(profiles_file, unsafe_allow_html=True)

        probabilities_col, consumption_col = st.beta_columns((1, 2))
        probabilities = util.create_probabilities_df()
        probabilities_col.write(probabilities.style.format("{:.2}"))
        consumption_types = util.create_consumption_types_df()
        consumption_col.write(consumption_types)

        sub_group = list(cm_templates.customers.keys())
        customers = util.create_customers_df()
        selected_profile = st.selectbox("Select a Consumer:", sub_group, index=2)

        if selected_profile == "Michael":
            michael_file = util.read_markdown_file("michael.md")
            st.markdown(michael_file, unsafe_allow_html=True)
            st.write(customers.loc["Michael"])
        elif selected_profile == "Liana":
            liana_file = util.read_markdown_file("liana.md")
            st.markdown(liana_file, unsafe_allow_html=True)
            st.write(customers.loc["Liana"])
        elif selected_profile == "Joe":
            joe_file = util.read_markdown_file("joe.md")
            st.markdown(joe_file, unsafe_allow_html=True)
            st.write(customers.loc["Joe"])

    with st.beta_expander("Desired Consumption"):
        consumption_file = util.read_markdown_file("setup_consumption.md")
        st.markdown(consumption_file, unsafe_allow_html=True)
        if st.checkbox("View pseudo-code for desired consumption generation"):
            pseudo_code_servings_file = util.read_markdown_file(
                "pseudocode_single_day_servings.md"
            )
            st.markdown(pseudo_code_servings_file, unsafe_allow_html=True)

        consumption_plot_checkbox, consumption_table_check_box = st.beta_columns((1, 1))
        desired_servings = util.create_desired_servings_df()
        if consumption_plot_checkbox.checkbox("View Desired Consumption"):
            if consumption_table_check_box.checkbox("View as Table"):
                servings_with_dt_fmt = desired_servings.copy()
                servings_with_dt_fmt.index = servings_with_dt_fmt.index.strftime(
                    "%-a %b-%d %y"
                )
                st.write(servings_with_dt_fmt)
            else:
                chart = pl.create_servings_fig(desired_servings)
                st.plotly_chart(chart)
    with st.beta_expander("Analyze Subscription"):
        analyze_standard_sub_file = util.read_markdown_file(
            "setup_analyze_standard_sub.md"
        )
        st.markdown(analyze_standard_sub_file, unsafe_allow_html=True)
        try:
            servings_created = desired_servings is not None
        except NameError:
            st.write(
                "You must create desired consumption with the step above before analyzing the subscriptions."
            )
        else:
            linear_subscription_data = util.create_linear_subscription_data(
                desired_servings, [14, 21, 28, 35, 42]
            )

            select_cust_col, select_duration_col = st.beta_columns((1, 3))
            selected_customer = select_cust_col.radio("Select Customer:", sub_group)
            selected_duration = select_duration_col.select_slider(
                "Select Subscription Period (days):", [14, 21, 28, 35, 42]
            )
            sub_subscription_data = linear_subscription_data.loc[
                (linear_subscription_data["customer"] == selected_customer)
                & (linear_subscription_data["duration"] == selected_duration)
            ]
            subscription_fig = pl.create_single_subscription_fig(sub_subscription_data)
            st.plotly_chart(subscription_fig)
            # TODO create bar chart or table for count of day classification

    with st.beta_expander("Standard Subscription Breakdown"):
        close_standard_sub_file = util.read_markdown_file(
            "setup_standard_sub_closeout.md"
        )
        st.markdown(close_standard_sub_file, unsafe_allow_html=True)
        heat_map_col, dist_plot_col = st.beta_columns((3, 1))
        consumption_heatmap = Image.open("figures/bottomless_consumption.png")
        heat_map_col.image(
            consumption_heatmap,
            caption="A Wildly Inconsistent 2020. Fitting, right?",
            use_column_width=True,
        )
        consumption_dist = Image.open("figures/personal_order_dist.png")
        dist_plot_col.image(
            consumption_dist, use_column_width=True,
        )


if sb_on_demand_consumption_checkbox:
    st.markdown(
        """
                ## Subscriptions in Service of On-Demand Consumption
                On-demand consumption is the ultimate consumer experience,
                and subscriptions are a means to provide supply for reoccurring demand.
                """
    )
    with st.beta_expander("Digital vs. Physical", expanded=True):
        on_demand_file = util.read_markdown_file("setup_on_demand.md")
        st.markdown(on_demand_file, unsafe_allow_html=True)
    with st.beta_expander("Smart(er) not Harder"):
        smart_subscription_file = util.read_markdown_file("setup_smart_subscription.md")
        st.markdown(smart_subscription_file, unsafe_allow_html=True)
        st.markdown("**Or watch Bottomless in action:**")
        st.video(
            "https://bottomless-products.s3-us-west-1.amazonaws.com/miscellaneous/video/reordering-1920.webm"
        )

    # TODO review smart subscription for Michael, Liana, Joe

if sb_smart_subscription_architecture_checkbox:
    st.markdown("## Architecting Smart Subscriptions")
    st.markdown(
        "![Alt Text](https://media.giphy.com/media/fVeAI9dyD5ssIFyOyM/giphy.gif)"
    )
    st.write("section in-progress, check back later for progress.")
    # TODO explain the architecture

if sb_consumption_forecasting_checkbox:
    st.markdown("## Forecasting Consumption")
    st.markdown(
        "![Alt Text](https://media.giphy.com/media/fVeAI9dyD5ssIFyOyM/giphy.gif)"
    )
    st.write("section in-progress, check back later for progress.")
    # TODO explain focus on forecasting consumption
