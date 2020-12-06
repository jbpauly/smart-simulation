import base64
import pathlib

import numpy as np
import pandas as pd

import app.forecasting as forecasting
import app.plotting as pl
import app.utilities as util
import smart_simulation.cfg_templates.config as config
import smart_simulation.cfg_templates.customers as cm_templates
import smart_simulation.ds_tools.data_eng as de
import streamlit as st

package_path = pathlib.Path(config.package_dir)
app_path = package_path / "app"

consumption_type_templates = cm_templates.consumption_types
consumption_probability_templates = cm_templates.probabilities
customer_templates = cm_templates.customers

st.set_page_config(
    page_title="Smart Subscriptions",
    page_icon="📦",
    layout="centered",
    initial_sidebar_state="expanded",
)
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
                st.plotly_chart(chart, use_container_width=True)
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
            subscription_fig = pl.create_single_subscription_fig(
                sub_subscription_data, single_bag_weight=12
            )
            st.plotly_chart(subscription_fig, use_container_width=True)

    with st.beta_expander("Standard Subscription Breakdown"):
        close_standard_sub_file = util.read_markdown_file(
            "setup_standard_sub_closeout.md"
        )
        st.markdown(close_standard_sub_file, unsafe_allow_html=True)
        heat_map_col, dist_plot_col = st.beta_columns((3, 1))
        heat_map_col.image(
            "app/figures/bottomless_consumption.png",
            caption="A Wildly Inconsistent 2020. Fitting, right?",
            use_column_width=True,
        )
        dist_plot_col.image(
            "app/figures/personal_order_dist.png", use_column_width=True,
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
        st.text("")
        st.markdown("**Or watch Bottomless in action:**")
        st.image(
            "app/figures/bottomless_scale.gif", use_column_width=True,
        )


if sb_smart_subscription_architecture_checkbox:
    st.markdown("## Technical Architecture")
    setup_arch_file = util.read_markdown_file("setup_smart_sub_arch.md")
    st.markdown(setup_arch_file, unsafe_allow_html=True)
    st.image(
        "app/figures/architecture.png", use_column_width=True,
    )

if sb_consumption_forecasting_checkbox:
    st.markdown("## Consumption Forecasting")
    with st.beta_expander("Forecasting Approach", expanded=True):
        model_approach = util.read_markdown_file("model_approach.md")
        st.markdown(model_approach, unsafe_allow_html=True)

    with st.beta_expander("Consumption Calculation", expanded=False):
        consumption_calculation = util.read_markdown_file("consumption_calculation.md")
        st.markdown(consumption_calculation, unsafe_allow_html=True)

    with st.beta_expander("Model Options", expanded=False):
        model_options = util.read_markdown_file("model_options.md")
        st.markdown(model_options, unsafe_allow_html=True)

    with st.beta_expander("Forecasting Example", expanded=False):
        st.write("")
        st.markdown(
            """
        Test the models at various forecasting horizons and on different dates with an sample dataset.
        You can also set the _empty stock threshold_. For this dataset, the average consumption is 1.25 oz.
        """
        )
        st.write("")
        sample_weight = de.load_sim_data((app_path / "sample_weight.csv"), ["weight"])
        eod_weights = de.eod_weights(sample_weight.weight)
        consumption_adjustments = de.create_consumption_adjustments(
            weight_series=eod_weights, adjustment_weight=14
        )
        consumption = de.calculate_consumption(
            weight_series=eod_weights, adjustments=consumption_adjustments
        )
        avg_consumption = float(
            "{:.2f}".format(
                de.calcuate_consumption_avg(
                    consumption_series=consumption, all_timesteps=False
                )
            )
        )
        threshold_col, forecast_range_col = st.beta_columns(2)
        threshold_options = {
            0: "True Zero: 0 oz.",
            avg_consumption: f"Avg. Consumption: {avg_consumption} oz.",
        }
        threshold = threshold_col.radio(
            label="Empty Stock Threshold",
            options=list(threshold_options.keys()),
            format_func=threshold_options.get,
        )
        modeling_data = pd.concat([eod_weights, consumption], axis=1)
        remaining_consumption_days = de.all_residual_days(
            weights_consumption=modeling_data, threshold=threshold
        )

        modeling_data = pd.concat([modeling_data, remaining_consumption_days], axis=1)

        prediction_dates = forecasting.create_prediction_dates(
            sample_weight.index, min_train=50, max_forecast=14
        )
        forecast_size_range = np.arange(7, 15)
        forecast_size = forecast_range_col.select_slider(
            label="Select Forecast Range (days)", options=list(forecast_size_range),
        )
        pred_date = st.select_slider(
            label="Select Forecast Date",
            options=list(prediction_dates.strftime("%B %d, %Y")),
        )
        pred_date = pd.to_datetime(pred_date)
        train_end_date = pred_date - pd.Timedelta("1D")
        test_end_date = pred_date + pd.Timedelta(str(forecast_size - 1) + "D")
        y_train = modeling_data.consumption[:train_end_date]
        y_true = modeling_data.consumption[pred_date:test_end_date]
        sma_forecast, sarima_forecast, rf_forecast = forecasting.forecast_consumption(
            forecast_size=forecast_size, y_train=y_train
        )
        start_weight = modeling_data.weight[pred_date - pd.Timedelta("1D")]

        consumption_forecast_fig = pl.create_consumption_forecast_fig(
            y_train=y_train,
            y_true=y_true,
            sma_pred=sma_forecast,
            sarima_pred=sarima_forecast,
            rf_pred=rf_forecast,
        )
        st.write("")
        st.plotly_chart(consumption_forecast_fig, use_container_width=True)

        rmse_explanation, rmse_table = st.beta_columns(2)
        rmse_dateset = forecasting.rmse_table(
            y_true=y_true,
            sma_pred=sma_forecast,
            sarima_pred=sarima_forecast,
            rf_pred=rf_forecast,
        )
        rmse_file = util.read_markdown_file("rmse.md")
        rmse_explanation.markdown(rmse_file, unsafe_allow_html=True)
        rmse_table.table(rmse_dateset.style.format("{:.2}"))

        train_weight = modeling_data.weight[y_train.index]
        true_theoretical_weight = de.calculate_theoretical_weights(
            start_weight=start_weight, consumption_series=y_true
        )
        sma_theoretical_weight = de.calculate_theoretical_weights(
            start_weight=start_weight, consumption_series=sma_forecast
        )
        sarima_theoretical_weight = de.calculate_theoretical_weights(
            start_weight=start_weight, consumption_series=sarima_forecast
        )
        rf_theoretical_weight = de.calculate_theoretical_weights(
            start_weight=start_weight, consumption_series=rf_forecast
        )
        train_test_range = y_train.index.union(y_true.index)
        threshold_range = pd.Series(data=threshold, index=train_test_range)

        weight_forecast_fig = pl.create_weight_forecast_fig(
            train_weight=train_weight,
            true_weight=true_theoretical_weight,
            sma_weight=sma_theoretical_weight,
            sarima_weight=sarima_theoretical_weight,
            rf_weight=rf_theoretical_weight,
            threshold_weight=threshold_range,
        )
        st.plotly_chart(weight_forecast_fig, use_container_width=True)

        residuals_dataset = forecasting.residuals_table(
            residual_weight=start_weight,
            threshold=threshold,
            forecast_size=forecast_size,
            y_true=y_true,
            sma_pred=sma_forecast,
            sarima_pred=sarima_forecast,
            rf_pred=rf_forecast,
        )
        residuals_explanation, residuals_table = st.beta_columns(2)
        residuals_file = util.read_markdown_file("residual_days.md")
        residuals_explanation.markdown(residuals_file, unsafe_allow_html=True)
        residuals_table.table(residuals_dataset)
        st.write("")

    with st.beta_expander(
        "Forecasting Assessment and Expanded Application", expanded=False
    ):
        model_assessment = util.read_markdown_file("model_assessment.md")
        st.markdown(model_assessment, unsafe_allow_html=True)
