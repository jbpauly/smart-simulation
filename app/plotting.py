import pandas as pd

import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RANGE_SELECTOR = dict(
    buttons=list(
        [
            dict(count=1, label="1m", step="month", stepmode="backward",),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all"),
        ]
    )
)

RANGE_SLIDER = dict(visible=True)

LAYOUT = dict(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward",),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)


def create_servings_fig(dataset: pd.DataFrame) -> go.Figure:
    x = list(dataset.index)
    quarter_index = int(len(x) / 4)
    last_quarter = [str(x[-quarter_index]), str(x[-1])]
    start_end = [str(x[0]), str(x[-1])]
    data = []
    for consumer in dataset.columns:
        y = list(dataset[consumer])
        bar = go.Bar(name=consumer, x=x, y=y)
        data.append(bar)

    range_slider = RANGE_SLIDER
    range_slider["range"] = start_end

    fig_layout = dict(
        xaxis=dict(
            rangeselector=RANGE_SELECTOR,
            range=last_quarter,
            rangeslider=range_slider,
            type="date",
            tickformat="%a %b-%d<br>%Y",
        ),
        yaxis=dict(title=dict(text="Servings (count)")),
        title=dict(text="Desired Consumption", xanchor="center", yanchor="top", x=0.5),
    )
    fig = go.Figure(dict(data=data, layout=fig_layout))
    return fig


def create_linear_subscriptions_figs(dataset: pd.DataFrame) -> go.Figure:
    unique_customers = dataset.customer.unique()
    unique_durations = dataset.duration.unique()
    num_plots = len(unique_customers) * len(unique_durations)
    groups = dataset.groupby(["duration", "customer"])

    color_map = {"available": "green", "unavailable": "red", "excess": "blue"}
    row_count = 1
    traces = []
    subplot_titles = []
    rows = []
    cols = []
    for customer in unique_customers:
        for duration in unique_durations:
            scenario = groups.get_group((duration, customer))
            trace = go.Scatter(
                x=scenario.date,
                y=scenario.weight,
                mode="markers+lines",
                marker=dict(color=scenario.classification.map(color_map)),
                line=dict(color="black"),
                showlegend=False,
            )
            traces.append(trace)
            sub_plot_title = f"{customer}: {duration} day reoccurring subscription"
            subplot_titles.append(sub_plot_title)
            rows.append(row_count)
            cols.append(1)
            row_count += 1
    fig = make_subplots(
        rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=subplot_titles,
    )
    fig.add_traces(traces, rows=rows, cols=cols)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="red"),
            name="Unavailable",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="green"),
            name="Available",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="blue"),
            name="Excess",
        )
    )
    fig.update_layout(legend_title_text="Coffee Availability")
    fig.update_yaxes(title_text="Weight (oz)")
    fig.update_layout(
        title=dict(
            text="Subscription Schedule Analysis",
            xanchor="center",
            yanchor="top",
            y=0.95,
            x=0.46,
        ),
        height=(150 * row_count),
    )
    return fig


def create_single_subscription_fig(
    dataset: pd.DataFrame, single_bag_weight: int = None
) -> go.Figure:
    color_map = {"available": "green", "unavailable": "red", "excess": "blue"}
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset.date,
            y=dataset.weight,
            mode="markers+lines",
            marker=dict(color=dataset.classification.map(color_map)),
            line=dict(color="black"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="red"),
            name="Out of Stock",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="green"),
            name="Stocked / Fresh",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", color="blue"),
            name="Overstocked / Stale",
        )
    )
    if single_bag_weight:
        single_bag_threshold = pd.Series(single_bag_weight, index=dataset.date)
        fig.add_trace(
            go.Scatter(
                x=single_bag_threshold.index,
                y=single_bag_threshold,
                mode="lines",
                line=dict(dash="dash", color="black"),
                name="New Bag Weight",
                showlegend=True,
            )
        )

    layout = LAYOUT
    layout["title"] = dict(
        text="Consumer's Stock of Coffee", xanchor="center", yanchor="top", x=0.5
    )
    layout["yaxis_title"] = "Weight (oz)"
    layout["xaxis_title"] = "Date"
    layout["xaxis_tickformat"] = "%a %b-%d<br>%Y"
    fig.layout = layout
    return fig


def create_consumption_forecast_fig(
    y_train: pd.Series,
    y_true: pd.Series,
    sma_pred: pd.Series,
    sarima_pred: pd.Series,
    rf_pred: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train.round(2),
            mode="lines",
            line=dict(color="black"),
            name="Historic",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true.index,
            y=y_true.round(2),
            mode="lines",
            line=dict(dash="dash", color="black"),
            name="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_pred.index,
            y=sma_pred.round(2),
            mode="lines",
            line=dict(color="green"),
            name="SMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sarima_pred.index,
            y=sarima_pred.round(2),
            mode="lines",
            line=dict(color="blue"),
            name="SARIMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rf_pred.index,
            y=rf_pred.round(2),
            mode="lines",
            line=dict(color="gray"),
            name="Random Forest",
        )
    )

    train_dates = list(y_train.index)
    pred_dates = list(y_true.index)
    default_range = [str(train_dates[-10]), str(pred_dates[-1])]
    start_end = [str(train_dates[0]), str(pred_dates[-1])]
    range_slider = RANGE_SLIDER
    range_slider["range"] = start_end

    fig_layout = dict(
        xaxis=dict(
            rangeselector=RANGE_SELECTOR,
            range=default_range,
            rangeslider=range_slider,
            type="date",
            tickformat="%a %b-%d<br>%Y",
        ),
        yaxis=dict(title=dict(text="Consumption (oz)")),
        title=dict(text="Consumption Forecast", xanchor="center", yanchor="top", x=0.5),
    )
    fig.layout = fig_layout
    return fig


def create_weight_forecast_fig(
    train_weight: pd.Series,
    true_weight: pd.Series,
    sma_weight: pd.Series,
    sarima_weight: pd.Series,
    rf_weight: pd.Series,
    threshold_weight: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_weight.index,
            y=train_weight.round(2),
            mode="lines",
            line=dict(color="black"),
            name="Historic",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=true_weight.index,
            y=true_weight.round(2),
            mode="lines",
            line=dict(dash="dash", color="black"),
            name="True",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_weight.index,
            y=sma_weight.round(2),
            mode="lines",
            line=dict(color="green"),
            name="SMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sarima_weight.index,
            y=sarima_weight.round(2),
            mode="lines",
            line=dict(color="blue"),
            name="SARIMA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rf_weight.index,
            y=rf_weight.round(2),
            mode="lines",
            line=dict(color="gray"),
            name="Random Forest",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_weight.index,
            y=threshold_weight.round(2),
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Empty Threshold",
        )
    )

    train_dates = list(train_weight.index)
    pred_dates = list(true_weight.index)
    default_range = [str(train_dates[-10]), str(pred_dates[-1])]
    start_end = [str(train_dates[0]), str(pred_dates[-1])]
    range_slider = RANGE_SLIDER
    range_slider["range"] = start_end

    fig_layout = dict(
        xaxis=dict(
            rangeselector=RANGE_SELECTOR,
            range=default_range,
            rangeslider=range_slider,
            type="date",
            tickformat="%a %b-%d<br>%Y",
        ),
        yaxis=dict(title=dict(text="Weight (oz)")),
        title=dict(
            text="Scale Weight Forecast", xanchor="center", yanchor="top", x=0.5
        ),
    )
    fig.layout = fig_layout
    return fig
