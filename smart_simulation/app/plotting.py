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


def create_single_subscription_fig(dataset: pd.DataFrame) -> go.Figure:
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

    layout = LAYOUT
    layout["title"] = dict(
        text="Consumer's Stock of Coffee", xanchor="center", yanchor="top", x=0.5
    )
    layout["legend_title_text"] = "Coffee Availability"
    layout["yaxis_title"] = "Weight (oz)"
    layout["xaxis_title"] = "Date"
    layout["xaxis_tickformat"] = "%a %b-%d<br>%Y"
    fig.layout = layout
    return fig


def create_consumption_forecast_fig(
    y_train: pd.Series,
    y_test: pd.Series,
    sma_pred: pd.Series,
    sarima_pred: pd.Series,
    rf_pred: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train,
            mode="lines",
            line=dict(color="black"),
            name="Historic Consumption",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test,
            mode="lines",
            line=dict(color="red"),
            name="True Consumption",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_pred.index,
            y=sma_pred,
            mode="lines",
            line=dict(color="green"),
            name="Moving Average Forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sarima_pred.index,
            y=sarima_pred,
            mode="lines",
            line=dict(color="blue"),
            name="SARIMA Forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rf_pred.index,
            y=rf_pred,
            mode="lines",
            line=dict(color="gray"),
            name="Random Forest Forecast",
        )
    )

    train_dates = list(y_train.index)
    test_dates = list(y_test.index)
    default_range = [str(train_dates[-10]), str(test_dates[-1])]
    start_end = [str(train_dates[0]), str(test_dates[-1])]
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
