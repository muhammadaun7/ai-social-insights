import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Select, Div, HoverTool
from bokeh.layouts import row, column, layout

# === Data Setup ===
DATA_FOLDER = "data"
clients = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
clients.sort()

client_select = Select(title="Client", value=clients[0], options=clients)
platform_select = Select(title="Platform")
metric_select = Select(title="Metric")
model_select = Select(title="Model", value="LSTM", options=["LSTM", "Prophet"])

source_actual = ColumnDataSource(data=dict(x=[], y=[]))
source_forecast = ColumnDataSource(data=dict(x=[], y=[]))
source_anomaly = ColumnDataSource(data=dict(x=[], y=[]))

note_div = Div(text="", width=450, styles={"color": "#34495e", "font-size": "16px", "margin-top": "10px"})
insight_div = Div(text="", width=450, styles={"color": "#2c3e50", "font-size": "14px"})

# === Plot ===
p = figure(title="Forecast", x_axis_type="datetime", height=450, width=450)
p.line("x", "y", source=source_actual, legend_label="Actual", line_width=2, color="blue")
p.line("x", "y", source=source_forecast, legend_label="Forecast", line_width=2, color="orange")
p.circle("x", "y", source=source_anomaly, size=8, color="red", legend_label="Anomaly")
hover = HoverTool(tooltips=[("Date", "@x{%F}"), ("Value", "@y{0.0}")], formatters={"@x": "datetime"})
p.add_tools(hover)
p.legend.location = "top_left"

# === Functions ===
def load_client_data(client_file):
    path = os.path.join(DATA_FOLDER, client_file)
    try:
        df = pd.read_excel(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except:
        return pd.DataFrame(columns=["Date", "Platform", "Metric", "Value"])

def update_options():
    global current_df
    current_df = load_client_data(client_select.value)
    platforms = current_df["Platform"].unique().tolist()
    metrics = current_df["Metric"].unique().tolist()
    if platforms:
        platform_select.options = platforms
        platform_select.value = platforms[0]
    if metrics:
        metric_select.options = metrics
        metric_select.value = metrics[0]

def prepare_lstm_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

def lstm_forecast(series, n_steps=10, n_forecast=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    X, y = prepare_lstm_data(scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([LSTM(50, activation='relu', input_shape=(n_steps, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    input_seq = scaled[-n_steps:].reshape(1, n_steps, 1)
    preds = []
    for _ in range(n_forecast):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast_scaled = np.array(preds).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast_scaled).flatten()
    return forecast

def prophet_forecast(df, periods=60):
    df = df.rename(columns={"Date": "ds", "Value": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast = forecast[forecast["ds"] > df["ds"].max()]
    return forecast["ds"], forecast["yhat"]

def detect_anomalies(series, window=7, threshold=2.5):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    anomalies = (abs(series - rolling_mean) > threshold * rolling_std)
    return series[anomalies]

def compute_rmse(actual, predicted):
    if len(predicted) > len(actual):
        predicted = predicted[:len(actual)]
    return sqrt(mean_squared_error(actual, predicted))

def generate_insight(df):
    if df.empty:
        return "No data available to generate insights."
    recent = df[df["Date"] > df["Date"].max() - pd.Timedelta(days=7)]
    previous = df[(df["Date"] <= df["Date"].max() - pd.Timedelta(days=7)) &
                  (df["Date"] > df["Date"].max() - pd.Timedelta(days=14))]
    if recent.empty or previous.empty:
        return "Not enough data to compare recent trends."
    recent_mean = recent["Value"].mean()
    previous_mean = previous["Value"].mean()
    change_pct = ((recent_mean - previous_mean) / previous_mean) * 100
    change_pct = round(change_pct, 1)
    if change_pct > 0:
        return f"📈 The average value increased by {change_pct}% in the last 7 days."
    elif change_pct < 0:
        return f"📉 The average value decreased by {abs(change_pct)}% in the last 7 days."
    else:
        return "⚖️ No significant change in the past 7 days."

def update_plot():
    if not current_df.empty:
        df = current_df[(current_df["Platform"] == platform_select.value) &
                        (current_df["Metric"] == metric_select.value)].copy()
        if df.empty:
            note_div.text = "No data available for selection."
            insight_div.text = ""
            source_actual.data = dict(x=[], y=[])
            source_forecast.data = dict(x=[], y=[])
            source_anomaly.data = dict(x=[], y=[])
            return

        actual_dates = df["Date"]
        actual_values = df["Value"]
        source_actual.data = dict(x=actual_dates, y=actual_values)

        # Forecast
        if model_select.value == "LSTM":
            future_dates = pd.date_range(start=actual_dates.max() + pd.Timedelta(days=1), periods=60)
            forecast = lstm_forecast(actual_values.values)
        else:
            future_dates, forecast_series = prophet_forecast(df)
            forecast = forecast_series.values

        source_forecast.data = dict(x=future_dates, y=forecast)

        # RMSE calculation
        overlap = min(10, len(actual_values))
        if model_select.value == "LSTM":
            test_forecast = lstm_forecast(actual_values.values, n_steps=10, n_forecast=overlap)
        else:
            forecast_df = df.rename(columns={"Date": "ds", "Value": "y"})
            model = Prophet()
            model.fit(forecast_df)
            future = model.make_future_dataframe(periods=overlap)
            forecast_vals = model.predict(future)["yhat"].values[-overlap:]
            test_forecast = forecast_vals

        rmse = compute_rmse(actual_values.values[-overlap:], test_forecast)
        note_div.text = f"<b>RMSE ({model_select.value}):</b> {rmse:.2f}"

        # Anomalies
        anomalies = detect_anomalies(df.set_index("Date")["Value"])
        source_anomaly.data = dict(x=anomalies.index, y=anomalies.values)

        # Insight
        insight_text = generate_insight(df)
        insight_div.text = f"<b>AI Insight:</b><br>{insight_text}"
    else:
        note_div.text = "No data to show."
        insight_div.text = ""
        source_actual.data = dict(x=[], y=[])
        source_forecast.data = dict(x=[], y=[])
        source_anomaly.data = dict(x=[], y=[])

def on_change(attr, old, new):
    update_plot()

# === Events ===
client_select.on_change("value", lambda attr, old, new: (update_options(), update_plot()))
platform_select.on_change("value", on_change)
metric_select.on_change("value", on_change)
model_select.on_change("value", on_change)

# === Initial Load ===
update_options()
update_plot()

# === UI Layout ===
header = Div(text="""
    <div style='display:flex; align-items:center;'>
        <h1 style='color:#2c3e50; margin:0;'>AI Forecast Dashboard</h1>
    </div>
""", width=450)

for widget in [client_select, platform_select, metric_select, model_select]:
    widget.width = 150
    widget.css_classes = ["custom-dropdown"]

main_layout = layout([
    [header],
    [row(client_select, platform_select, metric_select, model_select)],
    [p],
    [note_div],
    [insight_div]
], sizing_mode="fixed")

curdoc().add_root(main_layout)
curdoc().title = "AI Social Insights Dashboard"
