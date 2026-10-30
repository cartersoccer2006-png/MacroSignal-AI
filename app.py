
import streamlit as st
import pandas as pd
import altair as alt
import requests
import zipfile
import textwrap
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="MacroSignal AI",
    page_icon="📈",
    layout="wide"
)










st.markdown(
    '<div style="background:#080A0D; border:1px solid #343B44; border-left:5px solid #FF9F1C; padding:26px; border-radius:8px;">'
    '<p style="color:#FF9F1C; font-size:13px; letter-spacing:2px;">MACROECONOMIC INTELLIGENCE / CANADA</p>'
    '<h1 style="color:white; font-size:42px; margin:5px 0;">MACROSIGNAL <span style="color:#FF9F1C;">AI</span></h1>'
    '<p style="color:#B5C0CD; font-size:15px;">CANADIAN LABOUR MARKET INTELLIGENCE | SIX-MONTH UNEMPLOYMENT FORECAST</p>'
    '<hr style="border:0; border-top:1px solid #343B44;">'
    '<p style="color:#8B9BAD; font-size:12px;">DATA SOURCE: STATISTICS CANADA | MODEL: RANDOM FOREST | STATUS: EXPERIMENTAL</p>'
    '</div>',
    unsafe_allow_html=True
)


# Download and prepare official economic data

# Load the smaller unemployment dataset
@st.cache_data
def get_data():
    df = pd.read_csv("unemployment_data.csv")

    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df["VALUE"] = pd.to_numeric(df["VALUE"])

    df = df.dropna()
    df = df.sort_values("REF_DATE")
    df = df.drop_duplicates("REF_DATE")

    return df.reset_index(drop=True)

    url = (
        "https://www150.statcan.gc.ca/n1/tbl/csv/"
        "14100287-eng.zip"
    )

    
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    response.raise_for_status()

    # Check whether Statistics Canada returned a ZIP file
    if not zipfile.is_zipfile(BytesIO(response.content)):
        raise ValueError(
            f"Statistics Canada did not return a ZIP file. "
            f"HTTP status: {response.status_code}, "
            f"Content type: {response.headers.get('Content-Type')}, "
            f"Final URL: {response.url}"
        )

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        with z.open("14100287.csv") as file:
            data = pd.read_csv(file, low_memory=False)


    df = data[
        (data["GEO"] == "Canada")
        & (data["Labour force characteristics"]
           == "Unemployment rate")
        & (data["Gender"] == "Total - Gender")
        & (data["Age group"] == "15 years and over")
        & (data["Statistics"] == "Estimate")
        & (data["Data type"] == "Seasonally adjusted")
    ].copy()

    df = df[["REF_DATE", "VALUE"]].dropna()
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df = df.sort_values("REF_DATE")
    df = df.drop_duplicates("REF_DATE")

    return df.reset_index(drop=True)


try:
    unemployment = get_data()
except Exception as e:
    st.error(f"Could not load economic data: {e}")
    st.stop()

# Prepare 12 months of historical unemployment
df = unemployment.copy()

for lag in range(12):
    df[f"lag_{lag}"] = df["VALUE"].shift(lag)

# Target: unemployment 6 months later
df["future_rate"] = df["VALUE"].shift(-6)

features = [f"lag_{lag}" for lag in range(12)]

# Only rows with known future outcomes can be used
# for training and historical testing
labeled = df.dropna().reset_index(drop=True)

if len(labeled) < 40:
    st.error("Not enough historical data to train the model.")
    st.stop()

# Historical train/test split
split = int(len(labeled) * 0.8)
train = labeled.iloc[:split - 6]
test = labeled.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(train[features], train["future_rate"])

# Evaluate the AI on unseen historical data
predictions = model.predict(test[features])

ai_error = mean_absolute_error(
    test["future_rate"],
    predictions
)

# Traditional forecast: no change from current rate
baseline_error = mean_absolute_error(
    test["future_rate"],
    test["lag_0"]
)

# Train a final model on all available labeled data
final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

final_model.fit(
    labeled[features],
    labeled["future_rate"]
)

# Predict from the latest 12-month history
latest = df.dropna(subset=features).iloc[[-1]]

forecast = final_model.predict(
    latest[features]
)[0]

current_rate = unemployment["VALUE"].iloc[-1]
latest_date = unemployment["REF_DATE"].iloc[-1]
forecast_date = latest_date + pd.DateOffset(months=6)

st.divider()


# BLOOMBERG-STYLE FORECAST METRICS

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        f'<div style="background:#0A0D13; border:1px solid #343B44; border-top:3px solid #FF9F1C; padding:24px; border-radius:6px; min-height:155px;">'
        f'<div style="color:#FF9F1C; font-size:12px; font-weight:bold; letter-spacing:1px;">CURRENT UNEMPLOYMENT</div>'
        f'<div style="color:#FFFFFF; font-size:38px; font-weight:bold; margin-top:16px;">{current_rate:.1f}%</div>'
        f'<div style="color:#8B9BAD; font-size:12px; margin-top:10px;">LATEST OBSERVED RATE</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div style="background:#0A0D13; border:1px solid #343B44; border-top:3px solid #22C55E; padding:24px; border-radius:6px; min-height:155px;">'
        f'<div style="color:#22C55E; font-size:12px; font-weight:bold; letter-spacing:1px;">AI SIX-MONTH FORECAST</div>'
        f'<div style="color:#FFFFFF; font-size:38px; font-weight:bold; margin-top:16px;">{forecast:.2f}%</div>'
        f'<div style="color:#8B9BAD; font-size:12px; margin-top:10px;">EXPERIMENTAL ESTIMATE</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div style="background:#0A0D13; border:1px solid #343B44; border-top:3px solid #60A5FA; padding:24px; border-radius:6px; min-height:155px;">'
        f'<div style="color:#60A5FA; font-size:12px; font-weight:bold; letter-spacing:1px;">FORECAST TARGET</div>'
        f'<div style="color:#FFFFFF; font-size:30px; font-weight:bold; margin-top:20px;">{forecast_date:%b %Y}</div>'
        f'<div style="color:#8B9BAD; font-size:12px; margin-top:10px;">SIX MONTHS AHEAD</div>'
        f'</div>',
        unsafe_allow_html=True
    )


st.caption(
    f"Latest observation: {latest_date:%B %Y}. "
    "Forecasts are estimates, not guarantees."
)



# HISTORICAL UNEMPLOYMENT GRAPH

st.subheader("Canadian Unemployment History")

history = unemployment.copy()
history["REF_DATE"] = pd.to_datetime(history["REF_DATE"])
history = history.dropna(subset=["REF_DATE", "VALUE"])
history = history.sort_values("REF_DATE").tail(120)

history_chart = (
    alt.Chart(history)
    .mark_line(color="#FF9F1C", strokeWidth=2)
    .encode(
        x=alt.X(
            "REF_DATE:T",
            title="Date",
            axis=alt.Axis(format="%b %Y", labelAngle=-35, tickCount=8)
        ),
        y=alt.Y(
            "VALUE:Q",
            title="Unemployment Rate (%)",
            scale=alt.Scale(zero=False)
        ),
        tooltip=[
            alt.Tooltip("REF_DATE:T", title="Month", format="%B %Y"),
            alt.Tooltip("VALUE:Q", title="Unemployment Rate", format=".2f")
        ]
    )
    .properties(height=350)
    .interactive()
)

st.altair_chart(history_chart, use_container_width=True)


# SIX-MONTH AI FORECAST GRAPH

st.subheader("Six-Month AI Forecast")

recent = unemployment.tail(12).copy()

chart_data = recent.set_index("REF_DATE")[["VALUE"]]
chart_data.columns = ["Historical"]
chart_data["AI Forecast"] = float("nan")

# Connect the forecast to the latest known rate
chart_data.loc[latest_date, "AI Forecast"] = current_rate
chart_data.loc[forecast_date, "AI Forecast"] = forecast

chart_data = chart_data.sort_index().reset_index()
chart_data = chart_data.rename(columns={"REF_DATE": "Date"})

# Convert columns to the format Altair expects
forecast_plot = chart_data.melt(
    id_vars="Date",
    value_vars=["Historical", "AI Forecast"],
    var_name="Series",
    value_name="Rate"
).dropna(subset=["Rate"])

# Set a sensible, non-negative vertical range
minimum_rate = min(recent["VALUE"].min(), current_rate, forecast)
maximum_rate = max(recent["VALUE"].max(), current_rate, forecast)

y_min = max(0, float(minimum_rate) - 0.5)
y_max = float(maximum_rate) + 0.5

forecast_chart = (
    alt.Chart(forecast_plot)
    .mark_line(strokeWidth=3, point=True)
    .encode(
        x=alt.X(
            "Date:T",
            title="Date",
            axis=alt.Axis(format="%b %Y", labelAngle=-35, tickCount=6)
        ),
        y=alt.Y(
            "Rate:Q",
            title="Unemployment Rate (%)",
            scale=alt.Scale(domain=[y_min, y_max])
        ),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=["Historical", "AI Forecast"],
                range=["#FF9F1C", "#22C55E"]
            ),
            legend=alt.Legend(title="")
        ),
        tooltip=[
            alt.Tooltip("Date:T", title="Date", format="%B %Y"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Rate:Q", title="Rate (%)", format=".2f")
        ]
    )
    .properties(height=400)
)

st.altair_chart(forecast_chart, use_container_width=True)

st.caption(
    "The amber line shows the latest 12 months of observed unemployment. "
    "The green line connects the latest observation to the six-month "
    "AI forecast; it does not represent individual monthly predictions. "
    "The vertical scale is zoomed in for readability."
)




# Compare historical AI predictions with actual outcomes
st.subheader("Historical Backtesting: AI vs Baseline vs Actual")

# Allow users to change how much history is displayed
months_to_show = st.slider(
    "Months of historical test results",
    min_value=6,
    max_value=min(60, len(test)),
    value=min(24, len(test))
)

# Correctly calculate forecast dates
forecast_dates = (
    pd.to_datetime(test["REF_DATE"])
    + pd.DateOffset(months=6)
)

# Prepare test results
comparison = pd.DataFrame({
    "Date": forecast_dates.to_numpy(),
    "Actual Unemployment": test["future_rate"].to_numpy(),
    "AI Prediction": predictions,
    "Baseline Prediction": test["lag_0"].to_numpy()
})

# Select the most recent test observations
comparison = comparison.tail(months_to_show)

# Display historical comparison with clearer axes

comparison_plot = comparison.melt(
    id_vars="Date",
    value_vars=[
        "Actual Unemployment",
        "AI Prediction",
        "Baseline Prediction"
    ],
    var_name="Series",
    value_name="Rate"
)

backtest_chart = (
    alt.Chart(comparison_plot)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X(
            "Date:T",
            title="Forecast Target Month",
            scale=alt.Scale(
                domain=[
                    comparison["Date"].min().isoformat(),
                    comparison["Date"].max().isoformat()
                ]
            ),
            axis=alt.Axis(format="%b %Y", labelAngle=-35, tickCount=6)
        ),
        y=alt.Y(
            "Rate:Q",
            title="Unemployment Rate (%)",
            scale=alt.Scale(zero=False)
        ),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=[
                    "Actual Unemployment",
                    "AI Prediction",
                    "Baseline Prediction"
                ],
                range=["#FF9F1C", "#22C55E", "#60A5FA"]
            )
        ),
        tooltip=[
            alt.Tooltip("Date:T", title="Month", format="%B %Y"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Rate:Q", title="Rate (%)", format=".2f")
        ]
    )
    .properties(height=400)
)

st.altair_chart(backtest_chart, use_container_width=True)

st.caption(
    "Historical six-month-ahead forecasts compared "
    "with actual unemployment rates."
)



st.subheader("Model Accuracy Comparison")

results = pd.DataFrame({
    "Model": ["Traditional Baseline", "Random Forest AI"],
    "Average Error": [baseline_error, ai_error]
})

st.bar_chart(results.set_index("Model"))

st.write(
    f"Traditional baseline error: "
    f"{baseline_error:.2f} percentage points"
)

st.write(
    f"AI forecast error: "
    f"{ai_error:.2f} percentage points"
)

if ai_error < baseline_error:
    st.success("AI outperformed the baseline on this test.")
else:
    st.warning(
        "The baseline outperformed the AI on this test."
    )

st.caption(
    "Source: Statistics Canada. "
    "This prototype uses historical data and "
    "machine learning for experimental forecasting."
)
# VERSION 4: Walk-forward backtesting results

st.divider()
st.header("Historical Forecast Performance")

st.write(
    "Explore how MacroSignal AI performed when "
    "predicting historical unemployment rates."
)

try:
    backtest = pd.read_csv("backtest_results.csv")

    backtest["Date"] = pd.to_datetime(backtest["Date"])

    backtest = backtest.sort_values("Date")

    st.subheader("AI vs Actual Unemployment")

    st.line_chart(
        backtest,
        x="Date",
        y=[
            "Actual",
            "AI Prediction",
            "Baseline Prediction"
        ]
    )

    st.subheader("Walk-Forward Backtest Accuracy")

    ai_mae = mean_absolute_error(
        backtest["Actual"],
        backtest["AI Prediction"]
    )

    baseline_mae = mean_absolute_error(
        backtest["Actual"],
        backtest["Baseline Prediction"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "AI Average Error",
            f"{ai_mae:.2f} pp"
        )

    with col2:
        st.metric(
            "Baseline Average Error",
            f"{baseline_mae:.2f} pp"
        )

    if ai_mae < baseline_mae:
        st.success("AI performed better in this backtest.")
    else:
        st.info(
            "The traditional baseline performed "
            "better in this backtest."
        )

    st.caption(
        "Historical walk-forward evaluation. "
        "Lower average error is better. "
        "Results are experimental."
    )

except FileNotFoundError:
    st.warning(
        "Backtest results not found. "
        "Run python backtest.py first."
    )


# PROJECT INFORMATION SIDEBAR

with st.sidebar:
    st.title("MacroSignal AI")
    st.caption("Economic Forecasting Platform")

    st.divider()

    st.subheader("About the Project")

    st.write(
        "MacroSignal AI is an experimental "
        "machine-learning platform for forecasting "
        "Canadian unemployment."
    )

    st.subheader("Technology")
    st.write("Python")
    st.write("Scikit-learn")
    st.write("Random Forest Regression")
    st.write("Streamlit")
    st.write("Statistics Canada")

    st.divider()

    st.subheader("Model Evaluation")

    st.info(
        "The current Random Forest model has not "
        "outperformed the traditional baseline "
        "in historical testing."
    )

    st.caption(
        "Forecasts are experimental and should "
        "not be used as financial advice."
    )
