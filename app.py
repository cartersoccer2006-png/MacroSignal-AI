
import streamlit as st
import pandas as pd
import requests
import zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="MacroSignal AI",
    page_icon="📈",
    layout="wide"
)


# MACROSIGNAL AI — PROFESSIONAL DASHBOARD DESIGN

st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

/* Main page spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1350px;
}

/* Headings */
h1, h2, h3 {
    color: #F8FAFC !important;
    font-family: Arial, sans-serif;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #151F32;
    border: 1px solid #26364D;
    border-radius: 12px;
    padding: 20px;
}

/* Metric numbers */
[data-testid="stMetricValue"] {
    color: #60A5FA;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Dividers */
hr {
    border-color: #26364D;
}

</style>
""", unsafe_allow_html=True)


st.title("📈 MacroSignal AI")
st.subheader("Canadian Economic Forecasting Platform")

st.write(
    "Using machine learning to forecast Canada's "
    "unemployment rate six months into the future."
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

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Unemployment Rate",
        f"{current_rate:.1f}%"
    )

with col2:
    st.metric(
        "AI Six-Month Forecast",
        f"{forecast:.2f}%"
    )

with col3:
    st.metric(
        "Forecast Target",
        forecast_date.strftime("%B %Y")
    )

st.caption(
    f"Latest observation: {latest_date:%B %Y}. "
    "Forecasts are estimates, not guarantees."
)


# Historical unemployment graph
st.subheader("Canadian Unemployment History")

history = unemployment.copy()

# Ensure dates are interpreted correctly
history["REF_DATE"] = pd.to_datetime(
    history["REF_DATE"],
    errors="coerce"
)

history = history.dropna(subset=["REF_DATE", "VALUE"])
history = history.sort_values("REF_DATE")

# Display recent historical data
history = history.tail(120)

st.line_chart(
    history,
    x="REF_DATE",
    y="VALUE",
    x_label="Date",
    y_label="Unemployment Rate (%)"
)



# Improved six-month forecast visualization
st.subheader("Six-Month AI Forecast")

# Get the latest 12 months of unemployment data
recent = unemployment.tail(12).copy()

# Create historical chart data
chart_data = recent.set_index("REF_DATE")[["VALUE"]]
chart_data.columns = ["Historical"]

# Add a separate AI forecast column
chart_data["AI Forecast"] = float("nan")

# Connect the forecast to the latest observation
chart_data.loc[latest_date, "AI Forecast"] = current_rate

# Add the six-month-ahead forecast
chart_data.loc[forecast_date, "AI Forecast"] = forecast

# Sort observations chronologically
chart_data = chart_data.sort_index()

# Display the updated graph
st.line_chart(
    chart_data,
    color=["#60A5FA", "#F59E0B"]
)

st.caption(
    "Historical unemployment rates and a six-month "
    "AI forecast. The orange line connects the current "
    "rate to the forecast; it does not represent "
    "monthly predictions."
)



# Compare historical AI predictions with actual outcomes
st.subheader("AI Predictions vs Actual Unemployment")

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

# Display interactive chart
st.line_chart(
    comparison,
    x="Date",
    y=[
        "Actual Unemployment",
        "AI Prediction",
        "Baseline Prediction"
    ],
    color=["#22C55E", "#60A5FA", "#F59E0B"]
)

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
