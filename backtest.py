
import pandas as pd
import requests
import zipfile
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# STEP 1: Download Canada's unemployment data
print("Downloading economic data...")

url = (
    "https://www150.statcan.gc.ca/n1/tbl/csv/"
    "14100287-eng.zip"
)

response = requests.get(url, timeout=120)
response.raise_for_status()

if not zipfile.is_zipfile(BytesIO(response.content)):
    raise ValueError("Download was not a valid ZIP file.")

with zipfile.ZipFile(BytesIO(response.content)) as z:
    with z.open("14100287.csv") as file:
        data = pd.read_csv(file, low_memory=False)

# STEP 2: Select Canadian unemployment rates
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
df = df.reset_index(drop=True)

# STEP 3: Create historical features
for lag in range(12):
    df[f"lag_{lag}"] = df["VALUE"].shift(lag)

df["future_rate"] = df["VALUE"].shift(-6)

features = [f"lag_{lag}" for lag in range(12)]
labeled = df.dropna().reset_index(drop=True)

# STEP 4: Walk-forward backtesting
results = []

# Evaluate up to the latest 60 historical months
start = max(72, len(labeled) - 60)

if len(labeled) <= start:
    raise ValueError("Not enough historical observations.")

print("Testing historical AI predictions...")

for i in range(start, len(labeled)):

    # A training target must already be known
    # at the forecast origin
    train = labeled.iloc[:i - 5]
    test = labeled.iloc[[i]]

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["future_rate"]
    )

    ai_prediction = model.predict(
        test[features]
    )[0]

    actual = test["future_rate"].iloc[0]
    baseline = test["lag_0"].iloc[0]

    results.append({
        "Date": test["REF_DATE"].iloc[0]
                + pd.DateOffset(months=6),
        "Actual": actual,
        "AI Prediction": ai_prediction,
        "Baseline Prediction": baseline
    })

# STEP 5: Measure accuracy
results = pd.DataFrame(results)

ai_error = mean_absolute_error(
    results["Actual"],
    results["AI Prediction"]
)

baseline_error = mean_absolute_error(
    results["Actual"],
    results["Baseline Prediction"]
)

print("\nMACROSIGNAL AI — BACKTEST RESULTS")
print("--------------------------------")
print(f"AI error: {ai_error:.2f} percentage points")
print(f"Baseline error: {baseline_error:.2f} percentage points")

if ai_error < baseline_error:
    print("AI performed better in this backtest.")
else:
    print("Baseline performed better in this backtest.")

# Save results for the Streamlit dashboard
results.to_csv("backtest_results.csv", index=False)

print("\nResults saved to backtest_results.csv")
