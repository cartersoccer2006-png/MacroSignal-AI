import pandas as pd
import requests
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt




# Download economic data from Statistics Canada
url = "https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip"

response = requests.get(url, timeout=120)
response.raise_for_status()

# Extract the CSV file
with zipfile.ZipFile(BytesIO(response.content)) as z:
    with z.open("14100287.csv") as file:
        data = pd.read_csv(file, low_memory=False)

# Select Canada's unemployment rate
unemployment = data[
    (data["GEO"] == "Canada")
    & (data["Labour force characteristics"] == "Unemployment rate")
    & (data["Gender"] == "Total - Gender")
    & (data["Age group"] == "15 years and over")
    & (data["Statistics"] == "Estimate")
    & (data["Data type"] == "Seasonally adjusted")
].copy()

# Organize the data by date
unemployment["REF_DATE"] = pd.to_datetime(
    unemployment["REF_DATE"]
)

unemployment = unemployment.sort_values("REF_DATE")

# Display the latest 12 months
print(unemployment[["REF_DATE", "VALUE"]].tail(12))

# Create a graph
plt.plot(unemployment["REF_DATE"], unemployment["VALUE"])

plt.title("Canadian Unemployment Rate")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)

plt.show()












from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Keep dates and unemployment rates
df = unemployment[["REF_DATE", "VALUE"]].dropna().copy()
df = df.sort_values("REF_DATE").reset_index(drop=True)

# Use unemployment from the previous 12 months
for lag in range(12):
    df[f"lag_{lag}"] = df["VALUE"].shift(lag)

# Target: unemployment rate 6 months ahead
df["future_rate"] = df["VALUE"].shift(-6)

df = df.dropna().reset_index(drop=True)

features = [f"lag_{lag}" for lag in range(12)]

# Train on the earliest 80% of the data
split = int(len(df) * 0.8)

# Leave a 6-month gap so training targets
# don't overlap with the test period
train = df.iloc[:split - 6]
test = df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(train[features], train["future_rate"])

# Predict historical test periods
predictions = model.predict(test[features])

error = mean_absolute_error(
    test["future_rate"],
    predictions
)

print(f"AI average forecast error: {error:.2f} percentage points")


# Traditional baseline: assume unemployment stays the same
baseline_predictions = test["lag_0"]

baseline_error = mean_absolute_error(
    test["future_rate"],
    baseline_predictions
)

print(f"Baseline forecast error: {baseline_error:.2f}")
print(f"AI forecast error: {error:.2f}")

if error < baseline_error:
    print("AI outperformed the baseline!")
else:
    print("The baseline outperformed the AI.")
