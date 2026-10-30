
# MacroSignal AI
### Canadian Economic Forecasting Platform

An interactive machine-learning application that forecasts
Canada's unemployment rate six months into the future
using historical economic data.

[Launch the Live Dashboard](https://macrosignal-ai-43hflvwughytk6cbktshtt.streamlit.app)

## Project Overview

MacroSignal AI explores whether machine-learning algorithms
can forecast changes in Canada's unemployment rate.

The application uses official Statistics Canada data,
Random Forest regression, and historical backtesting
to evaluate forecasting performance.

## Features

- Six-month-ahead unemployment forecasting
- Interactive economic data visualizations
- Historical forecast comparisons
- Walk-forward backtesting
- AI versus traditional baseline evaluation
- Publicly accessible Streamlit dashboard

## Technologies

- Python
- Pandas
- Scikit-learn
- NumPy
- Streamlit
- Statistics Canada datasets

## Machine Learning Approach

The Random Forest model uses unemployment observations
from the previous 12 months to forecast the rate
six months into the future.

A traditional persistence model, which assumes
unemployment remains unchanged, serves as a baseline.

## Backtesting Results

| Model | Mean Absolute Error |
|---|---|
| Random Forest AI | 2.14 percentage points |
| Traditional Baseline | 0.50 percentage points |

The traditional baseline outperformed the Random Forest
in this historical evaluation.

The results demonstrate the importance of comparing
machine-learning methods against simpler forecasting models.

## Limitations

- The model currently uses historical unemployment data only.
- Forecasts are experimental, not guaranteed predictions.
- Historical testing does not fully account for data revisions
  or publication timing.
- Further research is needed to improve accuracy and assess
  robustness across different economic conditions.

## Future Development

- Integrate inflation, interest rates, and GDP growth
- Evaluate additional machine-learning models
- Improve uncertainty estimation
- Develop more comprehensive historical testing

## Data Source

Statistics Canada — Labour Force Survey

## Project Status

Functional prototype deployed publicly.
Ongoing model evaluation and development.
