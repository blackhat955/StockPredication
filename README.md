# Real-Time Stock Prediction Dashboard

This is a web application that visualizes stock market data and predicts future prices. I built it using Python (Dash) for the interface and added a C++ extension to make the forecasting calculations run faster.

## What it does

The app has three main sections:

1.  **Visualization**: You can pick a stock (like Apple, Amazon, Nvidia) and see its price history, volume, and gross profit over different time ranges.
2.  **Forecasting**: This predicts where the stock price might go next. The app currently uses:
    *   **ElasticNet**: A linear model that uses past prices to predict future ones (using C++ backend).
    *   **ARIMA**: A standard time-series model (simplified for speed).
    *   **Kalman Filter (HFT)**: A recursive state estimator commonly used in High-Frequency Trading (implemented in C++ for maximum speed).
    *   **Monte Carlo (HFT)**: Runs 1000 geometric brownian motion simulations to estimate the average future price path (implemented in C++).
3.  **Decomposition**: This breaks down the stock price into three parts: the overall trend, seasonal patterns, and random noise.

## Live Data Features
*   **Real-Time Data**: The app fetches live data from Yahoo Finance. You can enter **any stock ticker** (e.g., TSLA, NVDA) or cryptocurrency (e.g., BTC-USD) into the input box.
*   **Auto-Refresh**: The dashboard automatically updates every **60 seconds** to ensure you always have the latest price information without needing to reload the page.

## How to run it

You'll need Python installed. Here is how to get it running on your machine:

### 1. Install dependencies
First, grab all the necessary Python libraries.
```bash
pip install -r requirements.txt
```

### 2. Build the C++ extension
I moved the heavy number-crunching to C++ to speed things up. You need to compile it once before running the app:
```bash
python3 setup.py build_ext --inplace
```
*If you skip this, the app will still work, but it will fall back to the slower Python version.*

### 3. Start the app
Run the main script:
```bash
python3 app.py
```
Then open your browser and go to `http://127.0.0.1:8050/`.

## Model Performance
We compared different models to see which one predicted stock prices best (using Mean Absolute Error):

*   **ElasticNet**: 0.615 (Best Performer)
*   **LSTM**: 0.757
*   **Baseline Naive**: 0.996
*   **ARIMA**: 0.716

ElasticNet worked the best for our tests, which is why it's the main focus of this app.

## Project Team
*   **Luke Denoncourt** (Team Lead): LSTM, Regression, Model Predictions. - [GitHub](https://github.com/LukeD77)
*   **Michael Casey**: Time series decomposition and visualization. - [GitHub](https://github.com/mdcasey1983)
*   **Durgesh Tiwari**: ARIMA, Kalman Filter (HFT), Monte Carlo (HFT) implementations, and C++ optimization. - [GitHub](https://github.com/blackhat955)

## Tech Stack
*   **Python**: The main logic and UI (Dash, Plotly, Pandas, yfinance).
*   **C++**: Used for the recursive forecasting loop (pybind11).
*   **Dash Bootstrap Components**: For the dark-themed UI layout.
