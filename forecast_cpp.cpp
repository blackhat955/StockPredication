#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>

namespace py = pybind11;

std::vector<double> recursive_forecast(
    const std::vector<double>& coefficients,
    double intercept,
    const std::vector<double>& initial_window,
    int steps
) {
    std::vector<double> predictions;
    predictions.reserve(steps);

    // Make a copy of the window to update it
    std::vector<double> current_window = initial_window;
    
    // Ensure dimensions match
    if (current_window.size() != coefficients.size()) {
        throw std::runtime_error("Window size must match number of coefficients");
    }
    
    size_t window_size = current_window.size();

    for (int i = 0; i < steps; ++i) {
        // Calculate dot product: y = w*x + b
        double pred = intercept;
        for (size_t j = 0; j < window_size; ++j) {
            pred += coefficients[j] * current_window[j];
        }
        
        predictions.push_back(pred);
        
        
        for (size_t j = window_size - 1; j > 0; --j) {
            current_window[j] = current_window[j-1];
        }
        current_window[0] = pred;
    }

    return predictions;
}

// Simple 1D Kalman Filter for Price Forecasting
// Using a Local Level Model (Random Walk + Noise) which is common for financial time series
// Q: Process Variance (how much the "true" price changes)
// R: Measurement Variance (how much noise is in the observation)
// data: historical data to train/filter on
// n_forecast: steps to forecast
std::vector<double> kalman_forecast(
    const std::vector<double>& data,
    int n_forecast,
    double Q = 1e-5, 
    double R = 1e-3
) {
    // Initial state
    double x_est = data[0];
    double P_est = 1.0;
    
    // Filter over history to get latest state
    for (double z : data) {
        // Predict
        double x_pred = x_est;
        double P_pred = P_est + Q;
        
        // Update
        double K = P_pred / (P_pred + R);
        x_est = x_pred + K * (z - x_pred);
        P_est = (1.0 - K) * P_pred;
    }
    
    // Forecast
    // For a random walk model, the best forecast is the last estimated state
    // (This is a simplified view; often HFT uses velocity/trend, but let's stick to robust 1D first)
    std::vector<double> predictions;
    predictions.reserve(n_forecast);
    for(int i=0; i<n_forecast; ++i) {
        predictions.push_back(x_est);
    }
    return predictions;
}

// Monte Carlo Simulation using Geometric Brownian Motion (GBM)
// A classic quantitative method for pricing and forecasting.
// data: historical price data
// n_forecast: steps to forecast
// n_sims: number of simulation paths to run (default 1000)
std::vector<double> monte_carlo_forecast(
    const std::vector<double>& data,
    int n_forecast,
    int n_sims = 1000
) {
    if (data.size() < 2) return std::vector<double>(n_forecast, data.back());

    // 1. Calculate Log Returns
    std::vector<double> log_returns;
    log_returns.reserve(data.size() - 1);
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i-1] > 0) {
            log_returns.push_back(std::log(data[i] / data[i-1]));
        } else {
            log_returns.push_back(0.0);
        }
    }

    // 2. Calculate Drift and Volatility
    double sum = std::accumulate(log_returns.begin(), log_returns.end(), 0.0);
    double mean = sum / log_returns.size();

    double sq_sum = std::inner_product(log_returns.begin(), log_returns.end(), log_returns.begin(), 0.0);
    double variance = (sq_sum / log_returns.size()) - (mean * mean);
    double std_dev = std::sqrt(variance);

    // Drift for GBM = mean + 0.5 * variance (commonly used) 
    // OR simpler: drift = mean - 0.5 * variance (Ito's Lemma correction)
    // Let's use standard GBM drift: mu - 0.5 * sigma^2
    double drift = mean - (0.5 * variance);

    // 3. Simulate Paths
    std::vector<double> avg_forecast(n_forecast, 0.0);
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1); // Standard normal N(0,1)

    double last_price = data.back();

    for (int s = 0; s < n_sims; ++s) {
        double current_price = last_price;
        for (int t = 0; t < n_forecast; ++t) {
            double shock = d(gen);
            // GBM Formula: P_t = P_{t-1} * exp(drift + sigma * shock)
            double growth = drift + std_dev * shock;
            current_price = current_price * std::exp(growth);
            
            // Accumulate for averaging
            avg_forecast[t] += current_price;
        }
    }

    // 4. Average the paths
    for (int t = 0; t < n_forecast; ++t) {
        avg_forecast[t] /= n_sims;
    }

    return avg_forecast;
}

PYBIND11_MODULE(forecast_cpp, m) {
    m.doc() = "C++ extension for recursive stock forecasting";
    m.def("recursive_forecast", &recursive_forecast, "Recursive forecast using linear model weights");
    m.def("kalman_forecast", &kalman_forecast, "Forecast using 1D Kalman Filter",
          py::arg("data"), py::arg("n_forecast"), py::arg("Q") = 1e-5, py::arg("R") = 1e-3);
    m.def("monte_carlo_forecast", &monte_carlo_forecast, "Forecast using Monte Carlo GBM",
          py::arg("data"), py::arg("n_forecast"), py::arg("n_sims") = 1000);
}
