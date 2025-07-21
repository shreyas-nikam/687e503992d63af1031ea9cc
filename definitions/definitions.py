import pandas as pd
import numpy as np

def generate_synthetic_ild(num_losses, mean_loss, std_loss, dist_type, frequency_rate):
    """Generates a DataFrame with synthetic ILD."""
    if num_losses <= 0:
        return pd.DataFrame({'Loss_Amount': [], 'Loss_Date': [], 'Event_Type': [], 'Business_Unit': []})

    if dist_type == 'lognormal':
        # Ensure mean_loss is positive for lognormal distribution
        mean_loss = abs(mean_loss)
        # Parameters for log-normal distribution
        mu = np.log(mean_loss**2 / np.sqrt(mean_loss**2 + std_loss**2))
        sigma = np.sqrt(np.log(1 + (std_loss**2 / mean_loss**2)))
        loss_amounts = np.random.lognormal(mu, sigma, num_losses)
    elif dist_type == 'pareto':
        # Pareto distribution parameters
        alpha = (mean_loss / std_loss)**2
        x_m = mean_loss / (alpha / (alpha - 1)) if alpha > 1 else mean_loss  #Scale parameter. Making sure alpha > 1 to avoid undefined mean
        loss_amounts = (np.random.pareto(alpha, num_losses) + 1) * x_m
    else:
        return None

    loss_amounts = np.round(loss_amounts, 2)

    # Generate dates
    start_date = pd.to_datetime('2020-01-01')
    loss_dates = start_date + pd.to_timedelta(np.random.randint(0, 365 * 3, num_losses), unit='D')

    # Generate event types and business units
    event_types = ['Cyber Attack', 'Natural Disaster', 'Operational Error', 'Fraud']
    business_units = ['Retail', 'Finance', 'Technology', 'Operations']
    event_type = np.random.choice(event_types, num_losses)
    business_unit = np.random.choice(business_units, num_losses)

    # Create DataFrame
    df = pd.DataFrame({
        'Loss_Amount': loss_amounts,
        'Loss_Date': loss_dates,
        'Event_Type': event_type,
        'Business_Unit': business_unit
    })

    return df

import numpy as np
from scipy.stats import lognorm, genpareto

def fit_distribution_to_data(data, distribution_type, fitting_method):
    """Fits a distribution to data and returns parameters and quantile function."""
    data = np.array(data)
    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    if distribution_type == 'lognormal':
        try:
            shape, loc, scale = lognorm.fit(data)
        except ValueError:
            data = np.array(data)
            shape, loc, scale = lognorm.fit(data[data > 0])

        quantile_func = lambda p: lognorm.ppf(p, shape, loc=loc, scale=scale)
        return (shape, loc, scale), quantile_func
    elif distribution_type == 'pareto':
        shape, loc, scale = genpareto.fit(data)
        quantile_func = lambda p: genpareto.ppf(p, shape, loc=loc, scale=scale)
        return (shape, loc, scale), quantile_func
    else:
        raise ValueError("Unsupported distribution type.")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, kstest

def fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='constant'):
    """Fits a distribution to scenario points using QLS."""

    if not scenario_points:
        raise ValueError("Scenario points cannot be empty.")

    if frequency_lambda <= 0:
        raise ValueError("Frequency lambda must be greater than zero.")

    for _, frequency in scenario_points:
        if frequency < 0:
            raise ValueError("Frequencies in scenario points cannot be negative.")
    
    thresholds = np.array([point[0] for point in scenario_points])
    frequencies = np.array([point[1] for point in scenario_points])

    # Define the objective function for quantile least squares
    def objective_function(params):
        try:
            # Calculate the theoretical frequencies from the distribution
            theoretical_frequencies = 1 - target_dist_type.cdf(thresholds, *params)

            # Calculate the weights based on the specified type
            if weights_type == 'constant':
                weights = np.ones_like(frequencies)
            elif weights_type == 'variance':
                weights = 1 / (frequencies + 1e-6)  # Adding a small constant to avoid division by zero
            else:
                raise ValueError("Invalid weights type. Choose 'constant' or 'variance'.")
            
            # Calculate the weighted squared differences
            weighted_squared_errors = weights * (frequencies - frequency_lambda * theoretical_frequencies) ** 2
            
            # Return the mean of the weighted squared errors
            return np.mean(weighted_squared_errors)

        except Exception as e:
            # Handle potential errors during frequency calculation
            return np.inf

    # Initial guess for the distribution parameters
    initial_guess = [1.0] * target_dist_type.numargs  # Example: shape, loc, scale for lognorm

    # Optimization using minimize
    result = minimize(objective_function, initial_guess, method='L-BFGS-B')

    # Extract the optimized parameters
    params = result.x

    # Create the quantile function
    def quantile_func(p):
        return target_dist_type.ppf(p, *params)

    return tuple(params), quantile_func

import numpy as np

def combine_distributions(ild_quantile_func, scenario_quantile_func, n_I, n_S, method):
    """Combines ILD and Scenario distributions using specified method."""
    if method == 'Quantile Averaging with Constant Weights':
        total = n_I + n_S
        weight_I = n_I / total
        weight_S = n_S / total

        def combined_quantile_func(q):
            return ild_quantile_func(q)**weight_I * scenario_quantile_func(q)**weight_S

        return combined_quantile_func
    elif method == 'Quantile Averaging with Non-constant Weights':
        raise NotImplementedError("Non-constant weights not implemented.")
    else:
        raise ValueError("Invalid method specified.")

import numpy as np

def simulate_aggregate_losses(frequency_dist_func, severity_dist_func, num_simulations):
    """Simulates aggregate losses using Monte Carlo simulation."""
    aggregate_losses = np.zeros(num_simulations)
    for i in range(num_simulations):
        num_losses = frequency_dist_func()
        total_loss = 0
        for _ in range(num_losses):
            total_loss += severity_dist_func()
        aggregate_losses[i] = total_loss
    return aggregate_losses

def calculate_VaR(loss_data, quantile_level):
                """Computes the Value-at-Risk (VaR) at a specified quantile from the loss data."""
                loss_data_sorted = sorted(loss_data)
                index = int(quantile_level * len(loss_data))
                return loss_data_sorted[index-1] if index > 0 else loss_data_sorted[0]