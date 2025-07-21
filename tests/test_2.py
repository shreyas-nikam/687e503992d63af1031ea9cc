import pytest
import numpy as np
from definition_f2b401bd79b74196ad08287db14c5d38 import fit_scenario_to_distribution
from scipy.stats import lognorm

def test_fit_scenario_to_distribution_valid_input():
    """Test with valid scenario points and log-normal distribution."""
    scenario_points = [(1000, 0.1), (5000, 0.01)]
    frequency_lambda = 1.0
    target_dist_type = lognorm
    params, quantile_func = fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='constant')
    assert isinstance(params, tuple) or isinstance(params, list) # Assuming it returns parameters
    assert callable(quantile_func)

def test_fit_scenario_to_distribution_empty_scenario():
    """Test with an empty list of scenario points."""
    scenario_points = []
    frequency_lambda = 1.0
    target_dist_type = lognorm

    with pytest.raises(ValueError):
        fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='constant')

def test_fit_scenario_to_distribution_zero_frequency_lambda():
    """Test with a zero frequency lambda, should raise error"""
    scenario_points = [(1000, 0.1), (5000, 0.01)]
    frequency_lambda = 0.0
    target_dist_type = lognorm
    with pytest.raises(ValueError):
         fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='constant')

def test_fit_scenario_to_distribution_invalid_scenario_points():
    """Test with invalid scenario points (e.g., negative values)."""
    scenario_points = [(1000, -0.1), (5000, 0.01)]
    frequency_lambda = 1.0
    target_dist_type = lognorm
    with pytest.raises(ValueError):
        fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='constant')
def test_fit_scenario_to_distribution_different_weights():
    """Test that the different weights run and does not crash."""
    scenario_points = [(1000, 0.1), (5000, 0.01)]
    frequency_lambda = 1.0
    target_dist_type = lognorm
    params, quantile_func = fit_scenario_to_distribution(scenario_points, target_dist_type, frequency_lambda, weights_type='variance')
    assert isinstance(params, tuple) or isinstance(params, list) # Assuming it returns parameters
    assert callable(quantile_func)
