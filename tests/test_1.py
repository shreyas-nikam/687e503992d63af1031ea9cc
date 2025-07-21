import pytest
import numpy as np
from scipy.stats import lognorm, genpareto
from definition_6fec4f3be899436f95d777cfd900e656 import fit_distribution_to_data

def test_fit_distribution_to_data_lognormal():
    data = [1, 2, 3, 4, 5]
    params, _ = fit_distribution_to_data(data, 'lognormal', 'mle')
    assert isinstance(params, tuple)  # Check if parameters are returned
    assert len(params) == 3  # Check if the expected number of parameters is returned (shape, loc, scale)

def test_fit_distribution_to_data_pareto():
    data = [1, 2, 3, 4, 5]
    params, _ = fit_distribution_to_data(data, 'pareto', 'mle')
    assert isinstance(params, tuple)  # Check if parameters are returned
    assert len(params) == 3 # Check if the expected number of parameters is returned (shape, loc, scale)

def test_fit_distribution_to_data_empty_data():
    data = []
    with pytest.raises(ValueError):  # Expect a ValueError because of insufficient data
        fit_distribution_to_data(data, 'lognormal', 'mle')

def test_fit_distribution_to_data_invalid_distribution_type():
    data = [1, 2, 3]
    with pytest.raises(ValueError):  # Expect a ValueError because of unsupported distribution
        fit_distribution_to_data(data, 'invalid_distribution', 'mle')

def test_fit_distribution_to_data_zero_values_lognormal():
    data = [0, 1, 2, 3]
    params, _ = fit_distribution_to_data(data, 'lognormal', 'mle')
    assert isinstance(params, tuple)  # Check if parameters are returned
    assert len(params) == 3  # Check if the expected number of parameters is returned (shape, loc, scale)
