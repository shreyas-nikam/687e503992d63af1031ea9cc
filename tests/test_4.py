import pytest
from definition_e28b60fdae374e1eafa425d49b9ade1f import simulate_aggregate_losses
import numpy as np

def mock_frequency_dist_func():
    return np.random.poisson(5)

def mock_severity_dist_func():
    return np.random.exponential(100000)

@pytest.mark.parametrize("num_simulations, expected_length", [
    (10, 10),
    (100, 100),
    (0, 0),
])
def test_simulate_aggregate_losses_length(num_simulations, expected_length):
    result = simulate_aggregate_losses(mock_frequency_dist_func, mock_severity_dist_func, num_simulations)
    assert len(result) == expected_length

def test_simulate_aggregate_losses_returns_array():
    result = simulate_aggregate_losses(mock_frequency_dist_func, mock_severity_dist_func, 5)
    assert isinstance(result, np.ndarray)

def test_simulate_aggregate_losses_handles_zero_frequency():
    def zero_frequency_dist_func():
        return 0

    result = simulate_aggregate_losses(zero_frequency_dist_func, mock_severity_dist_func, 10)
    assert all(x == 0 for x in result)

def test_simulate_aggregate_losses_valid_input():
    try:
        simulate_aggregate_losses(mock_frequency_dist_func, mock_severity_dist_func, 10)
    except Exception as e:
        assert False, f"Unexpected exception: {e}"