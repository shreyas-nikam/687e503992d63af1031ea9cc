import pytest
from definition_fe973832b51f485199b3a60bca6aa37b import combine_distributions
import numpy as np

def ild_quantile_func(q):
    return np.exp(q)

def scenario_quantile_func(q):
    return 2 * np.exp(q)


@pytest.mark.parametrize("n_I, n_S, method, expected_quantile", [
    (10, 20, 'Quantile Averaging with Constant Weights', lambda q: ild_quantile_func(q)**(10/30) * scenario_quantile_func(q)**(20/30)),
    (5, 5, 'Quantile Averaging with Constant Weights', lambda q: np.sqrt(ild_quantile_func(q) * scenario_quantile_func(q))),
    (10, 20, 'Quantile Averaging with Non-constant Weights', None),

])
def test_combine_distributions_constant_weights(n_I, n_S, method, expected_quantile):

    if method == 'Quantile Averaging with Constant Weights':
        combined_quantile_func = combine_distributions(ild_quantile_func, scenario_quantile_func, n_I, n_S, method)
        q = 0.5
        assert combined_quantile_func(q) == expected_quantile(q)
    else:
        with pytest.raises(NotImplementedError):
            combine_distributions(ild_quantile_func, scenario_quantile_func, n_I, n_S, method)
