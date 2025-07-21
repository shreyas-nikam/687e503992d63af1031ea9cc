import pytest
from definition_6f6061bd18824a708763d096870e1581 import calculate_VaR
import numpy as np

@pytest.mark.parametrize("loss_data, quantile_level, expected", [
    ([1, 2, 3, 4, 5], 0.95, 5),
    ([10, 20, 30, 40, 50], 0.99, 50),
    ([100, 200, 300, 400, 500], 0.5, 300),
    ([100, 200, 300, 400, 500], 0, 100),
    ([5, 4, 3, 2, 1], 0.95, 5),
])
def test_calculate_VaR(loss_data, quantile_level, expected):
    assert calculate_VaR(loss_data, quantile_level) == expected
