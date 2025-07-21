import pytest
from definition_af8f994a01e34ab992433ae6498ae719 import generate_synthetic_ild
import pandas as pd
import numpy as np

def is_valid_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        return False
    expected_columns = ['Loss_Amount', 'Loss_Date', 'Event_Type', 'Business_Unit']
    if not all(col in df.columns for col in expected_columns):
        return False
    if not pd.api.types.is_numeric_dtype(df['Loss_Amount']):
        return False
    if not pd.api.types.is_datetime64_any_dtype(df['Loss_Date']):
        return False
    return True


@pytest.mark.parametrize("num_losses, mean_loss, std_loss, dist_type, frequency_rate, expected_type", [
    (100, 100000, 50000, 'lognormal', 10, pd.DataFrame),
    (0, 100000, 50000, 'lognormal', 10, pd.DataFrame),
    (50, 50000, 25000, 'pareto', 5, pd.DataFrame),
    (100, 100000, 50000, 'invalid_dist', 10, None), #Handles invalid dist_type gracefully.  If invalid the DataFrame will not pass the is_valid_dataframe test.
    (10, -100, 50, 'lognormal', 1, pd.DataFrame) # Handles negative mean_loss
])
def test_generate_synthetic_ild(num_losses, mean_loss, std_loss, dist_type, frequency_rate, expected_type):
    try:
        df = generate_synthetic_ild(num_losses, mean_loss, std_loss, dist_type, frequency_rate)
        if expected_type is None:
            assert df is None or not is_valid_dataframe(df)  # Either None or invalid DataFrame
        else:
            assert isinstance(df, expected_type)
            if isinstance(df, pd.DataFrame):
                assert is_valid_dataframe(df)
                if num_losses > 0:
                  assert len(df) == num_losses
    except Exception as e:
        if expected_type is Exception:
            assert isinstance(e, expected_type)
        else:
            raise e
