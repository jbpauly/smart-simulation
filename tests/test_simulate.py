import pathlib

import pandas as pd

import pytest
import smart_simulation.simulate as sim


def test_write_output(tmpdir):
    """
    Test the write_output() function of the simulate module
    """
    test_path = tmpdir
    test_df = pd.DataFrame()
    test_file_name = "test_write_out"
    sim.write_output(test_df, test_path, test_file_name)
    file = pathlib.Path(test_path / "test_write_out.csv")

    # Positive testing
    assert file.is_file()
    if file.is_file():
        file.unlink()

    # Negative testing
    with pytest.raises(TypeError):
        assert sim.write_output(
            test_df="not_a_df",  # must be a Pandas DataFrame
            directory_path=test_path,
            file_name=test_file_name,
        )
