import pathlib
import sqlite3

import pandas as pd

import pytest
import pytest_mock
from smart_simulation.cfg_templates.config import package_dir
from smart_simulation.database import db_utils as dbu


def test_connect(mocker):
    """
    Test the connect function from the db_utils module
    Args:
        mocker: pytest-mock object
    """
    dummy_db_path = pathlib.Path("/dummy.db")
    invalid_db_path = "not a path"
    invalid_db_path_ext = pathlib.Path("/dummy.csv")
    dummy_memory_connection = sqlite3.connect(":memory:")

    mocker.patch("sqlite3.connect", return_value=dummy_memory_connection)

    # Positive testing
    test_output = dbu.connect(dummy_db_path)
    assert isinstance(test_output, sqlite3.Connection)
    test_output.close()
    dummy_memory_connection.close()

    # Negative testing
    with pytest.raises(TypeError):
        assert dbu.connect(invalid_db_path)
    with pytest.raises(TypeError):
        assert dbu.connect(invalid_db_path_ext)


def test_weight_csv_to_db(mocker):
    """
    Test the weight_csv_to_db function from the db_utils module
    Args:
        mocker: pytest-mock object
    """
    test_file = pathlib.Path(package_dir) / "tests/test_components/test_weights.csv"
    test_file_df = pd.read_csv(test_file, parse_dates=True, index_col=0)
    valid_first_row_dt = str(test_file_df.index[0])
    valid_first_row_weight = test_file_df.weight[0]
    valid_num_rows = test_file_df.shape[0]
    valid_uuid = "00000001"
    invalid_uuid_type = 1
    invalid_uuid_len = "1"
    valid_first_row = (valid_first_row_dt, valid_uuid, valid_first_row_weight)
    first_row_query = "SELECT * FROM weights ORDER BY ROWID ASC LIMIT 1;"
    num_rows_query = "SELECT COUNT(*) FROM weights;"

    with sqlite3.connect(":memory:") as connection:
        mocker.patch(
            "smart_simulation.database.db_utils.connect", return_value=connection
        )
        c = connection.cursor()
        c.execute(
            """CREATE TABLE weights(
                date_time text,
                scale_id text,
                weight real)
                """
        )
        dbu.weight_csv_to_db(test_file, valid_uuid, connection)
        first_row_return = c.execute(first_row_query).fetchone()
        num_rows_return = c.execute(num_rows_query).fetchone()
        test_num_rows = num_rows_return[0]
        # Positive testing
        assert test_num_rows == valid_num_rows
        assert first_row_return == valid_first_row

    # Negative testing
    with pytest.raises(TypeError):
        assert dbu.weight_csv_to_db(test_file, invalid_uuid_type, "Dummy Connection")

    with pytest.raises(ValueError):
        assert dbu.weight_csv_to_db(test_file, invalid_uuid_len, "Dummy Connection")


def test_delete_all_rows(mocker):
    """
    Test the delete_all_rows function from the db_utils module
    Args:
        mocker: pytest-mock object
    """

    dummy_db_path = pathlib.Path("/dummy.db")
    valid_table = "weights"
    valid_sql_statement = f"DELETE FROM {valid_table};"
    invalid_table = ["weights"]

    class MockConnect:
        def __init__(self):
            self.sql_str = ""
            return

        def cursor(self):
            return self

        def execute(self, sql_statement, *args, **kwargs):
            self.sql_str = sql_statement

        def finalize(self):
            print("Finalizing the Class")

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.finalize()

        def __enter__(self):
            return self

    mock_connect = MockConnect()

    mocker.patch(
        "smart_simulation.database.db_utils.connect", return_value=mock_connect
    )

    dbu.delete_all_rows(valid_table, dummy_db_path)
    assert mock_connect.sql_str == valid_sql_statement

    # Negative testing
    with pytest.raises(TypeError):
        assert dbu.delete_all_rows(invalid_table)


# def test_delete_all_rows(monkeypatch):
#
#     dummy_db_path = pathlib.Path("/dummy.db")
#     valid_table = "weights"
#     valid_sql_statement = f"DELETE FROM {valid_table};"
#     invalid_table = ["weights"]
#
#     class MockCursor:
#         def __init__(self):
#             self.sql_str = ""
#             return
#
#         def cursor(self):
#             return self
#
#         def execute(self, sql_statement, *args, **kwargs):
#             self.sql_str = sql_statement
#
#         def finalize(self):
#             print("Finalizing the Class")
#
#         def __exit__(self, exc_type, exc_val, exc_tb):
#             self.finalize()
#
#         def __enter__(self):
#             return self
#
#     mock_cursor = MockCursor()
#
#     def patch_connect(*args, **kwargs):
#         return mock_cursor
#
#     monkeypatch.setattr(dbu, "connect", patch_connect)
#
#     dbu.delete_all_rows(valid_table, dummy_db_path)
#     assert mock_cursor.sql_str == valid_sql_statement
