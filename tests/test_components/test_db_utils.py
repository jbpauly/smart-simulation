import pathlib
import sqlite3

import pytest
from smart_simulation.database import db_utils as dbu


@pytest.fixture
def create_database():
    """ Fixture to set up the in-memory database with test data """
    connection = sqlite3.connect(":memory:")
    c = connection.cursor()
    c.execute(
        """CREATE TABLE weights(
            date_time text,
            scale_id text,
            weight real)
            """
    )
    sample_data = [
        ("2020-01-01 00:00:00", "01", 14.0),
        ("2020-01-01 01:00:00", "01", 14.0),
    ]
    c.executemany("INSERT INTO weights VALUES(?, ?, ?)", sample_data)
    return connection


def test_connect(monkeypatch):
    """
    Test the connect function from the db_utils module
    Args:
        monkeypatch: Patch used in test.
    """
    dummy_db_path = pathlib.Path("/dummy.db")
    invalid_db_path = "not a path"
    invalid_db_path_ext = pathlib.Path("/dummy.csv")
    dummy_memory_connection = sqlite3.connect(":memory:")

    def mock_sqlite_connect(*args, **kwargs):
        """
        Mock the connection to a database with the connection to an in memory database.
        """
        return dummy_memory_connection

    monkeypatch.setattr(sqlite3, "connect", mock_sqlite_connect)

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


# TODO write test_weight_csv_to_db
# def test_weight_csv_to_db():
#     # create dummy dataset
#     # add to test db


# TODO write test_delete_all_rows
# def test_delete_all_rows():
#    # delete rows from test database
#    # query for any existing rows
