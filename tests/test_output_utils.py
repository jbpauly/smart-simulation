import pathlib

import pytest
from smart_simulation.outputs import output_utils as ou


def test_weight_files(tmp_path):
    """
    Test the weight_files() function from the output_utils module
    Args:
        tmp_path: Temporary path for test purposes.
    """
    file_01 = tmp_path / "file_01.csv"
    file_02 = tmp_path / "file_02.csv"
    with file_01.open("w") as wf:
        wf.write("test")
    with file_02.open("w") as wf:
        wf.write("test")
    valid_output = [file_01, file_02]
    test_output = ou.weight_files(tmp_path)

    # Positive testing
    assert sorted(test_output) == sorted(valid_output)

    # Negative testing
    with pytest.raises(TypeError):
        assert ou.weight_files("not a path")


def test_servings_files(tmp_path):
    """
    Test the servings_files() function from the output_utils module
    Args:
        tmp_path: Temporary path for test purposes.
    """
    file_01 = tmp_path / "file_01.csv"
    file_02 = tmp_path / "file_02.csv"
    with file_01.open("w") as wf:
        wf.write("test")
    with file_02.open("w") as wf:
        wf.write("test")
    valid_output = [file_01, file_02]
    test_output = ou.servings_files(tmp_path)

    # Positive testing
    assert sorted(test_output) == sorted(valid_output)

    # Negative testing
    with pytest.raises(TypeError):
        assert ou.servings_files("not a path")


def test_file_uuid():
    """
    Test the file_uuid() function from the output_utils module
    """
    sample_file_name = "servings/0a3a90d1-9699-4866-ab61-7587e40b409d_daily.csv"
    invalid_file_name = "servings/90d1-9699-4866-ab61-7587e40b409d_daily.csv"
    sample_file_path = pathlib.Path(sample_file_name)
    invalid_file_path = pathlib.Path(invalid_file_name)
    valid_output = "0a3a90d1-9699-4866-ab61-7587e40b409d"
    test_output = ou.file_uuid(sample_file_path)

    # Positive testing
    assert test_output == valid_output

    # Negavtive testing
    with pytest.raises(TypeError):
        assert ou.file_uuid("not a path")
    with pytest.raises(ValueError):
        assert ou.file_uuid(invalid_file_path)


def test_truncate_uuid():
    """
    Test the truncate_uuid() function from the output_utils module
    """
    sample_uuid = "0a3a90d1-9699-4866-ab61-7587e40b409d"
    invalid_uuid = "0a3a90d1"
    valid_output = "0a3a90d1"
    test_output = ou.truncate_uuid(sample_uuid)

    # Positive testing
    assert test_output == valid_output

    # Negavtive testing
    with pytest.raises(ValueError):
        assert ou.truncate_uuid(invalid_uuid)
