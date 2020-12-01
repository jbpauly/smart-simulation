import random

import pytest
import smart_simulation.consumer as cs
from smart_simulation.cfg_templates import customers as ct
from tests.test_components import test_cfg


def test_decide(monkeypatch):
    """
    Test the decide() function from the consumer module
    Args:
        monkeypatch: patch for necessary attributes
    """

    # mock the Python Random Class function random() used for random chance
    def mock_random():
        return 0.5

    monkeypatch.setattr(random, "random", mock_random)

    # Positive testing
    assert not cs.decide(probability=0.1)  # 0.5 < 0.1 should return false
    assert cs.decide(probability=0.9)  # 0.5 < 0.9 should return true

    # Negative testing
    with pytest.raises(ValueError):
        cs.decide(probability=-1)  # probability must be > 0
    with pytest.raises(ValueError):
        cs.decide(probability=1.1)  # probability must be < 1


def test_consume():
    """
    Test the consume() function from the consumer module
    """
    test_random_function = random.randint  # select function for consumption use
    test_function_parameters = (1, 1)  # parameters for randint, forcing a return of 1

    # Positive testing
    assert (
        cs.consume(
            random_function=test_random_function,
            function_params=test_function_parameters,
        )
        == 1
    )

    # Negative testing
    with pytest.raises(TypeError):
        cs.consume(
            random_function=test_random_function, function_params=None
        )  # function_params must be a tuple


def test_single_day(monkeypatch):
    """
    Test the single_day() function from the consumer module
    Args:
        monkeypatch: patch for necessary attributes
    """

    # Mock the decide() function from the consumers module
    def mock_decide(*args, **kwargs):
        return True  # force decision to always be true

    # Mock the consume() function from the consumers module
    def mock_consume(*args, **kwargs):
        return 1  # force consumption to always be 1

    monkeypatch.setattr(cs, "decide", mock_decide)
    monkeypatch.setattr(cs, "consume", mock_consume)

    customer = test_cfg.customers["test"]
    test_output = cs.single_day(customer_config=customer, day_of_week=0)

    # Positive testing
    assert test_output == 1
    # TODO validate the template items

    # Negative testing
    with pytest.raises(TypeError):
        cs.single_day(
            customer_config=customer, day_of_week="not_an_int"
        )  # must be an int
    with pytest.raises(ValueError):
        cs.single_day(customer_config=customer, day_of_week=7)  # must in range 0-6


def test_multi_day(monkeypatch):
    """
    Test the multi_day() function from the consumer module
    Args:
        monkeypatch: patch for necessary attributes
    """

    # Mock the get_customer() function from consumers module
    def mock_get_customer(*args, **kwargs):
        return test_cfg.customers["test"]

    # Mock the single_day() function from consumers module
    def mock_single_day(*args, **kwargs):
        return 1  # force consumption to always be 1

    monkeypatch.setattr(cs, "get_customer", mock_get_customer)
    monkeypatch.setattr(cs, "single_day", mock_single_day)

    test_output = cs.multi_day(customer_number="0", days=2)

    # Positive testing
    assert test_output["servings"].iloc[0] == 1
    assert test_output["servings"].iloc[1] == 1

    # Negative testing
    with pytest.raises(TypeError):
        assert cs.multi_day(customer_number=0, days=1)  # must be a string
    with pytest.raises(TypeError):
        assert cs.multi_day(customer_number="0", days="not_int")  # must be an int


def test_get_customer(monkeypatch):
    """
    Test the get_customer() function from the consumer module
    Args:
        monkeypatch: patch for necessary attributes
    """

    # Mock the customers configurations in the customers module
    def mock_customers():
        return test_cfg.customers

    monkeypatch.setattr(ct, "customers", mock_customers())

    # Positive testing
    valid_output = test_cfg.customers["test"]
    test_output = cs.get_customer(customer_number="test")
    assert valid_output == test_output

    # Negative testing
    with pytest.raises(Exception):
        assert cs.get_customer(customer_number="not_a_customer")
