import random

import pytest
import smart_simulation.consumer as cs


def test_decide(monkeypatch):
    def mock_random(*args, **kwargs):
        return 0.5

    monkeypatch.setattr(random, "random", mock_random)

    assert not cs.decide(0.1)
    assert cs.decide(0.9)

    with pytest.raises(ValueError):
        cs.decide(-1)
    with pytest.raises(ValueError):
        cs.decide(1.1)


def test_consume():
    test_random_function = random.randint
    test_function_parameters = (1, 1)

    assert cs.consume(test_random_function, test_function_parameters) == 1
    with pytest.raises(TypeError):
        cs.consume(test_random_function, None)
