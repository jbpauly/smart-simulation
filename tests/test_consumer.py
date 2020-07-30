import random

import pytest

from smart_simulation import consumer as cs


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
