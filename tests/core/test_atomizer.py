from typing import Any
import pytest
from fact_reasoner.core.atomizer import Atomizer

def test_cblock():
    atm = Atomizer(None)
    assert str(atm) == "This is the atomizer"
