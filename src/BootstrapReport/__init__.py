from .report import ObjectOfInterest
from .examples._generate_example_data import generate_example_dataset
from . import helpers
__version__ = "0.0.1"
__all__ = []

"""
    BootstrapReport module
    ============================
    Package structure
    -----------------
    .
    ├── __init__.py
    ├── examples -> Data calculations stored here
        ├── __init__.py
        ├── examples.py -> Examples file
    ├── BootstrapReport.py -> Main functions stored here
    ├── helpers.py -> Helper functions stored here
    ├── checkers.py -> Functions that check certain inputs and outputs stored here
"""
