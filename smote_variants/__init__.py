"""
This module brings together all functionalities.
"""

from ._version import __version__

__author__ = "György Kovács"
__license__ = "MIT"
__email__ = "gyuriofkovacs@gmail.com"

from .oversampling import *
from .queries import *
from .multiclassoversampling import *

from . import config
from . import base
from . import classifiers
from . import datasets
from . import evaluation
from . import multiclassoversampling
from . import queries
from . import visualization
