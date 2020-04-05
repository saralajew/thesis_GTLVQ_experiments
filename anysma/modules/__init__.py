from __future__ import absolute_import

from .transformation import *
from .measuring import *
from .routing import *
from .competition import *
from .final import *
from ..capsule import InputModule, OutputModule, Module, SplitModule  # make the modules visible


def globals_modules():
    globs = globals()
    from ..capsule import Capsule
    globs.update({'Capsule': Capsule})
    return globs
