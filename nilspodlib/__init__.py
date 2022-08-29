# -*- coding: utf-8 -*-
"""A python package to parse logged NilsPod Binary files.

@author: Arne Küderle, Nils Roth
"""
from .dataset import Dataset  # noqa: F401
from .session import Session, SyncedSession  # noqa: F401

__all__ = ["Dataset", "Session", "SyncedSession"]
__version__ = "3.3.0"
