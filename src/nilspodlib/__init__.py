"""A python package to parse logged NilsPod Binary files.

@author: Arne Küderle, Nils Roth
"""
from .dataset import Dataset
from .session import Session, SyncedSession

__all__ = ["Dataset", "Session", "SyncedSession"]
__version__ = "4.1.0"
