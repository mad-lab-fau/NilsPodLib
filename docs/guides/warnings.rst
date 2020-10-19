========
Warnings
========

The NilspodLib uses various warnings to indicate potential issues with recorded files or processing steps.
While these are helpful and can prevent hour long head scratching, they can be distracting in particular, when large
amount of data is processed.

To ignore warnings, you can use Python’s ``warnings`` package:

>>> import warnings
>>> from nilspodlib.exceptions import LegacyWarning, CorruptedPackageWarning
>>> warnings.simplefilter('ignore', (LegacyWarning, CorruptedPackageWarning))

The example above will ignore all ``LegacyWarning`` and ``CorruptedPackageWarning``.
For a list of all NilsPodlib specific warnings view the ``NilsPodLib/exceptions.py`` file.
