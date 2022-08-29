"""Exceptions and warnings used in the library."""
import warnings


class InvalidInputFileError(Exception):
    """Indicate an invalid binary file, which can not be loaded."""


class RepeatedCalibrationError(Exception):
    """Indicate that a datastream was already calibrated."""

    MESSAGE = 'The sensor "{0}" is already {1}calibrated. Repeated {1}calibration will lead to wrong values.'

    def __init__(self, sensor_name, factory):
        """Get a new Exception instance."""
        prefix = "factory-" if factory else ""
        message = self.MESSAGE.format(sensor_name, prefix)
        super().__init__(message)


class SynchronisationError(Exception):
    """Error that is raised for sync related issues."""


class SynchronisationWarning(Warning):
    """Indicate potential issues with sync."""


class LegacyWarning(Warning):
    """Indicate potential issues due to older firmware versions."""


class CorruptedPackageWarning(Warning):
    """Indicate potential issues with a recorded file."""


class VersionError(Exception):
    """Error related to Firmware Version issues."""


def datastream_does_not_exist_warning(sensor_type, operation):
    """Warn about not existing datastreams."""
    message = f'The datastream "{sensor_type}" does not exist for the current session. \
     The performed operation "{operation}" will have not effect'
    return warnings.warn(message)
