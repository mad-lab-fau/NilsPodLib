"""Exceptions and warnings used in the library."""
import warnings


class InvalidInputFileError(Exception):
    """Indicate an invalid binary file, which can not be loaded."""


class RepeatedCalibrationError(Exception):
    """Indicate that a datastream was already calibrated."""

    MESSAGE = 'The sensor "{}" is already calibrated. Repeated calibration will lead to wrong values.'

    def __init__(self, sensor_name):
        """Get a new Exception instance."""
        message = self.MESSAGE.format(sensor_name)
        super().__init__(message)


class SynchronisationError(Exception):
    """Error that is raised for sync related issues."""


class SynchronisationWarning(Warning):
    """Indicate potential issues with sync."""


class LegacyWarning(Warning):
    """Indicate potential issues due to older firmware versions."""


class CorruptedPackageWarning(Warning):
    """Indicate potential issues with a recorded file."""


class CalibrationWarning(Warning):
    """Indicate potential issues with a calibration."""


class VersionError(Exception):
    """Error related to Firmware Version issues."""


def datastream_does_not_exist_warning(sensor_type, operation):
    """Warn about not existing datastreams."""
    message = 'The datastream "{}" does not exist for the current session.\
     The performed operation "{}" will have not effect'.format(
        sensor_type, operation
    )
    return warnings.warn(message)
