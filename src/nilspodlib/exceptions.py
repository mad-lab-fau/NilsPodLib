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


class SessionValidationError(Exception):
    """Error raised when the validation of a SyncedSession fails."""

    def __init__(self, msg: str, parent_cls, *args: object) -> None:
        msg += self._message(parent_cls)
        super().__init__(msg, *args)

    def _message(self, cls):
        return (
            "\n\nIf you think this check is incorrect and want to load the session anyway to further investigate or "
            f"manipulate the datasets, set "
            f"`{cls.__name__}.VALIDATE_ON_INIT = False`, before loading the "
            "Session. "
            "This will allow you to load the session, but many features might not work as expected! You are on your "
            "own!"
            "\nNote that this will affect ALL sessions you load, until you disable this option again."
        )


def datastream_does_not_exist_warning(sensor_type, operation):
    """Warn about not existing datastreams."""
    message = f'The datastream "{sensor_type}" does not exist for the current session. \
     The performed operation "{operation}" will have not effect'
    return warnings.warn(message)
