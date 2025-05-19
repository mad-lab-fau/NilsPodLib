# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section), and 
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.1] - 19.05.2025
## Internal Changes
- Migrated from `poetry` to `uv` for dependency management and packaging

## [4.0.0] - 16.04.2025
Removed support for Python 3.8
### Bugfixes
- Made `NilsPodLib` compatible with `numpy >= 2.0`
- Fixed testcase bugs that did not raise a `ValueError` when files with incorrect suffix were loaded.

## [3.6.0] - 17.10.2022

- Removed upper version bounds to avoid version conflicts during installation

## [3.5.0] - 31.08.2022

### Changed

- Synchronisation related error messages have been improved massively and now provide tips on how to debug and resolve
  common problems
- All sync related issues now raise a `SynchronisationError` instead of other generic errors.
  Note, that this breaks the workaround detailed [here](https://github.com/mad-lab-fau/NilsPodLib/issues/15) for one of
  the common issues. Instead of catching a `ValueError`, you should now catch a `SynchronisationError`.
- All Validation related errors now throw a `SessionValidationError`.

## [3.4.1] - 30.08.2022

### Bugfixes
- Fixed bug originated due to switching from `distutils.StrictVersion` to `packaging.version.Version`

## [3.4.0] - 29.08.2022

!!! Dropped Python 3.7 support !!!

### Changed

- All usages of `distuils.StrictVersion` are replaced with `packaging.version.Version`, as the former is deprecated.
- `Self` type is used whenever a method returns itself (or a copy). A bunch of other typing issues were fixed while 
  doing that

### Packaging

- Dropped Python 3.7 support
- Now committing lock file
- Switched to `poethepoet` from doit
- Updated readthedocs config to not require a seperate requirements.txt

## [3.3.0] - 29.08.2022

### Added

- added `force_version` parameter to `Session` to force specific firmware versions for dev firmwares. 

## [3.2.2] - 20.05.2021

### Changed

- Version checks within the legacy converter now ignore dev builds. If you are using a dev build, you are on your own!

## [3.2.1] - 20.05.2021

### Added

- Added new option to force overwrite the version considered by the legacy resolver for testing purposes.
  At the moment this is only supported for Datasets and not Sessions.

### Fixed

- Converting and saving old legacy files into the new 0.18 format now works correctly.
- The Ipython representation of the header now ignores fields that throw error.
  This fixes an issue that the representation could not be displayed when no timezone was specified.


## [3.2] - 20.04.2021

### Added
- Nilspodlib is now timezone aware!
  All loading methods for datasets, headers, and sessions now support a `tz` argument that is the string name of a valid
  timezone (e.g. `Europe/Berlin`).
  This allows access to the local start and end time in the header and a new local datetime index that can also be used 
  when exporting data to pandas dataframes.
  
### Changed
- The `utc_datetime_{start/end}` attributes of the header and the session are now properly reported in utc time instead 
  of as naive datetime objects.
- The `utc_datetime_counter` is now a pandas series. 

## [3.1] - 11.01.2021

### Added

- Support for new 0.18.0 firmware, which adds a 16 bit analog channel.
- Legacy support for all firmware versions <0.18.0

### Migration Guide

- Because all version <0.18.0 are now legacy, you need to use legacy support manually or `legacy="resolve"`, when 
  loading files with the firmware versions 0.16 and 0.17

## [3.0] - 09.01.2021

3.0 is the first version that supports imucal 2.0. and has many further changes to calibrations.
Most notatbly, factory calibrations are now applied automatically and regular calibrations are expected to be applied to
the already factory calibrated imu data.
**This means all your old calibrations files are invalid and need to be recreated**.
The NilspodRefCal repo is also updated accordingly.
This means, you need to update it as well.

### Added

### Changed

- The default acc unit of the factory calibration is now m/s^2 and the factory calibration is automatically applied when
  loading a file.
- The way the units and columns of a datastream is changed internally.
  This should not affect you, unless you manually modified Datastreams before in your code.
- We only support `imucal >= 3.0` starting with this release.  

### Deprecated

### Removed

- Removed all factory calibration methods, as the factory calibration is now applied automatically.
- The ability to calibrate just the acc or just the gryo.
  Instead, you can only apply calibrations to the entire IMU.
  This is due to the removal of the respective functions in `imucal`.
- Removed `cut_to_syncregion` that was deprecated for a while now.  

### Fixed

### Migration Guide

- The most important migration in this release is that all old calibrations files are invalid.
  You need to fully recreate them (or perform surgical unit conversion) of these files.
  Ideally, if you still have the raw calibration session and the annotated session list, follow the information in the
  calibration guide and `imucals` migration guide to recreate the calibrations in the most up to date format and the
  correct from units.
  If you don't have the old sessions available, you can try to use the information about the factory calibration to 
  modify the calibration sessions, so that they work on the factory calibrated instead of the raw data.
  However, this is not recommended.
  Maybe you are better of just performing a new calibration.


