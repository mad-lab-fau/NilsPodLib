# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section), and 
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [3.0] - 

3.0 is the first version that supports imucal 2.0. and has many further changes to calibrations.
Most notatbly, factory calibrations are now applied automatically and regular calibrations are expected to be applied to
the already factory calibrated imu data.
**This means all your old calibrations files are invalid and need to be recreated**.

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
  This is due to the removal of the respective functions in `imucal`

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


