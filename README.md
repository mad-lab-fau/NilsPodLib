# NilsPodLib

[![PyPI](https://img.shields.io/pypi/v/nilspodlib)](https://pypi.org/project/nilspodlib/)
[![codecov](https://codecov.io/gh/mad-lab-fau/NilsPodLib/branch/master/graph/badge.svg?token=2CXLVYMHJF)](https://codecov.io/gh/mad-lab-fau/NilsPodLib)
![Test and Lint](https://github.com/mad-lab-fau/NilsPodLib/workflows/Test%20and%20Lint/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nilspodlib/badge/?version=latest)](https://nilspodlib.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nilspodlib)

A python package to parse logged NilsPod binary files.

## Installation

```
pip install nilspodlib --upgrade
```

If you have access to the mad-lab gitlab server, you can further install the `nilspodrefcal` repository, which contains
reference calibrations for a selected set of NilsPod sensors.
You can install it using:

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal.git --upgrade
```

## For users of NilsPodLib v1.0

With v2.0.0 the name of the library was updated from `NilsPodLib` to `nilspodlib` to comply with the recommended naming
style for Python packages.
Therefore, you need to update your import path, when updating to the new version!

## For developer

Install Python >=3.8 and [poetry](https://python-poetry.org).
Then run the commands below to get the latest source and install the dependencies:

```bash
git clone https://github.com/mad-lab-fau/NilsPodLib.git
cd nilspodlib
poetry install
```

To run any of the tools required for the development workflow, use the poe commands:

```
CONFIGURED TASKS
  format         
  lint           Lint all files with Prospector.
  check          Check all potential format and linting issues.
  test           Run Pytest with coverage.
  docs           Build the html docs using Sphinx.
  bump_version   
```

by calling

```bash
poetry run poe <command name>
````
