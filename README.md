# NilsPodLib

A python package to parse logged NilsPod Binary files.

## Installation

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib.git --upgrade
```

With ssh access:

```
pip install git+ssh://git@mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib.git --upgrade
```

For development:

```
git clone https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib.git
cd nilspodpythonlib
pip install -e . --upgrade
```

## Documentation

The documentation is available as SphinxDoc, which can be downloaded from the [pipelines page](https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib/-/jobs/artifacts/master/download?job=docs) for each commit.
The documentation is also available [online](http://lo94zeny.mad-pages.informatik.uni-erlangen.de/nilspodpythonlib/README.html) (only from the FAU university network).

Supplementary examples can be found in the [examples folder](https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib/tree/master/examples) of the git project.

## Legacy Support

The library aims to support the files recorded with the following NilsPod Firmware versions:

| Firmware      | Support           |
| ------------- |:------------------|
| 0.14.x        | full              |
| 0.13.x        | full *            |
| 0.12.x        | full *            |
| 0.11.255      | partial **        |
| 0.11.>2       | legacy support    |

\* The current version does not support the battery sensor available for versions <0.14.0.
However, to the best of our knowledge, this feature was never used.

\*\* 0.11.255 is the firmware version of older files converted to a compatible format.
They can be loaded as normal files, but certain header infos might not be supported.  

In case of *legacy support*, the library provides methods to convert the old fileformat to the new.
Here is an example for a session using the legacy 0.11.2 format.

```python
from NilsPodLib.legacy import convert_11_2
from NilsPodLib import Dataset

file_path = '...'  # Path to original file
new_file_path = '...'  # new path to converted file

convert_11_2(file_path, new_file_path)

ds = Dataset.from_bin_file(new_file_path)
```
 