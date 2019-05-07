# NilsPodLib

A python package to parse logged NilsPod Binary files.

## Installation

```
pip install git+https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/sensorcalibration.git --upgrade
```

With ssh access:

```
pip install git+ssh://git@mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/sensorcalibration.git --upgrade
```

For development:

```
git clone https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib.git
cd nilspodpythonlib
pip install -e . --upgrade
```

## Documentation

The documentation is available as SphinxDoc, which can be downloaded from the [pipelines page](https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib/-/jobs/artifacts/master/download?job=docs) for each commit.
The documentation is also available [online](http://lo94zeny.mad-pages.informatik.uni-erlangen.de/nilspodpythonlib/) (only from the FAU university network).

Supplementary examples can be found in the [examples folder](https://mad-srv.informatik.uni-erlangen.de/lo94zeny/nilspodpythonlib/tree/master/examples) of the git project.

## Legacy Support

The library aims to support the files recorded with the following NilsPod Firmware versions:

| Firmware      | Support          |
| ------------- |:----------------:|
| 0.13.x        | full             |
| 0.12.x        | full             |
| 0.11.3        | support planned  |
 