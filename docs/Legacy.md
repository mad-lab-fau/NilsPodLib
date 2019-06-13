## Legacy Support

The library aims to support the files recorded with the following NilsPod Firmware versions:

| Firmware      | Support           |
| ------------- |:------------------|
| 0.14.x        | full              |
| 0.13.255      | partial **        |
| 0.13.x        | legacy support    |
| 0.12.x        | legacy support    |
| 0.11.>2       | legacy support    |


\*\* 0.13.255 is the firmware version of older files converted to a compatible format.
They can be loaded as normal files, but certain header infos might not be supported.
See the docstrings of the specific conversion functions for more detail on supported features and potential issues.

In case of *legacy support*, the library provides methods to convert the old fileformat to the new.
There are two ways to do this:

The easiest way (but with the least control), is to simply pass `legacy_support="resolve"` to any function that can load a Dataset or Session.
This will automatically pick a conversion function, apply it and then load you Dataset without touching your original file.

Here is an example for a session using the legacy 0.11.2 format:
```python
from NilsPodLib import Dataset

file_path = '...'  # Path to original file

ds = Dataset.from_bin_file(file_path, legacy_support="resolve")
print(ds.info.version_firmware)  # 0.13.255
```

If you do not want to convert your file every time, you can perform the conversion and save the result:
Here is an example for a session using the legacy 0.11.2 format.

```python
from NilsPodLib.legacy import convert_11_2
from NilsPodLib import Dataset

file_path = '...'  # Path to original file
new_file_path = '...'  # new path to converted file

convert_11_2(file_path, new_file_path)

ds = Dataset.from_bin_file(new_file_path)
print(ds.info.version_firmware)  # 0.13.255
```