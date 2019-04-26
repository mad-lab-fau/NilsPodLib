import pytest
from NilsPodLib.header import Header


def test_header_init_warning():
    with pytest.warns(UserWarning) as warn:
        Header(not_an_kwarg=None)

    assert len(warn) == 1
    assert 'not_an_kwarg' in warn[0].message.args[0]
