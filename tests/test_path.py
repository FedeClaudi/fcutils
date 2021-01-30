from pathlib import Path
import pytest
import os
from fcutils import path


@pytest.fixture
def fld():
    return str(Path(os.getcwd()))


def test_subdirs(fld):
    from_str = path.subdirs(fld)
    from_pth = path.subdirs(Path(fld))

    assert len(from_str) == len(from_pth)
