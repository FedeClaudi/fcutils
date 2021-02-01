from pathlib import Path
import pytest
import os
from fcutils import path


@pytest.fixture
def fld():
    return str(Path(os.getcwd()))


@pytest.fixture
def fl():
    return Path(__file__)


def test_subdirs(fld):
    from_str = path.subdirs(fld)
    from_pth = path.subdirs(Path(fld))

    assert len(from_str) == len(from_pth)


def test_files(fld):
    from_str = path.files(fld)
    from_pth = path.files(Path(fld))

    assert len(from_str) == len(from_pth)


def test_get_size(fld, fl):
    path.size(fld)

    path.size(fl)
