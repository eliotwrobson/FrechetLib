import frechetlib.data as fld
import pytest


@pytest.fixture
def frechet_downloader() -> fld.FrechetDownloader:
    return fld.FrechetDownloader()
