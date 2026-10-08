"""
Shared test configuration and fixtures.

Tests that use the `driver` fixture run once per driver (host and cuda).
A driver that is not available is skipped, unless it is listed in the
GBKFIT_REQUIRE_DRIVERS environment variable (e.g. "host" or "host,cuda"),
in which case its tests fail. This lets CI make sure the native modules
were actually built and are being tested.
"""

import os
import pathlib

import pytest


DATA_DIR = pathlib.Path(__file__).parent / 'data'

REQUIRED_DRIVERS = {
    name.strip()
    for name in os.environ.get('GBKFIT_REQUIRE_DRIVERS', '').split(',')
    if name.strip()}


def _create_driver(name):
    """Create the named driver. Raise an exception if it is unavailable."""
    if name == 'host':
        from gbkfit.driver.drivers.host import DriverHost
        return DriverHost()
    if name == 'cuda':
        import cupy
        from gbkfit.driver.drivers.cuda import DriverCuda
        if not cupy.cuda.is_available():
            raise RuntimeError("no CUDA device found")
        return DriverCuda()
    raise ValueError(f"unknown driver: {name}")


@pytest.fixture(
    scope='session',
    params=['host', pytest.param('cuda', marks=pytest.mark.cuda)])
def driver(request):
    """A gbkfit driver; tests using it run once for each driver."""
    name = request.param
    try:
        return _create_driver(name)
    except Exception as e:
        message = f"{name} driver not available: {e}"
        if name in REQUIRED_DRIVERS:
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """
    Run every test inside its own temporary directory, because parts of
    gbkfit write their output files to the current working directory.
    """
    monkeypatch.chdir(tmp_path)
