"""
Shared fixtures: a workspace with a project of a small model (an image
of a disk), whose setup names its pieces, with one free parameter; and,
for every test, a folder of the user of its own (not ~/.gbkfit).
"""

import pytest
import ruamel.yaml

import gbkfit.project


yaml = ruamel.yaml.YAML(typ='safe')

PIECES = {
    'galaxies/disk.yaml': dict(components=[dict(
        type='thin_disk', loose=False, tilted=False, rnodes=list(range(8)),
        rptraits=dict(type='exponential'),
        emission=dict(type='band'))]),
    'observations/image.yaml': dict(
        type='sky',
        observable=dict(type='pixel_brightness', size=[16, 12]),
        line_of_sight=dict(type='flat')),
    'params/disk.yaml': dict(properties=dict(
        xpos=0, ypos=0, posa=30, incl=dict(value=45, min=0, max=90),
        rpt_a=1, rpt_s=3)),
    'setups/disk.yaml': dict(
        driver=dict(type='host'), galaxies=['disk'], observations='image',
        params=['disk'])}


@pytest.fixture(autouse=True)
def user_home(tmp_path_factory, monkeypatch):
    """The folder of GBKFIT of the user (see GBKFIT_HOME), empty."""
    folder = tmp_path_factory.mktemp('user') / '.gbkfit'
    monkeypatch.setenv('GBKFIT_HOME', str(folder))
    return folder


@pytest.fixture
def root(tmp_path):
    """The folder of a workspace, with the project 'study'."""
    gbkfit.project.create(str(tmp_path / 'study'))
    for path, piece in PIECES.items():
        (tmp_path / 'study' / path).parent.mkdir(exist_ok=True)
        yaml.dump(piece, tmp_path / 'study' / path)
    return str(tmp_path)


@pytest.fixture
def setup():
    """The setup of the project, relative to the workspace."""
    return 'study/setups/disk.yaml'

