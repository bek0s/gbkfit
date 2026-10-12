"""Tests for the helpers of the tasks."""

import pytest

from gbkfit.tasks import _detail
from gbkfit.utils.parseutils import ConfigError


MODEL = dict(type='intensity_2d', components=[dict(
    type='smdisk', loose=False, tilted=False, rnmin=0, rnmax=4, rnsep=1,
    bptraits=dict(type='exponential'))])

HOST = dict(type='host')


def test_make_output_dir(tmp_path):
    path = tmp_path / 'out'
    assert _detail.make_output_dir(str(path), 'terminate') == str(path)
    assert path.is_dir()
    with pytest.raises(RuntimeError, match="already exists"):
        _detail.make_output_dir(str(path), 'terminate')
    assert _detail.make_output_dir(str(path), 'overwrite') == str(path)
    # Each unique directory is new
    first = _detail.make_output_dir(str(path), 'unique')
    second = _detail.make_output_dir(str(path), 'unique')
    assert first == str(tmp_path / 'out_1')
    assert second == str(tmp_path / 'out_2')
    (tmp_path / 'file').write_text('')
    with pytest.raises(RuntimeError, match="exists as a file"):
        _detail.make_output_dir(str(tmp_path / 'file'), 'overwrite')


def test_configs_resolve_relative_paths_against_their_root(tmp_path):
    # (their directory, or their section 'root', relative to it)
    config = tmp_path / 'configs' / 'fit.yaml'
    config.parent.mkdir()
    config.write_text("models: {}\nroot: ../data\n")
    cfg, root = _detail.read_config(str(config), ('models',))
    assert cfg == dict(models=[{}])
    assert root == str(tmp_path / 'data')
    config.write_text("models: {}\n")
    assert _detail.read_config(str(config), ('models',))[1] == \
        str(config.parent)
    with pytest.raises(ConfigError, match="needs the sections"):
        _detail.read_config(str(config), ('models', 'fitter'))


def observation(**options):
    """The configuration of an observation of MODEL, with options."""
    return dict(observable=dict(type='pixel_brightness', size=[8, 8])) \
        | options


def load_group(**sections):
    """The ObservationGroup of MODEL and the sections of a configuration."""
    return _detail.load_observation_group(dict(models=[MODEL]) | sections)


def test_observations_use_the_drivers_of_the_configuration():
    # One driver: every observation has it
    group = load_group(
        driver=HOST,
        observations=[observation(name='a'), observation(name='b')])
    a, b = [obs.driver() for obs in group.observations()]
    assert a is b
    # Several: each observation names its own
    group = load_group(
        drivers=dict(cpu1=HOST, cpu2=HOST),
        observations=[
            observation(name='a', driver='cpu2'),
            observation(name='b', driver='cpu1'),
            observation(name='c', driver='cpu2')])
    a, b, c = [obs.driver() for obs in group.observations()]
    assert a is c and a is not b
    # None: the host
    group = load_group(observations=[observation()])
    assert group.observations()[0].driver().type() == 'host'


@pytest.mark.parametrize('sections, message', [
    (dict(drivers=dict(cpu1=HOST, cpu2=HOST), observations=[observation()]),
     "must name its driver"),
    (dict(drivers=dict(cpu=HOST), observations=[observation(driver='gpu')]),
     "unknown driver 'gpu'"),
    (dict(driver=HOST, observations=[observation(driver='cpu')]),
     "unknown driver 'cpu'"),
    (dict(driver=HOST, drivers=dict(cpu=HOST), observations=[observation()]),
     "not both"),
    # (the form of configurations before drivers were declared at the top)
    (dict(observations=[observation(driver=HOST)]),
     "declares at its top level"),
])
def test_driver_errors(sections, message):
    with pytest.raises(ConfigError, match=message):
        load_group(**sections)
