
import json
import subprocess
import sys

import pytest

from gbkfit import outputs
from gbkfit.model import Extras
from modelutils import group_plan


@pytest.fixture
def evaluate_config(tmp_path):
    """
    Return a function that evaluates a configuration file with the eval
    task of gbkfit (python -m gbkfit.tasks) and returns its run (see
    gbkfit.outputs). Each configuration is evaluated in its own directory.
    """
    def evaluate(config):
        workdir = tmp_path / config.stem
        workdir.mkdir()
        options = dict(
            mode='model', config=str(config), profile_iters=0,
            output_dir='output', output_dir_mode='terminate')
        result = subprocess.run(
            [sys.executable, '-m', 'gbkfit.tasks', 'eval',
             json.dumps(options)],
            cwd=workdir, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"eval failed:\n{result.stderr[-3000:]}")
        return outputs.open(workdir / 'output')
    return evaluate


@pytest.fixture
def evaluate_cases():
    """
    Return a function that evaluates cases of the tests (see
    split_case) through the Python API. It returns the model data of
    every observation (on the host) and the extra outputs (e.g. the
    velocity field of each component; see gbkfit.model.Extras).
    """
    import gbkfit.params

    def evaluate(cases, properties, modes=None):
        plan = group_plan(cases)
        group = plan.group()
        params = gbkfit.params.ParamSpace(
            group.pdescs(), properties, constants=group.constants(),
            modes=modes)
        extra = Extras()
        data = plan.model_h(params.evaluate(), extra)
        return data, extra
    return evaluate
