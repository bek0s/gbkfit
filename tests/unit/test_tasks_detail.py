"""Tests for the helpers of the tasks."""

import pytest

from gbkfit.tasks import _detail


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
