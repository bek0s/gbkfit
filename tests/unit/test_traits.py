"""
Tests for the validation of trait options.
"""

import pytest

from gbkfit.model.gmodels import traits


@pytest.mark.parametrize('angle', [-10, 190])
def test_axis_range_rejects_angles_outside_0_to_180(angle):
    with pytest.raises(RuntimeError, match="angle"):
        traits.WPTraitAxisRange(axis=0, angle=angle, weight=1)


def test_axis_range_accepts_angles_from_0_to_180():
    for angle in (0, 90, 180):
        traits.WPTraitAxisRange(axis=0, angle=angle, weight=1)
