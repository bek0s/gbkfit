"""
The palette of the Studio (web/themes.json), however it was made (by the
generator of palette.themes, or by hand), meets the floors of its audit
(palette.audit): every theme, in either scheme.
"""

import json
import pathlib

import pytest

from palette.audit import audit


THEMES = json.loads((pathlib.Path(__file__).parents[1] / 'web'
                     / 'themes.json').read_text())['themes']


@pytest.mark.parametrize('theme', THEMES, ids=[t['id'] for t in THEMES])
def test_the_theme_meets_its_floors(theme):
    failed = [str(finding) for finding in audit(theme) if not finding.passed]
    assert not failed, '\n'.join(failed)
