"""
Make the themes of the Studio, write them to web/themes.json, and report
their audit (python -m palette, in apps/studio; -v: every finding).
"""

import json
import pathlib
import sys

from .audit import audit
from .themes import CHOICES, build, maps


PATH = pathlib.Path(__file__).parents[1] / 'web' / 'themes.json'


def main() -> int:
    themes = [build(choices) for choices in CHOICES]
    PATH.write_text(json.dumps(dict(themes=themes, maps=maps()), indent=2) + '\n')
    findings = [finding for theme in themes for finding in audit(theme)]
    failed = [finding for finding in findings if not finding.passed]
    for finding in findings if '-v' in sys.argv else failed:
        print(('' if finding.passed else 'FAILED ') + str(finding))
    print(f'{len(themes)} themes written to {PATH}: {len(failed)} of '
          f'{len(findings)} findings below their floors')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
