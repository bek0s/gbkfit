"""Print the schema of the API (its contract, openapi.json)."""

import json

from gbkfit_server.api import openapi

print(json.dumps(openapi(), indent=2))
