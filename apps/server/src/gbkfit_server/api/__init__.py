"""
The API of the server (FastAPI): thin routes over its services (see
gbkfit_server.services), versioned (/api/v1), whose contract is
openapi.json (python -m gbkfit_server.api writes it).
"""

from .app import *
