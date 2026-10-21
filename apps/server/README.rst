gbkfit-server
=============

The server of GBKFIT: it serves the projects of a folder (see
``gbkfit.project``) over HTTP, to the Studio and other clients: their
files and runs, jobs (eval and fit, run by ``python -m gbkfit.tasks``),
live sessions (a model set up once and evaluated at the values of its
parameters as they change), and their events.

Run it with ``gbkfit-server [FOLDER]``; it prints the link to connect
with. Its API is versioned (``/api/v1``), and its contract is
``openapi.json`` (``python -m gbkfit_server.api > openapi.json``).

The code is in two layers: ``services`` (plain Python, no web framework)
and ``api`` (FastAPI routes, thin over the services).

Install it from this directory with ``pip install .`` (it needs gbkfit).
