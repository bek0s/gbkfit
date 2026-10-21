gbkfit-studio
=============

The Studio of GBKFIT: a page, served by gbkfit-server, to explore the
projects of a folder, edit their pieces and setups, preview a setup live
(its models as its parameters change), run evaluations and fits and
follow them, and see the products of their runs.

Run it with ``gbkfit-studio [FOLDER]``: it starts the server and opens
the page in the browser.

The page is written in TypeScript (React, Vite) in ``web/``; users never
need Node.js, as the package ships the built page. To build it:
``npm ci`` and ``npm run build`` (into ``src/gbkfit_studio/page``); to
develop it, ``npm run dev`` with a server on port 8765. The types of the
API come from the contract of the server (``npm run api``).
