"""Feature routers.

Every module in this package that exposes a FastAPI ``router`` is included
by the server at start-up. A feature adds a file here and nothing else has to
change — which is also why the modules must not import each other's state
except through :mod:`agrosuite.app.server`'s ``state`` object.
"""
