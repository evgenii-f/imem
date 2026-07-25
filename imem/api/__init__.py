"""HTTP layer for iMem.

A thin FastAPI wrapper over the imem library (encoder + query + catalog),
serving the React frontend. Runs on the host (not in Docker) so it can use the
host GPU/MPS and read indexed image files at their real paths. See
``imem/api/app.py`` for the app factory and routes.
"""
