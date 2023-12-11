"""Common configuration for scripts."""

from os import path


PROJ_DIR = path.abspath(path.join(path.dirname(__file__), '..'))
DATA_DIR = path.join(PROJ_DIR, 'data')
