"""Setup script for python packaging."""
import site
import sys

from setuptools import setup

# enable installing package for user
# https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="imgx_datasets",
    version="0.1.0",
    description="TFDS-based data set package.",
    author="Yunguan Fu",
    entry_points={},
)
