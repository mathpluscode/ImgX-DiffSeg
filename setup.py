"""Setup script for python packaging."""
import site
import sys

from setuptools import setup

# enable installing package for user
# https://github.com/pypa/pip/issues/7953#issuecomment-645133255
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setup(
    name="imgx",
    version="0.1.0",
    description="",
    author="",
    entry_points={
        "console_scripts": [
            "imgx_train=imgx.run_train:main",
            "imgx_valid=imgx.run_valid:main",
            "imgx_test=imgx.run_test:main",
            "imgx_test_ensemble=imgx.run_test_ensemble:main",
        ],
    },
)
