from setuptools import setup, find_packages
import os

project_name = "amrvac_tools"

setup(
    name = project_name,
    keywords = "interface data-analysis",
    python_requires = ">=3.6",
    packages = find_packages(),
)