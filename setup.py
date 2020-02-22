import os
import sys
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'numpy',
    'tensorflow',
    'ConfigSpace',
]

setup(name='nas-tf2',
      version='0.0.1',
      description='',
      author='',
      author_email='',
      keywords='',
      packages=find_packages(),
      install_requires=requires,
    )
