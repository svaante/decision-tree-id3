from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='decision-tree-id3',
      version='0.1.2',
      description='A scikit-learn compatible package for id3 decision tree',
      author='Daniel Pettersson, Otto Nordander',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='svaante@gmail.com, otto.nordander@gmail.com',
      license='new BSD',
      url='https://github.com/svaante/decision-tree-id3'
      )
