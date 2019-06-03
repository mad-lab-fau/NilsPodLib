"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='NilsPodLib',
    version='0.1.0',
    description='A library to work with NilsPod data',

    packages=find_packages(exclude=('tests', 'examples', 'docs')),

    install_requires=[
        'scipy',
        'pandas',
        'numpy',
        'imucal @ git+https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/sensorcalibration.git'
    ],
    extras_require={
        'cal': [
            'NilsPodRefCal @ https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal.git',
        ]
    },
)
