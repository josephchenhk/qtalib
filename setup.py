# -*- coding: utf-8 -*-
# @Time    : 10/8/2022 3:15 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: setup.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# from setuptools import setup
from setuptools import find_packages
from pathlib import Path

import numpy

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

ext_modules = [
    Extension("qtalib.indicators", ["qtalib/indicators.pyx"],
              include_dirs=[numpy.get_include()],
              language="c++"),
    Extension("qtalib.util", ["qtalib/util.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name='qtalib',
    version='0.0.2',
    keywords=('Quantitative Trading', 'Technical Analysis', 'QTaLib'),
    description='QTALIB: Quantitative Technical Analysis Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='JXW',
    install_requires=['finta',
                      'pandas',
                      'numpy',
                      'Cython'],
    author='josephchen',
    author_email='josephchenhk@gmail.com',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_package_data=True,
    packages=find_packages(),
    package_data={"qtalib": [
        "*.ico",
        "*.ini",
        "*.dll",
        "*.so",
        "*.pyd",
        "*.pyx"
    ]},
    platforms='any',
    url='',
    entry_points={
        'console_scripts': [
            'example=examples.demo:run'
        ]
    },
)