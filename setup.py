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
from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='qtalib',
    version='0.0.1',
    keywords=('Quantitative Trading', 'Technical Analysis', 'QTaLib'),
    description='QTALIB: Quantitative Technical Analysis Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='JXW',
    install_requires=['finta',
                      'pandas',
                      'numpy'],
    author='josephchen',
    author_email='josephchenhk@gmail.com',
    include_package_data=True,
    packages=find_packages(),
    # package_data={"": [
    # "*.ico",
    # "*.ini",
    # "*.dll",
    # "*.so",
    # "*.pyd",
    # ]},
    platforms='any',
    url='',
    entry_points={
        'console_scripts': [
            'example=examples.demo:run'
        ]
    },
)