# -*- coding: utf-8 -*-
# @Time    : 9/8/2022 3:01 pm
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

import numpy

ext_modules = [
    Extension("indicators", ["indicators.pyx"],
              include_dirs=[numpy.get_include()],
              language="c++"),
    Extension("util", ["util.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    name="Technical Indicators",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
