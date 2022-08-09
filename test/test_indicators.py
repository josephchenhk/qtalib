# -*- coding: utf-8 -*-
# @Time    : 9/8/2022 3:10 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: test_indicators.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
import unittest
import numpy as np

import qtalib.indicators as ta


values = np.array([12.0, 14.0, 64.0, 32.0, 53.0])
values1 = np.array([1., 2., 3., 4., 5.])

float_format = lambda number: float("{:.2f}".format(number))

class TestFunctions(unittest.TestCase):

    def testSMA(self):
        expected = np.array([13.,  39.,  48.,  42.5])
        actual = ta.SMA(values, 2)
        res = np.array_equal(expected, actual)
        self.assertEqual(res, 1)
