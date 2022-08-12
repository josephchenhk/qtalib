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
import pandas as pd
from finta import TA

import qtalib.indicators as ta


def float_format(number): return float("{:.2f}".format(number))


def array_equal(x, y): return np.allclose(x, y, rtol=1e-07, atol=1e-08)


test_data1 = dict(
    open=np.array([10.0, 15.0, 59.5, 32.0, 55.0]),
    high=np.array([15.0, 18.0, 69.0, 35.0, 55.0]),
    low=np.array([10.0, 12.0, 55.0, 29.5, 50.0]),
    close=np.array([12.0, 14, 64.0, 32.0, 53.0]),
    volume=np.array([1000, 500, 1200, 800, 2000])
)


class TestFunctions(unittest.TestCase):

    def testSMA(self):
        test_data = test_data1
        ohlc = pd.DataFrame(test_data)
        closes = test_data["close"]
        for i in range(1, len(closes) + 1):
            expected = TA.SMA(ohlc, i).dropna().values
            actual = ta.SMA(closes, i)
            res = array_equal(expected, actual)
            self.assertEqual(res, 1)

    def testEMA(self):
        test_data = test_data1
        ohlc = pd.DataFrame(test_data)
        closes = test_data["close"]
        for i in range(1, len(closes) + 1):
            expected = TA.EMA(ohlc, i).dropna().values
            actual = ta.EMA(closes, i)
            res = array_equal(expected, actual)
            self.assertEqual(res, 1)

    def testMACD(self):
        test_data = test_data1
        ohlc = pd.DataFrame(test_data)
        closes = test_data["close"]
        expected = TA.MACD(ohlc, period_fast=12, period_slow=26, signal=9)
        actual = ta.MACD(closes, period_fast=12, period_slow=26, signal=9)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testTR(self):
        test_data = test_data1
        ohlc = pd.DataFrame(test_data)
        highs = test_data["high"]
        lows = test_data["low"]
        closes = test_data["close"]
        expected = TA.TR(ohlc)
        actual = ta.TR(highs, lows, closes)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testATR(self):
        test_data = test_data1
        ohlc = pd.DataFrame(test_data)
        highs = test_data["high"]
        lows = test_data["low"]
        closes = test_data["close"]
        expected = TA.ATR(ohlc, period=3)
        actual = ta.ATR(highs, lows, closes, 3)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)
