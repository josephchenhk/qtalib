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
import qtalib.py_indicators as pta


def float_format(number):
    return float("{:.2f}".format(number))


def array_equal(x, y):
    return np.allclose(x, y, rtol=1e-07, atol=1e-08)


class TestFunctions(unittest.TestCase):

    def prepare_test_data(self, dataset: str):
        ohlcv = pd.read_csv(f"{dataset}.csv")
        self.ohlcv = ohlcv
        self.opens = ohlcv["open"].to_numpy()
        self.highs = ohlcv["high"].to_numpy()
        self.lows = ohlcv["low"].to_numpy()
        self.closes = ohlcv["close"].to_numpy()
        self.volumes = ohlcv["volume"].to_numpy()

    def testSMA(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        closes = self.closes
        for i in range(1, len(closes) + 1):
            expected = TA.SMA(ohlc, i).dropna().values
            actual = ta.SMA(closes, i)
            res = array_equal(expected, actual)
            self.assertEqual(res, 1)

    def testEMA(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        closes = self.closes
        for i in range(1, len(closes) + 1):
            expected = TA.EMA(ohlc, i).dropna().values
            actual = ta.EMA(closes, i)
            res = array_equal(expected, actual)
            self.assertEqual(res, 1)

    def testMACD(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        closes = self.closes
        expected = TA.MACD(
            ohlc,
            period_fast=12,
            period_slow=26,
            signal=9).dropna().values
        actual = ta.MACD(closes, period_fast=12, period_slow=26, signal=9)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testTR(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        highs = self.highs
        lows = self.lows
        closes = self.closes
        expected = TA.TR(ohlc).dropna().values
        actual = ta.TR(highs, lows, closes)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testATR(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        highs = self.highs
        lows = self.lows
        closes = self.closes
        expected = TA.ATR(ohlc, period=3).dropna().values
        actual = ta.ATR(highs, lows, closes, 3)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testSAR(self):
        self.prepare_test_data("test_data1")
        ohlc = self.ohlcv
        highs = self.highs
        lows = self.lows
        expected = TA.SAR(ohlc, 0.02, 0.2).dropna().values
        actual = ta.SAR(highs, lows, 0.02, 0.2)
        res = array_equal(expected, actual)
        self.assertEqual(res, 1)

    def testST(self):
        self.prepare_test_data("test_data2")
        highs = self.highs
        lows = self.lows
        closes = self.closes

        N = 30
        super_trend = {}
        py_super_trend = {}
        for i in range(N):
            py_super_trend = pta.ST(
                py_super_trend,
                highs[i:i + N],
                lows[i:i + N],
                closes[i:i + N],
                10,
                3.0,
                1,
                0
            )
            super_trend = ta.ST(
                super_trend,
                highs[i:i + N],
                lows[i:i + N],
                closes[i:i + N],
                10,
                3.0
            )
            super_trend_fmt = {k.decode("utf-8"): v for k, v in
                               super_trend.items()}
            expected = py_super_trend
            actual = super_trend_fmt
            self.assertEqual(expected, actual)

    def testTSV(self):
        self.prepare_test_data("test_data2")
        closes = self.closes
        volumes = self.volumes
        expected = pta.TSV(
            close=closes,
            volume=volumes,
            tsv_length=13,
            tsv_ma_length=7,
            tsv_bands_length=44,
            tsv_lookback=60,
            tsv_resample_interval=1,
            tsv_offset=0
        )
        actual = ta.TSV(
            closes,
            volumes,
            tsv_length=13,
            tsv_ma_length=7,
            tsv_bands_length=44,
            tsv_lookback=60
        )
        actual = {k.decode("utf-8"): v for k, v in actual.items()}
        self.assertEqual(expected, actual)
