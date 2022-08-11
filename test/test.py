# -*- coding: utf-8 -*-
# @Time    : 10/8/2022 1:55 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: test.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""
import os
import pyximport

import numpy as np
import pandas as pd
from finta import TA

# Ref: https://github.com/cython/cython/issues/1725
numpy_path = np.get_include()
os.environ['CFLAGS'] = "-I" + numpy_path
pyximport.install(setup_args={"include_dirs":numpy_path})

import qtalib.indicators as ta


opens = np.array([10.0, 15.0, 59.5, 32.0, 55.0])
highs = np.array([15.0, 18.0, 69.0, 35.0, 55.0])
lows = np.array([10.0, 12.0, 55.0, 29.5, 50.0])
closes = np.array([12.0, 14, 64.0, 32.0, 53.0])
volumes = np.array([1000, 500, 1200, 800, 2000])

ohlc = pd.DataFrame({
    "open": opens,
    "high": highs,
    "low": lows,
    "close": closes,
})

print(TA.SMA(ohlc, 2))
print(ta.SMA(closes, 2))

print(TA.EMA(ohlc, 2))
print(ta.EMA(closes, 2))

print(TA.MACD(ohlc, period_fast=12, period_slow=26, signal=9))
print(ta.MACD(closes, period_fast=12, period_slow=26, signal=9))
print()