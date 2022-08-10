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

import numpy as np
import pandas as pd
from finta import TA

import qtalib.indicators as ta

values = np.array([12.0, 14, 64.0, 32.0, 53.0])

print(ta.EMA(values, 2))

ohlc = pd.DataFrame({
    "open": np.zeros_like(values),
    "high": np.zeros_like(values),
    "low": np.zeros_like(values),
    "close": values,
})

print(TA.EMA(ohlc, 2))
print()
# print(ta.SMA(values, 4))
# print(ta.SMA(values, 3))
# print(ta.SMA(values, 2))
# print(ta.SMA(values, 1))

