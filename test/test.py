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
import qtalib.py_indicators as pta
import qtalib.indicators as ta
import os
import pyximport

import numpy as np
import pandas as pd
from finta import TA

# Ref: https://github.com/cython/cython/issues/1725
numpy_path = np.get_include()
os.environ['CFLAGS'] = "-I" + numpy_path
pyximport.install(setup_args={"include_dirs": numpy_path})


ohlcv = pd.read_csv("test/test_data2.csv")
opens = ohlcv["open"].to_numpy()
highs = ohlcv["high"].to_numpy()
lows = ohlcv["low"].to_numpy()
closes = ohlcv["close"].to_numpy()
volumes = ohlcv["volume"].to_numpy()

# print(TA.SMA(ohlc, 2))
# print(ta.SMA(closes, 2))
#
# print(TA.EMA(ohlc, 2))
# print(ta.EMA(closes, 2))
#
# print(TA.MACD(ohlc, period_fast=12, period_slow=26, signal=9))
# print(ta.MACD(closes, period_fast=12, period_slow=26, signal=9))

# print(TA.TR(ohlc))
# print(ta.TR(highs, lows, closes))

# print(TA.ATR(ohlc, 3))
# print(ta.ATR(highs, lows, closes, 3))

# print(TA.SAR(ohlc, 0.02, 0.2))
# print(ta.SAR(highs, lows, 0.02, 0.2))

# N = 10
# super_trend = {}
# py_super_trend = {}
# for i in range(N):
#     py_super_trend = pta.ST(
#         py_super_trend,
#         highs[i:i + N],
#         lows[i:i + N],
#         closes[i:i + N],
#         10,
#         3.0,
#         1,
#         0
#     )
#     super_trend = ta.ST(
#         super_trend,
#         highs[i:i + N],
#         lows[i:i + N],
#         closes[i:i + N],
#         10,
#         3.0
#     )
#     super_trend_fmt = {k.decode("utf-8"): v for k, v in super_trend.items()}
#     check = py_super_trend == super_trend_fmt
#     print(f"{i} {check}\n\t{py_super_trend}\n\t{super_trend_fmt}")

# expected = pta.TSV(
#     close=closes,
#     volume=volumes,
#     tsv_length=13,
#     tsv_ma_length=7,
#     tsv_bands_length=44,
#     tsv_lookback=60,
#     tsv_resample_interval=1,
#     tsv_offset=0
# )
# actual = ta.TSV(
#     closes,
#     volumes,
#     tsv_length=13,
#     tsv_ma_length=7,
#     tsv_bands_length=44,
#     tsv_lookback=60
# )

# print(np.allclose(
#     np.array(TA.RSI(ohlcv, 14).dropna()),
#     ta.RSI(closes, 14)
# ))

# from qtalib.util import ffill
# print(np.allclose(
#     ffill(np.array(TA.OBV(ohlcv)))[1:],
#     ta.OBV(closes, volumes)[1:]
# ))

cyc = 0
for n in range(50, 0, -1):
    cyc = ta.CYC(closes[-n-59:-n], cyc)
    print(cyc)


