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

# cyc = 0
# for n in range(50, 0, -1):
#     cyc = ta.CYC(closes[-n-59:-n], cyc)
#     print(cyc)

exp_result = np.array(
    [0.        ,    70.89430894,   100.44864827,   139.1999938 ,
     400.1999938 ,   580.1999938 ,   678.9499938 ,   771.87195388,
     776.49007111,   912.49007111,  1102.45030967,  1102.45030967,
    1163.72198597,  1215.72198597,  1347.70168141,  1406.7925905 ,
    1544.09890754,  1574.90750529,  1620.37513119,  1699.68547602,
    1748.26118816,  1888.71062636,  1991.96062636,  2036.96062636,
    2327.96062636,  2475.51618192,  2624.86063961,  2683.4016455 ,
    2816.05689175,  3088.05689175,  3088.05689175,  3155.97284859,
    3399.97284859,  3433.35867536,  3501.68751633,  3536.39071268,
    3553.31045238,  3574.90872451,  3870.90872451,  3998.11894062,
    4094.20141485,  4164.57178522,  4269.57178522,  4388.50141852,
    4463.31057883,  4604.68173722,  4700.55475309,  4773.39637688,
    4855.85036461,  4897.67480523,  4970.05787872,  5127.39121206,
    5127.39121206,  5326.03297776,  5342.93438621,  5363.74627872,
    5420.60902382,  5473.60902382,  5582.9113494 ,  5639.89913773,
    5805.95657899,  5831.51789159,  6038.01441132,  6198.85925291,
    6397.85925291,  6448.8688172 ,  6448.8688172 ,  6490.47955546,
    6514.91137364,  6523.74328247,  6710.74328247,  6853.23814196,
    6871.64677312,  6929.11628531,  7100.41258161,  7188.54043549,
    7281.54043549,  7570.54043549,  7839.03358617,  8005.75022785,
    8172.09566892,  8223.98756081,  8245.17326264,  8357.30715386,
    8448.35518879,  8479.05139132,  8655.05139132,  8816.37403661,
    8919.42488407,  9029.04026869,  9228.04026869,  9384.48726009,
    9654.48726009,  9891.50978702, 10068.50978702, 10133.20698053,
   10209.92453778, 10386.73435373, 10386.73435373, 10456.39084228,
   10593.39084228, 10751.60409876, 10918.93083144, 11068.35611879,
   11181.77766889, 11427.77766889, 11449.35443237, 11455.8859639 ,
   11716.8859639 , 11797.8859639 , 11874.05956886, 12068.05956886,
   12141.25348521, 12313.25348521, 12531.60642639, 12738.60642639,
   12738.60642639, 12877.60642639, 12930.78150057, 13079.34196123,
   13101.28852611, 13101.28852611, 13227.28852611, 13268.81962659,
   13352.57172039, 13570.57172039, 13674.70965143, 13729.07011654,
   13771.22863256, 13780.16980903, 13836.46902163, 13864.01493148,
   14115.66970111, 14239.37510963, 14305.7359659 , 14305.7359659 ,
   14520.86870926, 14779.2748228 , 14845.4430471 , 14912.4430471 ,
   15010.4430471 , 15048.11617729, 15091.92805848, 15244.67754931,
   15408.36631012, 15533.29256381, 15621.1809595 , 15669.42039612,
   15828.63375818, 15936.54742725, 16000.54742725, 16039.33370693,
   16127.30621552, 16304.30621552, 16323.89884404, 16493.92942508,
   16792.92942508, 16810.69098226, 16823.21366828, 16952.22390719,
   17006.51943984, 17292.23372555, 17330.61474439, 17356.74377665,
   17435.41044332, 17569.76158835, 17607.76158835, 17656.74118019,
   17729.69670386, 17782.5348258 , 17803.71129639, 17860.71129639,
   17897.26482119, 17949.93905134, 17969.76309826, 17995.17518617,
   18074.83330583, 18121.76432158, 18307.57840003, 18423.34194683]
)
print(np.allclose(
    exp_result,
    ta.WOBV(opens, highs, lows, closes, volumes, cum_obv=0))
)


