# -*- coding: utf-8 -*-
# @Time    : 9/8/2022 2:26 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: indicators.pyx

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

from util import shift
import numpy as np
cimport numpy as np
# from libc.math cimport sqrt
# from libc.stdlib cimport malloc, free
# from cpython cimport array


cpdef np.ndarray[np.float64_t, ndim= 1] SMA(double[:] closes, int period):
    """
    Simple Moving Average function
    @param closes: list of closing candle prices
    @param period: period to calculate for
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim= 1] result = np.zeros(length - period + 1,
                                                              dtype=np.float64)
    cdef double total
    cdef int i
    for i in range(period - 1, length):
        if i == period - 1:
            total = 0
            for j in range(i - period + 1, i + 1):
                total += closes[j]
        else:
            total += closes[i]
            total -= closes[i - period]
        result[i - period + 1] = total / period
    return result

cpdef np.ndarray[np.float64_t, ndim= 1] EMA(double[:] closes, int period):
    """
    Exponential Moving Average function
    Ref1: https://github.com/peerchemist/finta/blob/master/finta/finta.py
    Ref2: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    @param closes: list of closing candle prices
    @param period: period to calculate for
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim= 1] result = np.zeros(length,
                                                              dtype=np.float64)
    cdef double alpha = 2. / (period + 1)
    cdef double w = 1. - alpha
    cdef double f1 = 1
    cdef double f2 = 1 + w
    cdef int i
    for i in range(0, length):
        if i == 0:
            result[i] = closes[0]
        else:
            result[i] = (closes[i] + w * f1 * result[i - 1]) / f2
            f1 += w**i
            f2 += w**(i + 1)
    return result

cpdef np.ndarray[np.float64_t, ndim= 2] MACD(
        double[:] closes,
        int period_fast=12,
        int period_slow=26,
        int signal=9):
    """
    MACD, MACD Signal and MACD difference.

    The MACD Line oscillates above and below the zero line, which is also known
    as the centerline.
    These crossovers signal that the 12-day EMA has crossed the 26-day EMA. The
    direction, of course, depends on the direction of the moving average cross.
    Positive MACD indicates that the 12-day EMA is above the 26-day EMA.
    Positive values increase as the shorter EMA diverges further from the longer
    EMA.
    This means upside momentum is increasing. Negative MACD values indicates
    that the 12-day EMA is below the 26-day EMA.
    Negative values increase as the shorter EMA diverges further below the
    longer EMA. This means downside momentum is increasing.

    Signal line crossovers are the most common MACD signals. The signal line is
    a 9-day EMA of the MACD Line.
    As a moving average of the indicator, it trails the MACD and makes it easier
    to spot MACD turns.
    A bullish crossover occurs when the MACD turns up and crosses above the
    signal line.
    A bearish crossover occurs when the MACD turns down and crosses below the
    signal line.

    :param closes: np.array
    :param period_fast: int
    :param period_slow: int
    :param signal: int
    :param Returns: np.ndarray
                    - col1: MACD
                    - col2: SIGNAL
    """
    cdef np.ndarray[np.float64_t, ndim= 2] result
    cdef np.ndarray[np.float64_t, ndim= 1] EMA_fast = EMA(closes, period_fast)
    cdef np.ndarray[np.float64_t, ndim= 1] EMA_slow = EMA(closes, period_slow)
    cdef np.ndarray[np.float64_t, ndim= 1] MACD = EMA_fast - EMA_slow
    cdef np.ndarray[np.float64_t, ndim= 1] MACD_signal = EMA(MACD, signal)
    result = np.concatenate(
        (
            MACD[:, None],
            MACD_signal[:, None]
        ), axis=1
    )
    return result

cpdef np.ndarray[np.float64_t, ndim= 1] TR(
        double[:] highs,
        double[:] lows,
        double[:] closes):
    """
    True Range is the maximum of three price ranges:
    1. Most recent period's high minus the most recent period's low.
    2. Absolute value of the most recent period's high minus the previous close.
    3. Absolute value of the most recent period's low minus the previous close.

    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :return: np.array
    """
    highs_arr = np.asarray(highs)
    lows_arr = np.asarray(lows)
    closes_arr = np.asarray(closes)
    # True Range1 = High less Low
    TR1 = np.abs(highs_arr - lows_arr)
    # True Range2 = High less Previous Close
    TR2 = np.abs(highs_arr - shift(closes_arr, 1))
    # True Range3 = Previous Close less Low
    TR3 = np.abs(shift(closes_arr, 1) - lows_arr)
    return np.nanmax(np.array([TR1, TR2, TR3]), axis=0)

cpdef np.ndarray[np.float64_t, ndim= 1] ATR(
        double[:] highs,
        double[:] lows,
        double[:] closes,
        int period=14):
    """
    Average True Range is moving average of True Range.

    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :param period: int
    :return: np.array
    """
    cdef np.ndarray[np.float64_t, ndim= 1] TR_ = TR(highs, lows, closes)
    return SMA(TR_, period=period)
