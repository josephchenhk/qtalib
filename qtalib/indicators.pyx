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

import numpy as np
cimport numpy as np
from libcpp.map cimport map as cppmap
from libcpp.string cimport string
# from libcpp.vector cimport vector
# from libc.math cimport sqrt as csqrt
# from libc.stdlib cimport malloc, free
# from cpython cimport array
from qtalib.util import shift
from qtalib.util import ewm


cpdef np.ndarray[np.float64_t, ndim= 1] SMA(double[:] closes, int period):
    """
    Simple Moving Average function
    Note: this method skips nan values
    
    @param closes: np.array, list of closing candle prices
    @param period: int, period to calculate for
    @return _sma: np.array
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim= 1] result = np.zeros(length - period + 1,
                                                              dtype=np.float64)
    cdef double total
    cdef int i
    cdef int eff_period = 0
    for i in range(period - 1, length):
        if i == period - 1:
            total = 0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(closes[j]):
                    total += closes[j]
                    eff_period += 1

        else:
            if not np.isnan(closes[i]):
                total += closes[i]
                eff_period += 1
            if not np.isnan(closes[i - period]):
                total -= closes[i - period]
                eff_period -= 1
        if eff_period == 0:
            result[i - period + 1] = np.nan
        else:
            result[i - period + 1] = total / eff_period
    return result

cpdef np.ndarray[np.float64_t, ndim= 1] EMA(double[:] closes, int period):
    """
    Exponential Moving Average function
    Ref1: https://github.com/peerchemist/finta/blob/master/finta/finta.py
    Ref2: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    
    @param closes: np.array, list of closing candle prices
    @param period: int, period to calculate for
    @return _ema: np.array
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

cpdef np.ndarray[np.float64_t, ndim= 1] MSTD(double[:] closes, int period):
    """
    Moving Standard Deviation function
    Ref: https://www.zaner.com/3.0/education/technicalstudies/MSD.asp#:~:text=The%20moving%20standard%20deviation%20is,moving%20average%20of%20the%20prices.
    The moving standard deviation is a measure of market volatility. It makes no
    predictions of market direction, but it may serve as a confirming indicator. 
    You specify the number of periods to use, and the study computes the 
    standard deviation of prices from the moving average of the prices.
    Note: this method skips nan values

    @param closes: np.array, list of closing candle prices
    @param period: int, period to calculate for
    @return _mstd: np.array
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim= 1] _sma
    cdef np.ndarray[np.float64_t, ndim= 1] result = np.zeros(length - period + 1,
                                                              dtype=np.float64)
    cdef double total
    cdef int i
    cdef int eff_period = 0
    _sma = SMA(closes, period)
    for i in range(period - 1, length):
        if i == period - 1:
            total = 0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(closes[j]):
                    total += (closes[j] - _sma[i - period + 1])**2
                    eff_period += 1

        else:
            if not np.isnan(closes[i]):
                total += (closes[i] - _sma[i - period + 1])**2
                eff_period += 1
            if not np.isnan(closes[i - period]):
                total -= (closes[i - period] - _sma[i - period])**2
                eff_period -= 1
        if eff_period == 0:
            result[i - period + 1] = np.nan
        else:
            result[i - period + 1] = total / eff_period
    return np.sqrt(result)

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

cpdef np.ndarray[np.float64_t, ndim= 1] RSI(
    double[:] closes,
    int period = 14,
):
    """Relative Strength Index (RSI) is a momentum oscillator that measures the 
    speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, 
    RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and 
    centerline crossovers.
    RSI can also be used to identify the general trend."""

    ## get the price diff
    delta = np.diff(closes)

    ## positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # EMAs of ups and downs
    # _gain = ewm(up, window=period)
    # _loss = ewm(abs(down), window=period)
    cdef np.ndarray[np.float64_t, ndim= 1] _gain = np.empty_like(delta)
    cdef np.ndarray[np.float64_t, ndim= 1] _loss = np.empty_like(delta)
    cdef np.ndarray[np.float64_t, ndim= 1] RS = np.empty_like(delta)
    # FINTA uses alpha = 1/period; our implementaion of EMA(or ewm) uses
    # alpha = 2 / (1 + period). To align with their results, we pass in
    # window as (2 * period - 1) here:
    _gain = EMA(up, 2 * period - 1)
    _loss = EMA(abs(down), 2 * period - 1)
    # Avoid dividing by zeros
    RS = np.divide(
        _gain,
        _loss,
        out=np.inf * np.ones_like(_gain),
        where=_loss != 0
    )
    return 100 - (100 / (1 + RS))

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

cpdef np.ndarray[np.float64_t, ndim= 1] SAR(
        double[:] highs,
        double[:] lows,
        double af=0.02,
        double amax=0.2):
    """
    SAR stands for “stop and reverse,” which is the actual indicator used in 
    the system.
    SAR trails price as the trend extends over time. The indicator is below 
    prices when prices are rising and above prices when prices are falling.
    In this regard, the indicator stops and reverses when the price trend 
    reverses and breaks above or below the indicator.
    
    :param highs: np.array
    :param lows: np.array
    :param af: float
    :param amax: float
    :return: sar: np.array
    """
    cdef int length = highs.shape[0]
    # Starting values
    cdef double sig0, sig1, xpt0
    # next values
    cdef double xpt1, af0, af1
    # auxilary variables
    cdef double lmax, lmin
    # result: sar
    cdef np.ndarray[np.float64_t, ndim= 1] _sar = np.empty_like(highs)

    sig0 = 1
    xpt0 = highs[0]
    af0 = af
    highs_arr = np.asarray(highs)
    lows_arr = np.asarray(lows)
    _sar[0] = lows[0] - np.std(highs_arr - lows_arr, ddof=1)

    for i in range(1, length):
        sig1, xpt1, af1 = sig0, xpt0, af0
        lmin = min(lows[i - 1], lows[i])
        lmax = max(highs[i - 1], highs[i])
        if sig1:
            sig0 = lows[i] > _sar[i-1]
            xpt0 = max(lmax, xpt1)
        else:
            sig0 = highs[i] >= _sar[i-1]
            xpt0 = min(lmin, xpt1)
        if sig0 == sig1:
            sari = _sar[i-1] + (xpt1 - _sar[i-1]) * af1
            af0 = min(amax, af1 + af)
            if sig0:
                af0 = af0 if xpt0 > xpt1 else af1
                sari = min(sari, lmin)
            else:
                af0 = af0 if xpt0 < xpt1 else af1
                sari = max(sari, lmax)
        else:
            af0 = af
            sari = xpt0
        _sar[i] = sari
    return _sar

cpdef cppmap[string, double] ST(
        cppmap[string, double] super_trend,
        double[:] highs,
        double[:] lows,
        double[:] closes,
        int timeperiod=10,
        double band_multiple=3.0):
    """
    SuperTrend
    ATR channel, with mid/up/dn lines
        - close falls below dn, sell
        - close climbs above up, buy
    Ref1: https://cn.tradingview.com/script/r6dAP7yi/
    Ref2: https://zhuanlan.zhihu.com/p/138461317
    
    :param super_trend: Dict[str, float]
    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :param timeperiod: int
    :param band_multiple: float
    :return: super_trend: Dict[str, float]
    """
    highs_arr = np.asarray(highs)
    lows_arr = np.asarray(lows)
    closes_arr = np.asarray(closes)
    cdef double trend, dn, dn1, up, up1
    cdef np.ndarray[np.float64_t, ndim=1] mids = (highs_arr + lows_arr) * 0.5
    cdef np.ndarray[np.float64_t, ndim=1] _atr = ATR(
        highs=highs_arr,
        lows=lows_arr,
        closes=closes_arr,
        period=timeperiod
    )
    dn = mids[-1] - band_multiple * _atr[-1]
    up = mids[-1] + band_multiple * _atr[-1]
    if super_trend.size() > 0:
        trend = super_trend["trend"]
        dn1 = super_trend["dn"]
        up1 = super_trend["up"]
        if closes[-2] > dn1:
            dn = max(dn, dn1)
        if closes[-2] < up1:
            up = min(up, up1)
    else:
        n = len(closes_arr) / 3
        if closes_arr[:n].mean() < closes_arr[n:2 * n].mean() < closes_arr[2 * n:].mean():
            trend = 1.0
        else:
            trend = -1.0
        dn1 = -float("inf")
        up1 = float("inf")

    if trend == -1.0 and closes[-1] > up1:
        trend = 1.0
        dn = mids[-1] - band_multiple * _atr[-1]
        up = mids[-1] + band_multiple * _atr[-1]
    elif trend == 1.0 and closes[-1] < dn1:
        trend = -1.0
        dn = mids[-1] - band_multiple * _atr[-1]
        up = mids[-1] + band_multiple * _atr[-1]
    super_trend["mid"] = mids[-1]
    super_trend["ATR"] = _atr[-1]
    super_trend["up"] = up
    super_trend["dn"] = dn
    super_trend["trend"] = trend
    return super_trend


cpdef dict TSV(
        double[:] closes,
        long[:] volumes,
        int tsv_length=13,
        int tsv_ma_length=7,
        int tsv_lookback=60):
    """
    Time Segmented Volume
    Ref1: https://tw.tradingview.com/script/fmuLoK0d-time-segmented-volume-bands/?utm_source=amp-version&sp_amp_linker=1*zznimo*amp_id*YW1wLWVhdFVadXpBcjRIczRCMWpfN1l6VUE.
    
    :param closes: np.array
    :param volumes: np.array
    :param tsv_length: int
    :param tsv_ma_length: int
    :param tsv_bands_length: int
    :param tsv_lookback: int
    :return _tsv: Dict[str, float]
    """
    closes_arr = np.asarray(closes)
    volumes_arr = np.asarray(volumes)
    # LOGIC - TSV
    cdef np.ndarray[np.float64_t, ndim=1] t, m, tp, tn, tpna, tnna
    cdef np.ndarray[np.float64_t, ndim=1] inflow, outflow, difference, total
    cdef np.ndarray[np.float64_t, ndim=1] inflow_p, outflow_p, avg_inflow, avg_outflow
    cdef dict _tsv = {}
    t = np.diff(closes_arr) * volumes_arr[1:]
    t = np.convolve(t, np.ones(tsv_length, dtype=int), 'valid')
    m = SMA(t, tsv_ma_length)

    # # LOGIC - Inflow / outflow
    tp = t.copy()
    tp[tp <= 0] = 0
    tn = t.copy()
    tn[tn >= 0] = 0
    inflow = np.convolve(tp, np.ones(tsv_lookback, dtype=int),
                         'valid')
    outflow = np.convolve(tn, np.ones(tsv_lookback, dtype=int),
                          'valid') * -1
    difference = inflow - outflow
    total = inflow + outflow
    inflow_p = inflow / total * 100
    outflow_p = outflow / total * 100

    # LOGIC - AVG bands
    tpna = t.copy()
    tpna[tpna <= 0] = np.nan
    tnna = t.copy()
    tnna[tnna >= 0] = np.nan
    avg_inflow = SMA(tpna, tsv_lookback)
    avg_outflow = SMA(tnna, tsv_lookback)
    _tsv["t"] = t[-1]
    _tsv["m"] = m[-1]
    _tsv["avg_inflow"] = avg_inflow[-1]
    _tsv["avg_outflow"] = avg_outflow[-1]
    _tsv["difference"] = difference[-1]
    _tsv["inflow_p"] = inflow_p[-1]
    _tsv["outflow_p"] = outflow_p[-1]
    _tsv["t_ts"] = t[:]
    _tsv["m_ts"] = m[:]
    _tsv["avg_inflow_ts"] = avg_inflow[:]
    _tsv["avg_outflow_ts"] = avg_outflow[:]
    _tsv["difference_ts"] = difference[:]
    _tsv["inflow_p_ts"] = inflow_p[:]
    _tsv["outflow_p_ts"] = outflow_p[:]
    return _tsv
