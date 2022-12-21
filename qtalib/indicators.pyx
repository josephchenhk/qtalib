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


cpdef np.ndarray[np.float64_t, ndim=1] SMA(double[:] closes, int period):
    """
    Simple Moving Average function
    Note: this method skips nan values
    
    :param closes: np.array
    :param period: int
    :return sma: np.array
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(
        length - period + 1,
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

cpdef np.ndarray[np.float64_t, ndim=1] EMA(double[:] closes, int period):
    """
    Exponential Moving Average function
    Ref1: https://github.com/peerchemist/finta/blob/master/finta/finta.py
    Ref2: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    
    :param closes: np.array
    :param period: int
    :return ema: np.array
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(
        length,
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

cpdef np.ndarray[np.float64_t, ndim=1] MSTD(double[:] closes, int period):
    """
    Moving Standard Deviation function
    Ref: https://www.zaner.com/3.0/education/technicalstudies/MSD.asp#:~:text=The%20moving%20standard%20deviation%20is,moving%20average%20of%20the%20prices.
    The moving standard deviation is a measure of market volatility. It makes no
    predictions of market direction, but it may serve as a confirming indicator. 
    You specify the number of periods to use, and the study computes the 
    standard deviation of prices from the moving average of the prices.
    Note: this method skips nan values

    :param closes: np.array
    :param period: int
    :return mstd: np.array
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] _sma
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(
        length - period + 1,
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
    return np.where(result < 0 | np.isnan(result), np.nan, np.sqrt(result))

cpdef np.ndarray[np.float64_t, ndim=2] MACD(
        double[:] closes,
        int period_fast=12,
        int period_slow=26,
        int signal=9
):
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
    :param period_fast: int (optional)
    :param period_slow: int (optional)
    :param signal: int (optional)
    :param Returns: macd: np.ndarray
                    - col1: MACD
                    - col2: SIGNAL
    """
    cdef np.ndarray[np.float64_t, ndim=2] result
    cdef np.ndarray[np.float64_t, ndim=1] EMA_fast = EMA(closes, period_fast)
    cdef np.ndarray[np.float64_t, ndim=1] EMA_slow = EMA(closes, period_slow)
    cdef np.ndarray[np.float64_t, ndim=1] MACD = EMA_fast - EMA_slow
    cdef np.ndarray[np.float64_t, ndim=1] MACD_signal = EMA(MACD, signal)
    result = np.concatenate(
        (
            MACD[:, None],
            MACD_signal[:, None]
        ), axis=1
    )
    return result

cpdef np.ndarray[np.float64_t, ndim=1] RSI(double[:] closes, int period = 14):
    """
    Relative Strength Index (RSI) is a momentum oscillator that measures the 
    speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, 
    RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and 
    centerline crossovers.
    RSI can also be used to identify the general trend.
    
    :param closes: np.array
    :param period: int (optional)
    :return: rsi: np.array
    """
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
    # Avoid being divided by zeros
    RS = np.divide(
        _gain,
        _loss,
        out=np.inf * np.ones_like(_gain),
        where=_loss != 0
    )
    return 100 - (100 / (1 + RS))

cpdef np.ndarray[np.float64_t, ndim=1] TR(
        double[:] highs,
        double[:] lows,
        double[:] closes
):
    """
    True Range is the maximum of three price ranges:
    1. Most recent period's high minus the most recent period's low.
    2. Absolute value of the most recent period's high minus the previous close.
    3. Absolute value of the most recent period's low minus the previous close.

    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :return: tr: np.array
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
        int period=14
):
    """
    Average True Range is moving average of True Range.

    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :param period: int (optional)
    :return: atr: np.array
    """
    cdef np.ndarray[np.float64_t, ndim= 1] TR_ = TR(highs, lows, closes)
    return EMA(TR_, period=period)

cpdef np.ndarray[np.float64_t, ndim=1] SAR(
        double[:] highs,
        double[:] lows,
        double af=0.02,
        double amax=0.2
):
    """
    SAR stands for “stop and reverse,” which is the actual indicator used in 
    the system.
    SAR trails price as the trend extends over time. The indicator is below 
    prices when prices are rising and above prices when prices are falling.
    In this regard, the indicator stops and reverses when the price trend 
    reverses and breaks above or below the indicator.
    
    :param highs: np.array
    :param lows: np.array
    :param af: float (optional)
    :param amax: float (optional)
    :return: sar: np.array
    """
    cdef int length = highs.shape[0]
    # Starting values
    cdef double sig0, sig1, xpt0
    # next values
    cdef double xpt1, af0, af1
    # auxilary variables
    cdef double lmax, lmin
    cdef long i
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
        double band_multiple=3.0
):
    """
    SuperTrend indicator depends on previous supertrend.
    Note: dn line stands for upper bound; while up line stands for lower bound
    ATR channel, with mid/up/dn lines
        - In a down trend, if close climbs above dn, trend reverse to up -> buy
        - In a up trend, if close falls below up, trend reverse to down -> sell
    Ref1: https://cn.tradingview.com/script/r6dAP7yi/
    Ref2: https://zhuanlan.zhihu.com/p/138461317
    
    :param super_trend: Dict[str, float]
    :param highs: np.array
    :param lows: np.array
    :param closes: np.array
    :param timeperiod: int (optional)
    :param band_multiple: float (optional)
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
    if super_trend.size() > 0:
        trend = super_trend["trend"]
        dn1 = super_trend["dn"]
        up1 = super_trend["up"]
    else:
        n = len(closes_arr) / 3
        if closes_arr[:n].mean() < closes_arr[n:2 * n].mean() < closes_arr[2 * n:].mean():
            trend = 1.0
        else:
            trend = -1.0
        dn1 = float("inf")
        up1 = -float("inf")

    dn = mids[-1] + band_multiple * _atr[-1]
    up = mids[-1] - band_multiple * _atr[-1]
    if trend == -1.0 and closes[-1] > dn1:
        trend = 1.0
    elif trend == -1.0:
        dn = min(dn, dn1)
    elif trend == 1.0 and closes[-1] < up1:
        trend = -1.0
    elif trend == 1.0:
        up = max(up, up1)
    super_trend["mid"] = mids[-1]
    super_trend["ATR"] = _atr[-1]
    super_trend["up"] = up
    super_trend["dn"] = dn
    super_trend["trend"] = trend
    return super_trend

cpdef cppmap[string, double] TSV(
        double[:] closes,
        long[:] volumes,
        long tsv_length=13,
        long tsv_ma_length=7,
        long tsv_lookback_length=60
):
    """
    Time Segmented Volume
    Ref1: https://tw.tradingview.com/script/fmuLoK0d-time-segmented-volume-bands/?utm_source=amp-version&sp_amp_linker=1*zznimo*amp_id*YW1wLWVhdFVadXpBcjRIczRCMWpfN1l6VUE.
    
    :param closes: np.array
    :param volumes: np.array
    :param tsv_length: int (optional)
    :param tsv_ma_length: int (optional)
    :param tsv_lookback_length: int (optional)
    :return tsv: Dict[str, float]
    """
    closes_arr = np.asarray(closes)
    volumes_arr = np.asarray(volumes)
    cdef np.ndarray[np.float64_t, ndim=1] t, _tsv, _tsv_ma
    cdef np.ndarray[np.float64_t, ndim=1] _tsvp, _tsvn
    cdef np.ndarray[np.float64_t, ndim=1] _inflow, _outflow
    cdef np.ndarray[np.float64_t, ndim=1] _avg_inflow, _avg_outflow
    # cdef np.ndarray[np.float64_t, ndim=1] _tsv_pct
    cdef cppmap[string, double] tsv
    t = np.diff(closes_arr) * volumes_arr[1:]
    _tsv = np.convolve(t, np.ones(tsv_length, dtype=int), 'valid')
    _tsv_ma = SMA(_tsv, tsv_ma_length)

    # Total inflow and outflow
    _tsvp = _tsv.copy()
    _tsvp[_tsvp <= 0] = 0
    _inflow = np.convolve(
        _tsvp, np.ones(tsv_lookback_length, dtype=int), 'valid')
    _tsvn = _tsv.copy()
    _tsvn[_tsvn >= 0] = 0
    _outflow = np.convolve(
        _tsvn, np.ones(tsv_lookback_length, dtype=int), 'valid')

    # Average inflow and outflow
    _tsvp = _tsv.copy()
    _tsvp[_tsvp <= 0] = np.nan
    _avg_inflow = SMA(_tsvp, tsv_lookback_length)
    _tsvn = _tsv.copy()
    _tsvn[_tsvn >= 0] = np.nan
    _avg_outflow = SMA(_tsvn, tsv_lookback_length)
    tsv["tsv"] = _tsv[-1]
    tsv["tsv_ma"] = _tsv_ma[-1]
    tsv["tsv_inflow"] = _inflow[-1]
    tsv["tsv_outflow"] = _outflow[-1]
    tsv["tsv_avg_inflow"] = _avg_inflow[-1]
    tsv["tsv_avg_outflow"] = _avg_outflow[-1]
    return tsv

cpdef np.ndarray[np.float64_t, ndim=1] OBV(
        double[:] closes,
        long[:] volumes,
        long cum_obv = 0
):
    """
    On Balance Volume (OBV) measures buying and selling pressure as a cumulative 
    indicator that adds volume on up days and subtracts volume on down days.
    OBV was developed by Joe Granville and introduced in his 1963 book, 
    Granville's New Key to Stock Market Profits.
    It was one of the first indicators to measure positive and negative volume 
    flow.Chartists can look for divergences between OBV and price to predict 
    price movements or use OBV to confirm price trends.
    source: https://en.wikipedia.org/wiki/On-balance_volume#The_formula
    
    :param closes: np.array
    :param volumes: np.array
    :param cum_obv: int (optional), initial cumulative obv
    :return obv: np.array
    """
    cdef int n = len(closes)
    cdef np.ndarray[np.float64_t, ndim= 1] _obv = np.zeros(n)
    _obv[0] = cum_obv
    cdef int i
    for i in range(1, n):
        if closes[i] < closes[i-1]:
            _obv[i] = -volumes[i]
        elif closes[i] > closes[i-1]:
            _obv[i] = volumes[i]
    return _obv.cumsum()

cpdef double CYC(
        double[:] data,
        double cyc=0,
        long short_ma_length=10,
        long long_ma_length=30,
        double alpha=0.33,
        long lookback_window=10
):
    """
    Price cyclicality (PCY) and volume cyclicality (VOC)
    source:
    1. https://www.researchgate.net/publication/329756995_Price_Cyclicality_Model_for_Financial_Markets_Reliable_Limit_Conditions_for_Algorithmic_Trading#:~:text=The%20price%20cyclicality%20model%20is,included%20in%20the%20price%20behavior.
    2. https://www.researchgate.net/publication/342586790_VOLUME_CYCLICALITY_RELIABLE_CAPITAL_INVESTMENT_SIGNALS_BASED_ON_TRADING_VOLUME_INFORMATION
    
    :param data: np.array
    :param cyc: float, previous cyclicality
    :param short_ma_length: int
    :param long_ma_length: int
    :param alpha: float
    :param lookback_window: 
    :return: cyc: float, cyclicality for the data series
    """
    cdef long data_len
    if short_ma_length >= long_ma_length:
        raise ValueError(
            "Parameters short_ma_length should be smaller than long_ma_length; "
            f"but short_ma_length={short_ma_length}, and "
            f"long_ma_length={long_ma_length}.")
    data_len = len(data) - long_ma_length - lookback_window
    if data_len < -1:
        raise ValueError(
            "Data length is not sufficient to calculate the results for "
            f"long_ma_length={long_ma_length}, and "
            f"lookback_window={lookback_window}, at least"
            f"{long_ma_length}+{lookback_window}-1="
            f"{long_ma_length+lookback_window-1} is needed.")

    cdef np.ndarray[np.float64_t, ndim=1] ma_short, ma_long, ma_diff
    cdef double ma_diff_max, ma_diff_min, delta
    ma_short = SMA(data, short_ma_length)
    ma_long = SMA(data, long_ma_length)
    ma_diff = ma_short[-lookback_window:] - ma_long[-lookback_window:]
    ma_diff_max = ma_diff.max()
    ma_diff_min = ma_diff.min()
    if abs(ma_diff_max - ma_diff_min) < 1e-9:
        return cyc
    else:
        delta = 100 * (ma_diff_max - ma_diff[-1]) / (ma_diff_max - ma_diff_min)
        return alpha * (delta - cyc) + cyc



