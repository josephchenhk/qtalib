# -*- coding: utf-8 -*-
# @Time    : 16/8/2022 5:45 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: py_indicators.py

"""
Copyright (C) 2020 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the
terms of the JXW license, which unfortunately won't be
written for another century.

You should have received a copy of the JXW license with
this file. If not, please write to: josephchenhk@gmail.com
"""

from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
from finta import TA


def sma(data_set: List[float], periods: int = 3) -> np.array:
    # weights = np.ones(periods) / periods
    # return np.convolve(data_set, weights, mode='valid')
    assert periods <= len(data_set), (
        f"periods {periods} can not be larger than the length of the input "
        "data_set!"
    )
    data_sma = np.zeros(len(data_set) - periods + 1)
    for i, idx in enumerate(range(periods, len(data_set) + 1)):
        data = data_set[idx - periods: idx]
        data = data[~np.isnan(data)]
        if len(data) > 0:
            data_sma[i] = data.mean()
    return data_sma


def offset_resample_ts(
        ts: List[float],
        resample_interval: int,
        resample_method: str,
        offset: int
) -> List[float]:
    """"""
    assert resample_method in ("first", "last", "max", "min", "sum"), (
        f"resample_method {resample_method} is not valid!"
    )
    # if offset > 0:
    #     ts = ts[resample_interval - offset:] + ts[-(resample_interval - offset):]
    if resample_interval > 1:
        resample_ts = []
        for i in range(0, len(ts), resample_interval):
            if i + resample_interval < len(ts):
                sample = ts[i:i + resample_interval]
            else:
                sample = ts[i:]
            if resample_method == "first":
                resample_ts.append(sample[0])
            elif resample_method == "last":
                resample_ts.append(sample[-1])
            elif resample_method == "max":
                resample_ts.append(max(sample))
            elif resample_method == "min":
                resample_ts.append(min(sample))
            elif resample_method == "sum":
                resample_ts.append(sum(ts[-resample_interval:]))
        ts = resample_ts[:]
    return ts


def ST(
        super_trend: Dict[str, Union[float, str]],
        high: np.array,
        low: np.array,
        close: np.array,
        timeperiod: int = 10,
        band_multiple: float = 3.0,
        trend_resample_interval: int = 1,
        trend_offset: int = 0
) -> Dict[str, Any]:
    """
    SuperTrend
    ATR channel, with mid/up/dn lines
     - close falls below dn, sell
     - close climbs above up, buy
    Ref1: https://cn.tradingview.com/script/r6dAP7yi/
    Ref2: https://zhuanlan.zhihu.com/p/138461317
    """
    assert len(
        high) >= timeperiod, "Not enough data to calculate supter trend!"

    # Offset and resample the data
    high = offset_resample_ts(
        ts=high,
        resample_interval=trend_resample_interval,
        resample_method="max",
        offset=trend_offset
    )
    low = offset_resample_ts(
        ts=low,
        resample_interval=trend_resample_interval,
        resample_method="min",
        offset=trend_offset
    )
    close = offset_resample_ts(
        ts=close,
        resample_interval=trend_resample_interval,
        resample_method="last",
        offset=trend_offset
    )

    high = np.array(high)
    low = np.array(low)
    close = np.array(close)

    mid = (high + low) * 0.5

    ohlc = pd.DataFrame({"open": None, "high": high,
                         "low": low, "close": close})
    ATR = TA.ATR(ohlc, timeperiod).to_list()

    dn = mid[-1] - band_multiple * ATR[-1]
    up = mid[-1] + band_multiple * ATR[-1]
    if super_trend is None:
        super_trend = {}
    if super_trend:
        trend = super_trend.get("trend")
        dn1 = super_trend.get("dn")
        up1 = super_trend.get("up")
        if close[-2] > dn1:
            dn = max(dn, dn1)
        if close[-2] < up1:
            up = min(up, up1)
    else:
        n = len(close) // 3
        if close[:n].mean() < close[n:2 * n].mean() < close[2 * n:].mean():
            trend = 1.0
        else:
            trend = -1.0
        dn1 = -float("inf")
        up1 = float("inf")

    if trend == -1.0 and close[-1] > up1:
        trend = 1.0
        dn = mid[-1] - band_multiple * ATR[-1]
        up = mid[-1] + band_multiple * ATR[-1]
    elif trend == 1.0 and close[-1] < dn1:
        trend = -1.0
        dn = mid[-1] - band_multiple * ATR[-1]
        up = mid[-1] + band_multiple * ATR[-1]
    super_trend["mid"] = mid[-1]
    super_trend["ATR"] = ATR[-1]
    super_trend["up"] = up
    super_trend["dn"] = dn
    super_trend["trend"] = trend
    return super_trend


def TSV(
        close: List[float],
        volume: List[float],
        tsv_length: int = 13,
        tsv_ma_length: int = 7,
        tsv_bands_length: int = 44,
        tsv_lookback: int = 60,
        tsv_resample_interval: int = 1,
        tsv_offset: int = 0
) -> Dict[str, Any]:
    """
    Time Segmented Volume
    Ref1: https://tw.tradingview.com/script/fmuLoK0d-time-segmented-volume-bands/?utm_source=amp-version&sp_amp_linker=1*zznimo*amp_id*YW1wLWVhdFVadXpBcjRIczRCMWpfN1l6VUE.
    """
    close = offset_resample_ts(
        ts=close,
        resample_interval=tsv_resample_interval,
        resample_method="last",
        offset=tsv_offset
    )
    volume = offset_resample_ts(
        ts=volume,
        resample_interval=tsv_resample_interval,
        resample_method="sum",
        offset=tsv_offset
    )

    # LOGIC - TSV
    t = np.diff(close) * volume[1:]
    t = np.convolve(t, np.ones(tsv_length, dtype=int), 'valid')
    m = sma(t, periods=tsv_ma_length)

    # LOGIC - Inflow / outflow
    tp = t.copy()
    tp[tp <= 0] = 0
    tn = t.copy()
    tn[tn >= 0] = 0
    inflow = np.convolve(tp, np.ones(tsv_lookback, dtype=int), 'valid')
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
    avg_inflow = sma(tpna, periods=tsv_bands_length)
    avg_outflow = sma(tnna, periods=tsv_bands_length)
    return {
        "t": t[-1],
        "m": m[-1],
        "avg_inflow": avg_inflow[-1],
        "avg_outflow": avg_outflow[-1],
        "difference": difference[-1],
        "inflow_p": inflow_p[-1],
        "outflow_p": outflow_p[-1]
    }

def SS(
        data: List[float],
        length: int
) -> List[float]:
    """
    Super Smoother（John Ehlers）
    Ref: https://zhuanlan.zhihu.com/p/557480350

    :param data:
    :param length:
    :return: List[float], Super smoothed price data
    """
    ssf = []
    for i, _ in enumerate(data):
        if i < 2:
            ssf.append(0)
        else:
            arg = 1.414 * 3.14159 / length
            a_1 = np.exp(-arg)
            b_1 = 2 * a_1 * np.cos(4.44 / float(length))
            c_2 = b_1
            c_3 = -a_1 * a_1
            c_1 = 1 - c_2 - c_3
            ssf.append(
                c_1 * (data[i] + data[i - 1]) / 2 + c_2 * ssf[i - 1] + c_3 *
                ssf[i - 2])
    return ssf
