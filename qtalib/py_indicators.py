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
from datetime import datetime

import numpy as np
import pandas as pd
from finta import TA


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
    """SuperTrend
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
