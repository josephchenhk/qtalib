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
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cpython cimport array


cpdef np.ndarray[np.float64_t, ndim=1] SMA(double[:] closes, int period):
    """
    Simple Moving Average function 
    @param closes: list of closing candle prices 
    @param period: period to calculate for 
    """
    cdef int length = closes.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(length - period + 1,
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