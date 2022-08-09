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
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cpython cimport array


cpdef float SMA(double[:] closes, int period):
    """
    Simple Moving Average function 
    @param closes: list of closing candle prices 
    @param period: period to calculate for 
    """
    cdef int length = closes.shape[0]
    cdef float total = closes[length-period]
    cdef int i
    for i in range((length-period)+1, length):
            total += closes[i]
    return float(total/period)