# -*- coding: utf-8 -*-
# @Time    : 10/8/2022 3:15 am
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: setup.py

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


cpdef np.ndarray[np.float64_t, ndim = 1] shift(
        double[:] arr,
        int num,
        double fill_value=np.nan
):
    """
    Shift a numpy array
    preallocate empty array and assign slice by chrisaycock
    Ref: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    """
    cdef np.ndarray[np.float64_t, ndim = 1] result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

cpdef np.ndarray[np.float64_t, ndim = 1] ewm(
        double[:] data,
        int window
):
    """Exponential Weighted Moving Average"""
    data_arr = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data_arr*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out
