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

################################################################################
#                              FFILL and BFILL                                 #
# Ref: https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
################################################################################
cpdef np.ndarray[np.float64_t, ndim = 2] ffill_2d(double[:,:] arr):
    cdef np.ndarray[np.float64_t, ndim = 2] out
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.asarray(arr)[np.arange(idx.shape[0])[:,None], idx]
    return out

cpdef np.ndarray[np.float64_t, ndim = 1] ffill_1d(double[:] arr):
    cdef np.ndarray[np.float64_t, ndim = 1] out
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = np.asarray(arr)[idx]
    return out

def ffill(arr: np.ndarray) -> np.ndarray:
    shape = np.asarray(arr).shape
    if len(shape) == 1:
        return ffill_1d(arr)
    elif len(shape) == 2:
        return ffill_2d(arr)
    else:
        raise ValueError("Array dimension is NOT allowed.")

cpdef np.ndarray[np.float64_t, ndim = 2] bfill_2d(double[:,:] arr):
    return ffill(np.asarray(arr[:, ::-1]))[:, ::-1]

cpdef np.ndarray[np.float64_t, ndim = 1] bfill_1d(double[:] arr):
    return ffill(np.asarray(arr[::-1]))[::-1]

def bfill(arr: np.ndarray) -> np.ndarray:
    shape = np.asarray(arr).shape
    if len(shape) == 1:
        return bfill_1d(arr)
    elif len(shape) == 2:
        return bfill_2d(arr)
    else:
        raise ValueError("Array dimension is NOT allowed.")
################################################################################

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

cpdef np.ndarray[np.float64_t, ndim=2] unstructured_to_structured(
        np.ndarray[np.float64_t, ndim=2] data,
        list column_names,
):
    """Covert a normal numpy ndarray to a structured numpy ndarray with column 
    names"""
    # create a structured numpy ndarray with column names
    structured_data = np.recarray(
        shape=(data.shape[0],),
        dtype=[(column_name, data.dtype) for column_name in column_names]
    )
    # copy the data into the structured numpy ndarray
    for i, column_name in enumerate(column_names):
        structured_data[column_name] = data[:, i]
    return structured_data
