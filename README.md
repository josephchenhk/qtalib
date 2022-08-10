# QTALIB: Quantitative Technical Analysis Library

<p align="center">
    <img src ="https://img.shields.io/badge/version-0.0.1-blueviolet.svg"/>
    <img src ="https://img.shields.io/badge/platform-windows|linux|macos-yellow.svg"/>
    <img src ="https://img.shields.io/badge/python-3.8-blue.svg" />
    <img src ="https://img.shields.io/github/workflow/status/vnpy/vnpy/Python%20application/master"/>
    <img src ="https://img.shields.io/badge/license-JXW-orange"/>
</p>

**Latest update on 2022-08-10**

## Available technical indicators

* Simple Moving Average (SMA)

* Exponential Moving Average (EMA)

## Installation

You may run the folllowing command to install QTalib immediately:

```python
# Virtual environment is recommended (python 3.8 or above is supported)
>> conda create -n qtalib python=3.8
>> conda activate qtalib

# Install stable version from pip (currently version 0.0.1)
>> pip install qtalib

# Alternatively, install latest version from github 
>> pip install git+https://github.com/josephchenhk/qtalib@master
```

## Usage

```python
import numpy as np
import qtalib.indicators as ta

values = np.array([12.0, 14.0, 64.0, 32.0, 53.0])

# Simple Moving Average
# [30.         36.66666667 49.66666667]
print(ta.SMA(values, 3))

# Exponential Moving Average
# [12.         13.33333333 42.28571429 36.8        45.16129032]
print(ta.EMA(values, 3))
```

## Contributing
* Fork it (https://github.com/josephchenhk/qtalib/fork)
* Study how it's implemented.
* Create your feature branch (git checkout -b my-new-feature).
* Use [flake8](https://pypi.org/project/flake8/) to ensure your code format
complies with PEP8.
* Commit your changes (git commit -am 'Add some feature').
* Push to the branch (git push origin my-new-feature).
* Create a new Pull Request.