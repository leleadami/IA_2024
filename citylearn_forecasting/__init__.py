"""
CityLearn Forecasting Package

A comprehensive time series forecasting toolkit for CityLearn building energy management,
featuring multiple deep learning architectures:
- LSTM
- LSTM Autoencoder
- LSTM with Attention
- LSTM Encoder-Decoder
- TimesNet

Author: Claude AI
Version: 1.0.0
"""

__version__ = '1.0.0'

from . import models
from . import data
from . import utils
from . import experiments

__all__ = ['models', 'data', 'utils', 'experiments']
