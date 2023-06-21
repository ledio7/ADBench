"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from .annthyroid import Annthyroid
from .arrhythmia import Arrhythmia
from .breastw import Breastw
from .cardio import Cardio
from .credit_card import CreditCard
from .http import HTTP
from .letter import Letter
from .mammography import Mammography
from .mnist import Mnist
from .musk import Musk
from .optdigits import Optdigits
from .pendigits import Pendigits
from .satimage import Satimage
from .smtp import SMTP
from .speech import Speech
from .thyroid import Thyroid
from .vowels import Vowels
from .wbc import Wbc
from .rse import RSE

__all__ = [
    "Annthyroid",
    "Arrhythmia",
    "Breastw",
    "Cardio",
    "CreditCard",
    "HTTP",
    "Letter",
    "Mammography",
    "Mnist",
    "Musk",
    "Optdigits",
    "Pendigits",
    "Satimage",
    "SMTP",
    "Speech",
    "Thyroid",
    "Vowels",
    "Wbc",
    "RSE",
]
