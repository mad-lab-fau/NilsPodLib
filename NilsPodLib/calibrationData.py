# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import pickle


class calibrationData:
    Ta = None
    Ka = None
    ba = None
    Tg = None
    Kg = None
    bg = None

    def __init__(self, calibrationFilePath):
        with open(calibrationFilePath, 'rb') as f:  # Python 3: open(..., 'rb')
            self.Ta, self.Ka, self.ba, self.Tg, self.Kg, self.bg = pickle.load(f, encoding="bytes")
