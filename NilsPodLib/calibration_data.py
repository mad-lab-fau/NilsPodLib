# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import pickle


class CalibrationData:
    Ta = None
    Ka = None
    ba = None
    Tg = None
    Kg = None
    bg = None

    def __init__(self, calibration_file_path):
        with open(calibration_file_path, 'rb') as f:
            self.Ta, self.Ka, self.ba, self.Tg, self.Kg, self.bg = pickle.load(f, encoding='bytes')
