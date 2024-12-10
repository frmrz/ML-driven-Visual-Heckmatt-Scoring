#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:07:28 2022

@author: francesco
"""

import pyreadstat
import pandas as pd

filename = '/media/francesco/DEV001/PROJECT-FSHD/DATA/TABULAR/clinicalDataFSHD.sav'

df, meta = pyreadstat.read_sav(filename)
colNames = meta.column_names
colLabels = meta.column_labels

filtered_columns_qmus = df.filter(like='QMUS', axis=1)
filtered_columns_zscore = df.filter(like='zscore', axis=1)
