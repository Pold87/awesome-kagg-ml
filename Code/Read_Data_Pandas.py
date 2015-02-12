# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:43:38 2015

@author: Trost
"""

import pandas as pd

data_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\1\1.csv"

data = pd.read_csv(data_path)

data.head()