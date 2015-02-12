# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:43:38 2015

@author: Trost
"""

import pandas as pd
import os

from matplotlib import pyplot as plt


data_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\1"

rand = 1

data = pd.read_csv(os.path.join(data_path, str(rand), ".csv"))

print(data.head())

distance = data.x + data.y

plt.plot(data.x,data.y)
