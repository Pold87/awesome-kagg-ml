# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:43:38 2015

@author: Trost
"""

import pandas as pd
import os

from matplotlib import pyplot as plt


###### Data paths 
# Thomas
# data_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\1"
# Volker
data_path = r"/home/pold/Documents/Radboud/kaggle/drivers/1"

rand = 6

data = pd.read_csv(os.path.join(data_path, str(rand) + ".csv"))

print(data.head())

distance = data.x + data.y

plt.plot(data.x,data.y)


###########Hic sunt dracones

f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
ax1.plot(data.x,data.y)
ax2.scatter(data.x,data.y)
ax3.scatter(data.y,data.x)
ax4.plot(data.y,data.x)


from pylab import *
import numpy as np
#
#x = np.linspace(0, 2*np.pi, 400)
#y = np.sin(x**2)
#
#subplots_adjust(hspace=0.000)
#number_of_subplots=3
#
#for i,v in enumerate(range(number_of_subplots)):
#    v = v+1
#    ax1 = subplot(number_of_subplots,1,v)
#    ax1.plot(x,y)
#
#plt.show()
#
#
#test_frame = pd.DataFrame()

from os import listdir
#
#data_list = listdir(data_path)
#print(data_list)
#data_list = data_list[:1]
#print(data_list)
#for file in data_list:
#    test_frame[str(file)] = pd.read_csv(data_path+"\\"+file)
