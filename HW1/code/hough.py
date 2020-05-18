# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:10:18 2019

@author: noron
"""

import numpy as np
import matplotlib.pyplot as plt
import math
thetas = np.deg2rad(np.arange(0, 360))

a = [(10,10),(15,15),(30,30)]

for i in range(len(a)):
    x = []
    y = []
    for j in thetas:
        
        p = (a[i][0])*math.cos(j) + (a[i][1])*math.sin(j)
        y.append(p)
        x.append(j)
    
    plt.plot(x,y)
    plt.xlabel('theta in radians')
    plt.ylabel('rho')
plt.show()
        