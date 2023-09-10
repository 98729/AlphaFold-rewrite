#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:04:26 2022

@author: sam
"""
import numpy as np
import math
np.random.seed(666)
a = np.array([-2,-1,0,1,2,3]).astype(np.float32)
a = np.reshape(a,(2,3))
print(a)    
index1 = np.where(a>4)
index2 = np.where(a<2)
print(index1)
b = a[index1]

print(b)
print(a)
a[index1] = b
print(a)
# if len(a[a>1]) > 0 or len(a[a<-1]) > 0:
#     print('no')

# print(np.log(-np.inf))

# print(np.nextafter(2.,np.inf))

# lower = 2.
# print(np.shape(lower))

print(np.issubdtype(np.float64, np.complexfloating))

shape = (2,2)
shape = [2, *shape]
print(shape)