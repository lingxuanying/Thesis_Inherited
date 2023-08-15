# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:37:38 2023

@author: iir
"""
import os
import sys

command = "python trainning_data_for_fall40.py"
for i in range(1, len(sys.argv)):
    command+=" "+sys.argv[i]
os.system(command)

command = "python similarity.py"
for i in range(1, len(sys.argv)):
    command+=" "+sys.argv[i]
os.system(command)

command = "python training.py"
for i in range(1, len(sys.argv)):
    command+=" "+sys.argv[i]
os.system(command)
