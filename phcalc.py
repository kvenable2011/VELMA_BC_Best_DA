# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:31:37 2021

@author: KVENABLE
"""

from traceback import format_exc
import pandas as pd
import numpy as np
#import seaborn as sns
import math

#"Inverse hyperbolic sine of a number"
def calc_asinh(x):
    return(math.asinh(x));

def calc_pH(CCO2C, T, DOC, Alk):
    "pH calculation based on Marmorek et al., 1998  (modified Small and Sutton, 1986)"
   # DOC =  Comes from VELMA output from run 
   # CCO2C = DOC/3;
    #print("C02 Scaled", CCO2C)
    CCO2 = CCO2C/ 44.0 * 1000.0; "Converts DOC in mg/L to ueq/mg"
    print("CO2 Estimate", CCO2)
    y =  (6.57 - 0.0118 * T + 0.00012 * (T*T))* 0.92
    print("Exponent value", y)
    pH2CO3 = math.pow(10.0, -y) 
    print("H2CO3 value", pH2CO3)
    pkw = math.pow(10.0,-14.0);
    print("pkw=", pkw) #ionizaiton constant water
    Alpha = pH2CO3 * CCO2 + pkw
    sq_alpha = math.sqrt(Alpha)
    print("Alpha value=", Alpha)
    print("SQRT of Alpha=", sq_alpha)
    A = - (math.log(sq_alpha,10.0)); 
    print("A value", A)
    B = 1.0/ math.log(10.0);
    print("B value", B)
    C = 2 * (math.sqrt(Alpha));
    print("C value", C)
   
    try:
        "Default ueq CaCO3/L Replace with site-specif and/or time specific data where avaiable"
        #Alk = 1020;
        print("")
        return(A + B * calc_asinh((Alk - 5.1 * DOC * 0.5)/ C))
    except:
        print(format_exc())
        return 7; 
    