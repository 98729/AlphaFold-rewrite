#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:47:44 2022

@author: sam
"""

import numpy as np
import math


def polevl(x:int or float or np.ndarray, coef:np.ndarray):
    
    ans = coef[0]
    n = len(coef)
    
    i = 1
    while i < n:
        ans = ans * x + coef[i]
        i += 1
    return ans

def p1evl(x:int or float or np.ndarray, coef:np.ndarray):
    
    n = len(coef)
    ans = x + coef[0]
    
    i = 1
    while i < n:
        ans = ans * x + coef[i]
        i += 1
    return ans 

def ndtri(y:int or float or np.ndarray):
    
    P0 = np.array([
            -5.99633501014107895267E1,
            9.80010754185999661536E1,
            -5.66762857469070293439E1,
            1.39312609387279679503E1,
            -1.23916583867381258016E0],'d')
    
    Q0 = np.array([
            1.95448858338141759834E0,
            4.67627912898881538453E0,
            8.63602421390890590575E1,
            -2.25462687854119370527E2,
            2.00260212380060660359E2,
            -8.20372256168333339912E1,
            1.59056225126211695515E1,
            -1.18331621121330003142E0],'d')
    
    P1 = np.array([
            4.05544892305962419923E0,
            3.15251094599893866154E1,
            5.71628192246421288162E1,
            4.40805073893200834700E1,
            1.46849561928858024014E1,
            2.18663306850790267539E0,
            -1.40256079171354495875E-1,
            -3.50424626827848203418E-2,
            -8.57456785154685413611E-4],'d')
    
    Q1 = np.array([
            1.57799883256466749731E1,
            4.53907635128879210584E1,
            4.13172038254672030440E1,
            1.50425385692907503408E1,
            2.50464946208309415979E0,
            -1.42182922854787788574E-1,
            -3.80806407691578277194E-2,
            -9.33259480895457427372E-4],'d')
    
    P2 = np.array([
            3.23774891776946035970E0,
            6.91522889068984211695E0,
            3.93881025292474443415E0,
            1.33303460815807542389E0,
            2.01485389549179081538E-1,
            1.23716634817820021358E-2,
            3.01581553508235416007E-4,
            2.65806974686737550832E-6,
            6.23974539184983293730E-9],'d')
    
    Q2 = np.array([
            6.02427039364742014255E0,
            3.67983563856160859403E0,
            1.37702099489081330271E0,
            2.16236993594496635890E-1,
            1.34204006088543189037E-2,
            3.28014464682127739104E-4,
            2.89247864745380683936E-6,
            6.79019408009981274425E-9],'d')
    
    s2pi = 2.50662827463100050242E0
    code = 1
    
    if isinstance(y,int) or isinstance(y,float):
        if y == 1:
            return math.inf
        if y == 0:
            return -math.inf
        
        if y < 0 or y > 1:
            raise ValueError('input for ndtir could not be smaller than 0 or larger than 1')
        
        
        if y > (1.0 - 0.13533528323661269189):      # 0.135... = exp(-2)
            y = 1.0 - y
            code = 0

        if y > 0.13533528323661269189:
            y = y - 0.5
            y2 = y * y
            x = y + y * (y2 * polevl(y2, P0) / p1evl(y2, Q0))
            x = x * s2pi
            return x

        x = math.sqrt(-2.0 * math.log(y))
        x0 = x - math.log(x) / x

        z = 1.0 / x
        if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
            x1 = z * polevl(z, P1) / p1evl(z, Q1)
        else:
            x1 = z * polevl(z, P2) / p1evl(z, Q2)

        x = x0 - x1
        if code != 0:
            x = -x

        return x
    
    elif isinstance(y, np.ndarray):
        
        index_min = np.where(y < 0)
        index_max = np.where(y > 1)
        
        
        if len(y[index_min]) > 0 or len(y[index_max]) > 0:
            raise ValueError('input contains value which is smaller than 0 or larger than 1, please check it')
        err = (True in np.isnan(y))
        
        if err:
            raise ValueError('There are missing values in the input, please check it')
        
        y[y == 0] = -np.inf
        y[y == 1] = np.inf
        
        index_mininf = np.where(y == -np.inf)
        index_maxinf = np.where(y == np.inf)
        
        index1 = np.where(y > (1.0 - 0.13533528323661269189))
        #index2 = np.where(y > 0.13533528323661269189)
        #index3 = np.where(y <= 0.13533528323661269189)
        y1 = y[index1]
        #print(y1)
        if len(y1) > 0:
            
            #y1 = y[index1]
            y1 = 1.0 - y1
            code = 0
            
            y[index1] = y1
            
            if len(y[index_mininf]) > 0:
                y[index_mininf] = -np.inf
            if len(y[index_maxinf]) > 0:
                y[index_maxinf] = np.inf
                
            #print(y)
            index2 = np.where(y > 0.13533528323661269189)
            index3 = np.where(y <= 0.13533528323661269189)
            y2 = y[index2]
            y3 = y[index3]
            if len(y2) > 0:
                
                y2 = y[index2]
                y2 = y2-0.5
                # y[index2] = y2
                # print(y)
                newy2 = y2 * y2
                x = y2 + y2 * (newy2 * polevl(newy2, P0) / p1evl(newy2, Q0))
                x = x * s2pi
                y[index2] = x
                #print(y)
                
                if len(y3) > 0:
                    
                    #print(y3)
                    x = np.sqrt(-2.0 * np.log(y3))
                    
                    #print(x)
                    x0 = x - np.log(x) / x
                    
                    x[np.isnan(x)] = -np.inf
                    x0[np.isnan(x0)] = -np.inf
                    z = 1.0 / x
                    
                    indexa = np.where(x < 8.0)
                    indexb = np.where(x >= 8.0)
                    x1 = x[indexa]
                    x2 = x[indexb]
                    
                    if len(x1) > 0:
                        
                        x1 = z * polevl(z, P1) / p1evl(z, Q1)
                        x[indexa] = x1
                    
                    if len(x2) > 0:
                        x2 = z * polevl(z, P2) / p1evl(z, Q2)
                        x[indexb] = x2
                    
                   
            
                    x = x0 - x
                    # if code != 0:
                    x = -x
                    y[index3] = x
                    y[index1] = -y[index1]
                    y[index_mininf] = -np.inf
                    y[index_maxinf] = np.inf
                return y
            
        elif len(y1) == 0:
            index2 = np.where(y > 0.13533528323661269189)
            index3 = np.where(y <= 0.13533528323661269189)
            y2 = y[index2]
            
            y3 = y[index3]
            #print('y3: ',y3)
            if len(y2) > 0:
                #y2 = y[index2]
                y2 = y2-0.5
                #[index2] = y2
                newy2 = y2 * y2
                x = y2 + y2 * (newy2 * polevl(newy2, P0) / p1evl(newy2, Q0))
                x = x * s2pi
                y[index2] = x
                #print(y)
                
                if len(y3) > 0:
                    
                    x = np.sqrt(-2.0 * np.log(y3))
                    x0 = x - np.log(x) / x
            
                    z = 1.0 / x
                    
                    
                    indexa = np.where(x < 8.0)
                    indexb = np.where(x >= 8.0)
                    x1 = x[indexa]
                    x2 = x[indexb]
                    
                    if len(x1) > 0:
                        
                        x1 = z * polevl(z, P1) / p1evl(z, Q1)
                        x[indexa] = x1
                    
                    if len(x2) > 0:
                        x2 = z * polevl(z, P2) / p1evl(z, Q2)
                        x[indexb] = x2
            
                    x = x0 - x
                    # if code != 0:
                    x = -x
                    y[index3] = x
                    
                return y
        
        x = math.sqrt(-2.0 * math.log(y))
        x0 = x - math.log(x) / x

        z = 1.0 / x
        if x < 8.0:                 # y > exp(-32) = 1.2664165549e-14
            x1 = z * polevl(z, P1) / p1evl(z, Q1)
        else:
            x1 = z * polevl(z, P2) / p1evl(z, Q2)

        x = x0 - x1
        if code != 0:
            x = -x

        return x



def erf_inv(x:int or float or np.ndarray):
    
    if isinstance(x, int) or isinstance(x, float):
        if x < -1 or x > 1:
            raise ValueError("`x` must be between -1 and 1 inclusive")

        if x == 0:
            return 0
        if x == 1:
            return math.inf
        if x == -1:
            return -math.inf
        
        result = ndtri((x + 1) / 2.0) / math.sqrt(2)
        
        return result
    
    elif isinstance(x, np.ndarray):
        if len(x[np.where(x < -1)]) > 0 or len(x[np.where(x > 1)]) > 0:
            raise ValueError("`x` must be between -1 and 1 inclusive")
        
        result = ndtri((x + 1) / 2.0) / np.sqrt(2)
        return result
    
    
P = np.array([2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2],'d')

Q = np.array([1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2],'d')

R = np.array([5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0],'d')

S = np.array([2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0],'d')


T = np.array([9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4],'d')

U = np.array([3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4],'d')

def erfc(x:int or float or np.ndarray):
    
    if isinstance(x, int) or isinstance(x, float):
        z = -x * x
        z = np.exp(z)
        #print(z)
        
        if x < 8:
            p = polevl(x, P)
            #print(p)
            q = p1evl(x, Q)
            #print(q)
            
        else:
            p = polevl(x, R)
            q = p1evl(x, S)
        
        y = (z * p) / q

        return y
    
    elif isinstance(x, np.ndarray):
        
        x = x.astype(np.float64)
        z = -x * x
        z = np.exp(z)
        #print(z)
        index1 = np.where(x < 8.0)
        index2 = np.where(x >= 8.0)
        
        x1 = x[index1]
        x2 = x[index2]

        if len(x1) > 0 and len(x2) > 0:
            p = np.zeros(x.shape,dtype = x.dtype)
            p1 = polevl(x1, P)
            p2 = polevl(x2, R)
            p[index1] = p1
            p[index2] = p2
            #print(p)
            q = np.zeros(x.shape,dtype = x.dtype)
            q1 = p1evl(x1, Q)
            q2 = p1evl(x2, S)
            q[index1] = q1
            q[index2] = q2
            #print(q)
            
        
        elif len(x1) > 0 and len(x2) == 0:
            p = polevl(x, P)
            q = p1evl(x, Q)
            
        elif len(x1) == 0 and len(x2) > 0:
            p = polevl(x, R)
            q = p1evl(x, S)
        
        
        y = (z * p) / q

        return y


def erf(x:int or float or np.ndarray):
    
    if not isinstance(x, np.ndarray) and x == float('-inf'):
        
        
        y = -1
        return y
            
    elif not isinstance(x, np.ndarray) and x == float('inf'):
        y = 1
        return y
    
    elif isinstance(x, int) or isinstance(x, float):
        
        if x < -1:
            
            return -(1 - erfc(abs(x)))
        
        elif -1 <= x < 0:
            x = abs(x)
            z = x * x
            y = -x * polevl(z, T) / p1evl(z, U)
            return y
        
        elif 0 <= x <= 1:
            z = x * x;

            y = x * polevl(z, T) / p1evl(z, U)
        
            return y
        
        elif x > 1:
            return (1 - erfc(x))        
        
        
    elif isinstance(x, np.ndarray):
        import warnings
        warnings.filterwarnings("ignore")
        
        x = x.astype(np.float64)
        
        index_mininf = np.where(x == -np.inf)
        index_maxinf = np.where(x ==  np.inf)
        
        index1 = np.where(x < -1)
        index2= np.where((x >= -1) & (x < 0))
        
        index3 = np.where((x >= 0) & (x<= 1))
        index4 = np.where(x > 1)
        
        x1 = x[index1]
        x2 = x[index2]
        x3 = x[index3]
        x4 = x[index4]
        
        y = np.zeros(x.shape)
        if len(x1) > 0:
            y1 = -(1 - erfc(np.abs(x1)))
            y[index1] = y1
        
        if len(x2) > 0:
            x2 = np.abs(x2)
            z = x2 * x2
            y2 = -x2 * polevl(z, T) / p1evl(z, U)
            y[index2] = y2
            
        if len(x3) > 0:
            z = x3 * x3
            y3 = x3 * polevl(z, T) / p1evl(z, U)
            y[index3] = y3
            
        if len(x4) > 0:
            y4 = 1 - erfc(x4)
            y[index4] = y4
        
        if len(y[index_mininf]) > 0:
            y[index_mininf] = -1
        
        if len(y[index_maxinf]) > 0:
            y[index_maxinf] = 1
            
        return y
    
            
# x = np.array([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,0.85,0.9,0.95])
# x = np.reshape(x,(2,2,3))
# print(x)
# y = ndtri(x)
# print(y)            
# print(erf_inv(x))



#x = np.array([3,4,5,6,7,8,9,10])
#x = np.reshape(x,(2,4))        
# y = erfc(x)
# print(y)

#x = np.array([-3,-2,-1,0,1,2,3,4])
#x = np.reshape(x,(2,4))
#y = erf(-2)
#print(y)
# from scipy.special import erfc

# print(erfc(x))

# from scipy.special import erf
# print(erf(6))

# from scipy.special import erfinv
# print('\n')
# print(erfinv(1))














# a = np.random.randint(-10,10,size = (2,3,4))
# print(a)    
# print(a[a>0])
# if len(a[a>1]) > 0 or len(a[a<-1]) > 0:
#     print('no')
    