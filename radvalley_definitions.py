# Define functions that define the transition radius marking the radius valley.
import numpy as np

def define_radval_simple():
    return 1.8


def define_radval_lowmassstar(P):
    '''The period-dependent radius valley measured around low mass stars from 
    Cloutier & Menou 2019 (https://arxiv.org/abs/1912.02170).'''
    return .11 * np.log10(P) + 1.52


def define_radval_CKS(P):
    '''The period-dependent radius valley measured around the CKS sample from 
    Martinez et al 2019 (https://arxiv.org/abs/1903.00174).'''
    return -.48 * np.log10(P) + 2.59


def define_radval_gaspoor(P, Ms):
    '''The period-dependent radius valley for a specific stellar mass under the 
    gas-poor formation model from Lopez & Rice 2018 
    (https://arxiv.org/abs/1610.09390) and using the intercept from Cloutier
    & Menou 2019 (https://arxiv.org/abs/1912.02170).'''
    return (Ms/.65)**(.14) * define_radval_lowmassstar(P)


def define_radval_photoevap(P, Ms):
    '''The period-dependent radius valley for a specific stellar mass under the 
    photoevaporation model from Wu 2019 (https://arxiv.org/abs/1806.04693) and 
    using the intercept from Martinez et al 2019 
    (https://arxiv.org/abs/1903.00174).'''
    return (Ms/1.01)**(.25) * define_radval_CKS(P)
