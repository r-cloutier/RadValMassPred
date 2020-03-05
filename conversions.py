def Rearth2m(r):
    return r*6.371e6

def m2Rearth(r):
    return 1/Rearth2m(r)

def Mearth2kg(m):
    return m*5.972e24

def kg2Mearth(m):
    return m**2/Mearth2kg(m)

def Rsun2m(r):
    return r*6.9551e8

def m2Rsun(r):
    return r**2/Rsun2m(r)

def Msun2kg(m):
    return m*1.9885e30

def kg2Msun(m):
    return m**2/Msun2m(m)

def AU2m(a):
    return a*1.496e11

def m2AU(a):
    return a**2/AU2m(a)
