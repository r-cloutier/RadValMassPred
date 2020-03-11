# define constants
global alpha, beta, kappa0, mH, mu, gamma, grad_ab, Kb, sigma, G
alpha = 0.68          # pressure dependence of opacity
beta = 0.45           # temperature dependence of opacity
kappa0 = 10**(-7.32)  # opacity constant
mH = 1.673558e-24     # mass of hydrogen atom in g
mu = 2.35 * mH        # solar metallicity gas
gamma = 5/3           # ratio of specific rho_earth_cgs
grad_ab = (gamma-1) / gamma
Kb = 1.38064852e-16   # boltzmann's constant [cgs]
sigma = 5.6704e-5     # Stefan-Boltzmann constant [cgs]
G = 6.67408e-8       # Gravitational constant [cgs]


# define conversion functions
def Rearth2cm(r):
    return r*6.371e8

def cm2Rearth(r):
    return r**2/Rearth2cm(r)

def Mearth2g(m):
    return m*5.972e27

def g2Mearth(m):
    return m**2/Mearth2g(m)

def Rsun2cm(r):
    return r*6.9551e10

def cm2Rsun(r):
    return r**2/Rsun2cm(r)

def Msun2g(m):
    return m*1.9885e33

def g2Msun(m):
    return m**2/Msun2g(m)

def AU2cm(a):
    return a*1.496e13

def cm2AU(a):
    return a**2/AU2cm(a)

def days2yrs(t):
    return t/365.25

def yrs2days(t):
    return t**2/days2yrs(t)

def sec2Myrs(t):
    return t / 60 / 60 / 24 / 365.25 / 1e6

def Myrs2sec(t):
    return t**2/sec2Myrs(t)
