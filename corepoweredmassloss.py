from imports import *
import planetary_structure as ps            



def compute_Mdot_energy_limited(Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    '''Calculate the mass loss rate in the energy-limited regime.'''
    # planet cooling luminosity (Eq 6 in Gupta & Schlichting 2019)
    _,Lcool,_ = ps.compute_cooling_time(Xenv, Mcore, Teq, Tkh, Xiron, Xice)

    # solve core
    Rcore = ps.mass2solidradius(Mcore, Xiron, Xice)
    g = G*Mearth2g(Mcore) / Rearth2cm(Rcore)**2
    
    # compute mass loss rate
    Mdot_E = Lcool / (g*Rearth2cm(Rcore))
    
    return Mdot_E



def compute_Mdot_Bondi_limited(Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    '''Calculate the mass loss rate in the energy limited regime.'''
    # compute sonic radius from Gupta & Schlichting 2019
    cs = np.sqrt(Kb * Teq / mu)
    Rs = cm2Rearth(G * Mearth2g(Mcore) / (2*cs**2))
    
    # solve rcb
    Rrcb,_,_ = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron, Xice)
    rho_rcb = ps.compute_rhorcb(Xenv, Rrcb, Mcore, Teq, Tkh, Xiron, Xice)
    
    # compute mass loss rate (Eq 10 Gupta & Schlichting 2019)
    Mdot_B = 4*np.pi*Rearth2cm(Rs)**2 * cs * rho_rcb
    Mdot_B *= np.exp(-G*Mearth2g(Mcore) / (cs**2 * Rearth2cm(Rrcb)))
    
    return Mdot_B



def evaluate_massloss_regime(Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    '''Compute the mass loss rates in the energy-limited and Bondi-limited 
    regimes to identify which regime the input set of parameters are in.'''
    args = Xenv, Mcore, Teq, Tkh, Xiron, Xice
    Mdot_E = compute_Mdot_energy_limited(*args)
    Mdot_B = compute_Mdot_Bondi_limited(*args)
    Mdot = np.min([Mdot_E, Mdot_B])
    regime = 'Bondi' if Mdot == Mdot_B else 'energy'
    return Mdot, regime


def compute_tmdot(Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    Mdot,_ = evaluate_massloss_regime(Xenv, Mcore, Teq, Tkh, Xiron, Xice)
    Menv = Xenv * Mearth2g(Mcore)
    tmdot = Menv / Mdot
    return tmdot


def _tmdot_to_maximize(lg_Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    # solve for mass loss timescale in the applicable mass loss regime
    tmdot = compute_tmdot(10**lg_Xenv, Mcore, Teq, Tkh, Xiron, Xice)
    return 1/tmdot 


def compute_Xmax(Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction, of either the rocky or gaseous planet,
    that maximizes its mass loss timescale given the planetary parameters.
    '''
    # maximize the planet's mass loss timescale
    args = Mcore, Teq, Tkh, Xiron, Xice
    res = minimize_scalar(_tmdot_to_maximize, bounds=(-8,0), args=args,
                          method='bounded')
    if res.success:
        Xenv_max = float(10**res.x)
        return Xenv_max
    else:
        raise ValueError("Cannot solve for the envelope mass fraction that maximizes gthe planet's mass loss timescale.")



#def compute_Mcore_min_no_self_gravity(Rp_now, Teq, age, Xiron, Xice):
#    '''Compute the minimum core mass required for the envelope to be non-self 
#    gravitating (i.e. Xenv = 1).'''
#    args = Rp_now, Teq, age, Xiron, Xice
#    Mcore_min = 10**fsolve(_Mcore_to_minimize, -1, args=args)
#    return Mcore_min
#def _Mcore_to_minimize(lg_Mcore, Rp_now, Teq, age, Xiron, Xice):
#    Mcore = 10**float(lg_Mcore)
#    Xenv,_ = ps.Rp_solver_gas(Rp_now, Mcore, Teq, age, Xiron, Xice)
#    return Xenv - 1


def _Mp_gas_to_solve(lg_Mcore, Teq, Xiron, Xice, Rp_now, age, tmdot_rocky):
    '''
    Objective function that equates the gaseous planet's mass loss timescale, 
    given its mass, to the input maximum mass loss timescale of the rocky
    planet.
    '''
    # get envelope mass fraction at age of the system
    Mcore = 10**lg_Mcore
    try:
        Xenv, Rp_full = ps.Rp_solver_gas(Rp_now, Mcore, Teq, age, Xiron, Xice)
    except ValueError:
        return np.inf

    # get gaseous planet's mass loss timescale
    tmdot_gaseous = compute_tmdot(Xenv, Mcore, Teq, age, Xiron, Xice)
    
    return tmdot_gaseous - tmdot_rocky



if __name__ == '__main__':
    args = 3.12, 812, 100, 1/3, 0
    Xenv_rocky = compute_Xmax_rock(*args)
    tmdot_rocky = compute_tmdot(Xenv_rocky, *args)
    Mgas_min = 10**brentq(_Mp_gas_to_solve, -2, 2,
                          args=(353, 1/3, 0, 2.3, 5e3, tmdot_rocky))
