from imports import *
import planetary_structure as ps            


def compute_Xmax_rock(P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the rocky planet that maximizes its
    mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    Xenv_max = fsolve(_tmdot_to_maximize, .01, args=args)
    return float(Xenv_max)



def compute_Xmax_gas(P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the gaseous planet that maximizes 
    its mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    Xenv_max = fsolve(_tmdot_to_maximize, .01, args=args)
    return float(Xenv_max)



def compute_Mcore_min_no_self_gravity(Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice):
    '''Compute the minimum core mass required for Xenv to be < 1 such that 
    atmospheric self gravity can be ignored.'''
    args = Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice
    Mcore_min = fsolve(_Mcore_to_minimize, .01, args=args)
    return float(Mcore_min)

    

def compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''Compute mass loss timescale and envelope depth.'''
    assert 0 <= Xenv <= 1  # avoid self-gravitating envelope when Menv>Mcore
    
    # solve planetary structure
    Rcore = ps.mass2solidradius(Mcore, Xiron, Xice)
    Rrcb, Rp_full, f = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh,
                                                 Xiron, Xice)
    depth_env = (Rp_full-Rcore)/Rcore # increases with Xenv, decreases with Mc
    rho1_scaled = 1 / ps.mass2solidradius(1, Xiron, Xice)**3
    rho_rcb = ps.compute_rhorcb(Xenv, Rrcb, Mcore, Teq, Tkh, Xiron, Xice)
    eta = compute_mass_loss_efficiency(Mcore, Rp_full)

    # compute mass loss timescale using Eq 20 from Owen & Wu 2017
    tmdot = 210 * (eta/.1)**(-1)
    tmdot *= (P/10)**(1.41)
    tmdot *= Ms**(.52)
    tmdot *= (f/1.2)**(-3)
    tmdot *= (Tkh/100)**(.37)
    tmdot *= rho1_scaled**(.18)
    tmdot *= (Mcore/5)**1.42
    a = 1.57 if depth_env < 1 else -1.69
    tmdot *= depth_env**a
   
    # this is the expression in EvapMass which I think has the wrong scaling with eta
    # verbatim from mass_loss.tmdot_rocky:tmdot = X * Mcore**2. * sep_cm**2. * eff / Rplanet**3.
    sma = Ms**(1/3) * (P/365.25)**(2/3)
    tmdot = Xenv * Mcore**2 * sma**2 * eta / Rp_full)
 
    return tmdot, depth_env



def _Mp_gas_to_solve(lg_Mcore, P, Ms, Teq, Tkh, Xiron, Xice, Rp_now, age,
                     tmdot_rocky):
    '''
    Objective function that equates the gaseous planet's mass loss timescale, 
    given its mass, to the input maximum mass loss timescale of the rocky
    planet.
    '''
    # evaluate the current envelope mass fraction
    Mcore = 10**lg_Mcore
    try:
        Xenv, Rp_full = ps.Rp_solver_gas(Rp_now, Mcore, Teq, age, Xiron, Xice)
    except ValueError:
        return np.inf
        
    if not np.isclose(float(Rp_full), Rp_now, rtol=1e-4):
        raise ValueError('No solution for the gaseous planet found.')

    # check for consistent X
    Rcore = ps.mass2solidradius(Mcore, Xiron, Xice)
    Rrcb, Rp_full, f = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron,
                                                 Xice)
    DRrcb = Rrcb - Rcore
    Xtest = ps.compute_Xenv(DRrcb, Mcore, Teq, Tkh, Xiron, Xice)
    
    if not np.isclose(Xtest, Xenv, rtol=1e-4):
        raise ValueError('No self-consistent structure.')

    # compute gaseous planet mass loss timescale
    tmdot_gaseous,_ = compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    return tmdot_gaseous - tmdot_rocky




def compute_mass_loss_efficiency(Mp, Rp):
    '''Compute the mass loss efficiency using the equation from
    Owen & Wu 2017.'''
    vesc = np.sqrt(2*G*Mearth2g(Mp)/Rearth2cm(Rp))
    eta = 0.1 * (1.5e6/vesc)**2
    return eta




def _tmdot_to_maximize(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Function to compute the maximum mass loss time equal to when 
    Delta_R / Rcore == 1.
    '''
    # compute mass loss timescale from the envelope depth
    _,depth_env = compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    # tmdot is maximized where depth_env equals one
    return depth_env - 1
    


def _Mcore_to_minimize(Mcore, Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice):
    '''
    Function to find the minimum gaseous planet mas for which Xenv < 1.
    '''
    # compute mass loss timescale from the envelope depth
    _,depth_env = compute_tmdot(Xenv_min, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    # tmdot is maximized where depth_env equals one
    return depth_env - 1

