from imports import *
import planetary_structure as ps            


def compute_Xmax_rock(Rp, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the rocky planet that maximizes its
    mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    #TEMPXenv_max = float(fsolve(_tmdot_to_maximizeOLD, .01, args=args))
    
    args = Mcore, Teq, Tkh, Xiron, Xice
    DRmin, DRmax = .1*Rp, 5*Rp
    res = minimize_scalar(_tmdot_to_maximize, bounds=(DRmin,DRmax), args=args, method='bounded')
    DRmax = float(res.x)
    Xenv_max = ps.compute_Xenv(DRmax, *args)

    return Xenv_max



def compute_Xmax_gas(Rp, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the gaseous planet that maximizes 
    its mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    #TEMPXenv_max = fsolve(_tmdot_to_maximize, .01, args=args)
   
    args = Mcore, Teq, Tkh, Xiron, Xice
    DRmin, DRmax = .1*Rp, 5*Rp
    res = minimize_scalar(_tmdot_to_maximize, bounds=(DRmin,DRmax), args=args, method='bounded')
    DRmax = float(res.x)
    Xenv_max = ps.compute_Xenv(DRmax, *args)

    return Xenv_max



def compute_Mcore_min_no_self_gravity(Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice):
    '''Compute the minimum core mass required for Xenv to be < 1 such that 
    atmospheric self gravity can be ignored.'''
    args = Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice
    ##TEMPMcore_min = fsolve(_Mcore_to_minimize, .01, args=args)
    Mcmin, Mcmax = .01, 100
    res = minimize_scalar(_Mcore_to_minimize, bounds=(Mcmin,Mcmax), args=args, method='bounded')
    Mcore_min = res.x
    return float(Mcore_min)


def compute_Mcore_max(Rp_now, Teq, age, Xiron, Xice):
    '''Compute the maximum core mass assuming no envelope (i.e. Rp==Rcore) that 
    still has a radius structure solution.'''
    Mcore_test = ps.solidradius2mass(Rp_now, Xiron, Xice)
    while Mcore_test > 0:
        try:
            _=ps.Rp_solver_gas(Rp_now, Mcore_test, Teq, age, Xiron, Xice)
            return float(Mcore_test)
        except (ValueError, AssertionError):
            Mcore_test *= .99
            continue
    

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
    tmdot = Xenv * Mcore**2 * AU2cm(sma)**2 * eta / Rearth2cm(Rp_full)**3

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
    ##TEMPpdb.set_trace()
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
    # TEMP
    #tmdot_gaseous,_ = compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)
    eta = compute_mass_loss_efficiency(Mcore, Rp_full)
    sma = Ms**(1/3) * (P/365.25)**(2/3)
    tmdot_gaseous = Xenv * Mcore**2 * AU2cm(sma)**2 * eta / Rearth2cm(Rp_full)**3
    ##print(Xenv, Mcore, AU2cm(sma), eta, Rearth2cm(Rp_full), tmdot_gaseous)


    return tmdot_gaseous - tmdot_rocky




def compute_mass_loss_efficiency(Mp, Rp):
    '''Compute the mass loss efficiency using the equation from
    Owen & Wu 2017.'''
    vesc = np.sqrt(2*G*Mearth2g(Mp)/Rearth2cm(Rp))
    eta = 0.1 * (1.5e6/vesc)**2
    return eta




def _tmdot_to_maximizeOLD(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Function to compute the maximum mass loss time equal to when 
    Delta_R / Rcore == 1.
    '''
    # compute mass loss timescale from the envelope depth
    _,depth_env = compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    # tmdot is maximized where depth_env equals one
    return depth_env - 1
    

def _tmdot_to_maximize(DRrcb, Mcore, Teq, Tkh, Xiron, Xice):
    Xenv = ps.compute_Xenv(DRrcb, Mcore, Teq, Tkh, Xiron, Xice)
    _,Rp_full,_ = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron, Xice)
    eta = compute_mass_loss_efficiency(Mcore, Rp_full)
    func_to_min = Rearth2cm(Rp_full)**3 / (Xenv*eta)   # shouldnt eta be in the numerator? apparently not because it fails if it is
    return func_to_min


def _Mcore_to_minimizeOLD(Mcore, Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice):
    '''
    Function to find the minimum gaseous planet mass for which Xenv < 1.
    '''
    # compute mass loss timescale from the envelope depth
    _,depth_env = compute_tmdot(Xenv_min, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    # tmdot is maximized where depth_env equals one
    return depth_env - 1


def _Mcore_to_minimize(Mcore, Xenv_min, P, Ms, Teq, Tkh, Xiron, Xice):
    _,Rp_full,_ = ps.solve_radius_structure(Xenv_min, Mcore, Teq, Tkh, Xiron, Xice)
    eta = compute_mass_loss_efficiency(Mcore, Rp_full)
    func_to_min = Rearth2cm(Rp_full)**3 / (Xenv_min*eta)   # shouldnt eta be in the numerator? apparently not because it fails if it is
    return func_to_min
