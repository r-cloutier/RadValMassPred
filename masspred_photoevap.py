from imports import *
import planetary_structure as ps


# step
# 1) solve for X by maximizing the ratio X/(eta*Rp). This Xmax also maximizes
# the rocky planet tmdot which we need



def compute_Mgas_min(tps, value_errors=True):
    '''
    Compute the minimum mass of the gaseous planet in order to be consistent 
    with the photoevaporation scenerio. 

    The minimum gaseous planet mass comes from equating the maximum mass loss 
    time of the rocky planet (i.e. it just lost its primordial H/He envelope)
    to the minimum mass loss timescale for the gaseous planet.
    '''
    assert tps._is_complete  # system is setup for calculations 
    
    # solve for Xenv that maximizes the rocky planet mass loss timescale
    args = _sample(tps.planet_rocky.Psamples), \
        _sample(tps.star.Mssamples), \
        _sample(tps.planet_rocky.mpsamples), \
        _sample(tps.planet_rocky.Teqsamples), \
        _sample(tps.planet_rocky.Tkhsamples), \
        _sample(tps.planet_rocky.Xironsamples), \
        _sample(tps.planet_rocky.Xicesamples)

    if value_errors:
        Xenv_rocky = compute_Xmax_rock(*args)
        # compute rocky planet mass loss timescale
        tmdot_rocky = compute_tmdot(Xenv_rocky, *args)
        
    else:
        try:
            Xenv_rocky = compute_Xmax_rock(*args)
            # compute rocky planet mass loss timescale
            tmdot_rocky = compute_tmdot(Xenv_rocky, *args)
            
        except (ValueError, AssertionError):
            return np.nan


    # set limits on Mcore
    Rp_gas = _sample(tps.planet_gaseous.rpsamples)
    Mcore_min = float(ps.solidradius2mass(.1*Rp_gas, 0, 1))
    Mcore_max = float(ps.solidradius2mass(Rp_gas, 1, 0))
    
    # solve for the minimum gaseous planet mass
    args = _sample(tps.planet_gaseous.Psamples), \
        _sample(tps.star.Mssamples), \
        _sample(tps.planet_gaseous.Teqsamples), \
        _sample(tps.planet_gaseous.Tkhsamples), \
        _sample(tps.planet_gaseous.Xironsamples), \
        _sample(tps.planet_gaseous.Xicesamples), \
        _sample(tps.planet_gaseous.rpsamples), \
        _sample(tps.star.agesamples), \
        tmdot_rocky
    if value_errors:
        Mgas_min = brentq(_Mp_gas_to_solve,
                          np.log10(Mcore_min),
                          np.log10(Mcore_max),
                          args=args)
        return Mgas_min

    else:
        try:
            Mgas_min = brentq(_Mp_gas_to_solve,
                              np.log10(Mcore_min),
                              np.log10(Mcore_max),
                              args=args)
            return Mgas_min
        except (ValueError, AssertionError):
            return np.nan
            


def compute_Xmax_rock(P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the rocky planet that maximizes its
    mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    results = minimize(_tmdot_to_maximize, [-2], args=args, bounds=Bounds(-8,0),
                       method='L-BFGS-B')

    # return Xenv if optimization was successful
    if results.success and np.isfinite(results.fun):
        Xenv_max = 10**float(results.x)
        return Xenv_max

    else:
        print(results)
        raise ValueError("Unable to maximize the rocky planet's mass loss timescale (minimize output printed above).")




def _tmdot_to_maximize(lg_Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Function to compute the inverse mass loss timescale for the purpose of 
    maximization.
    '''
    # compute mass loss timescale using Eq 20 from Owen & Wu 2017
    tmdot = compute_tmdot(10**lg_Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    # return inverse tmdot for use in scipy.minimize
    return 1/tmdot    



def compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    assert 0 <= Xenv <= 1  # avoid self-gravitating envelope when Menv>Mcore
    
    # solve planetary structure
    Rcore = ps.mass2solidradius(Mcore, Xiron, Xice)
    Rrcb, Rp_full, f = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh,
                                                 Xiron, Xice)
    depth_env = (Rp_full-Rcore) / Rcore
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
    
    return tmdot



def _Mp_gas_to_solve(lg_Mcore, P, Ms, Teq, Tkh, Xiron, Xice, Rp_now, age,
                     tmdot_rocky):
    '''
    Objective function that equates the gaseous planet's mass loss timescale, 
    given its mass, to the input maximum mass loss timescale of the rocky
    planet.
    '''
    # evaluate the current envelope mass fraction
    Mcore = 10**lg_Mcore
    Xenv, Rp_full = ps.Rp_solver(Rp_now, Mcore, Teq, age, Xiron, Xice)

    print(Rp_full, Rp_now)
    if (np.fabs(Rp_full-Rp_now)/Rp_now > 1e-4):
        #np.isclose(float(Rp_full), Rp_now, rtol=1e-4):
        raise ValueError('No solution for the gaseous planet found.')

    # check for consistent X
    Rcore = ps.mass2solidradius(Mcore, Xiron, Xice)
    Rrcb, Rp_full, f = ps.solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron,
                                                 Xice)
    DRrcb = Rrcb - Rcore
    Xtest = ps.compute_Xenv(DRrcb, Mcore, Teq, Tkh, Xiron, Xice)
    
    if (np.fabs(Xtest-Xenv)/Xenv > 1e-4):
        raise ValueError('No self-consistent structure.')

    # compute gaseous planet mass loss timescale
    tmdot_gaseous = compute_tmdot(Xenv, P, Ms, Mcore, Teq, Tkh, Xiron, Xice)

    return tmdot_gaseous - tmdot_rocky




def compute_mass_loss_efficiency(Mp, Rp):
    '''Compute the mass loss efficiency using the equation from
    Owen & Wu 2017.'''
    vesc = np.sqrt(2*G*Mearth2g(Mp)/Rearth2cm(Rp))
    eta = 0.1 * (1.5e6/vesc)**2
    return eta



def _sample(samples, N=1):
    '''Sample randomly from an input distribution.'''
    return np.random.choice(samples) if N == 1 \
        else np.random.choice(samples, size=int(N))
