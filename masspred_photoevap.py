from imports import *
import planetary_structure as ps


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
        tmdotmax_rocky,_ = compute_tmdot(Xenv_rocky, *args)
        
    else:
        try:
            Xenv_rocky = compute_Xmax_rock(*args)
            # compute rocky planet mass loss timescale
            tmdotmax_rocky,_ = compute_tmdot(Xenv_rocky, *args)
            
        except (ValueError, AssertionError):
            return np.nan


    # get minimum gaseous core mass for which Xenv < 1
    args = [_sample(tps.planet_gaseous.Psamples), \
            _sample(tps.star.Mssamples), \
            _sample(tps.planet_gaseous.Teqsamples), \
            _sample(tps.planet_gaseous.Tkhsamples), \
            _sample(tps.planet_gaseous.Xironsamples), \
            _sample(tps.planet_gaseous.Xicesamples)]
    Xenv_min = 1e-8
    Mcore_gas_min = compute_Mcore_min_no_self_gravity(Xenv_min, *tuple(args))
    
    
    # check that a solution exists
    # i.e. the gaseous planet's maximum tmdot > rocky planet's maximum tmdot
    # increase minimum gas planet core mass until its maxmimum tmdot exceeds
    # that of the rocky planet
    Mcore_gas_min /= 1.1
    args = list(np.insert(args, 2, Mcore_gas_min))
    tmdotmax_gas = 0
    while (tmdotmax_gas < tmdotmax_rocky) & \
          (Mcore_gas_min < tps.planet_gaseous.mass[0]):

        # increase minimum gas planet core mass
        Mcore_gas_min *= 1.1
        args[2] = Mcore_gas_min

        if value_errors:
            Xenv_gas = compute_Xmax_gas(*tuple(args))
            # compute gaseous planet mass loss timescale
            tmdotmax_gas,_ = compute_tmdot(Xenv_gas, *tuple(args))

        else:
            try:
                Xenv_gas = compute_Xmax_gas(*tuple(args))
                # compute gaseous planet mass loss timescale
                tmdotmax_gas,_ = compute_tmdot(Xenv_gas, *tuple(args))
            except (ValueError, AssertionError):
                return np.nan

     
    # ensure that a solution exists
    if (tmdotmax_gas < tmdotmax_rocky) | \
       (Mcore_gas_min > tps.planet_gaseous.mass[0]):
        raise ValueError("No solution exists because the gaseous planet's maximum mass loss timescale is less than the rocky planet's maximum mass loss timescale.")


    # just solved for minimum gaseous core mass to have a longer mass loss time
    # than the rocky planet
    # set maximum gaseous core mass for a pure iron ball
    Rp_gas = _sample(tps.planet_gaseous.rpsamples)
    Mcore_gas_max = float(ps.solidradius2mass(Rp_gas, 1, 0))
    
    # solve for the minimum gaseous planet mass
    args = _sample(tps.planet_gaseous.Psamples), \
        _sample(tps.star.Mssamples), \
        _sample(tps.planet_gaseous.Teqsamples), \
        _sample(tps.planet_gaseous.Tkhsamples), \
        _sample(tps.planet_gaseous.Xironsamples), \
        _sample(tps.planet_gaseous.Xicesamples), \
        _sample(tps.planet_gaseous.rpsamples), \
        _sample(tps.star.agesamples), \
        tmdotmax_rocky
    if value_errors:
        Mgas_min = 10**brentq(_Mp_gas_to_solve,
                              np.log10(Mcore_gas_min),
                              np.log10(Mcore_gas_max),
                              args=args)
        return Mgas_min

    else:
        try:
            Mgas_min = 10**brentq(_Mp_gas_to_solve,
                                  np.log10(Mcore_gas_min),
                                  np.log10(Mcore_gas_max),
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
    #results = minimize(_tmdot_to_maximize, [-2], args=args,
    #                   bounds=Bounds(-8,0), method='L-BFGS-B')
    Xenv_max = fsolve(_tmdot_to_maximize, .01, args=args)
    return float(Xenv_max)




def compute_Xmax_gas(P, Ms, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Solve for the envelope mass fraction of the gaseous planet that maximizes 
    its mass loss timescale given the planetary parameters.
    '''
    # maximize the rocky planet's mass loss timescale
    args = P, Ms, Mcore, Teq, Tkh, Xiron, Xice
    #results = minimize(_tmdot_to_maximize, [-2], args=args,
    #                   bounds=Bounds(-6,-1e-4), method='L-BFGS-B')
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




def _sample(samples, N=1):
    '''Sample randomly from an input distribution.'''
    return np.random.choice(samples) if N == 1 \
        else np.random.choice(samples, size=int(N))
