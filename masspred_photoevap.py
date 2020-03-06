from imports import *
import planetary_structure as ps


def calculate_min_mp_gaseous(tps, mp_rocky):
    '''
    Calculate the minimum mass of the gaseous planet required to be consistent
    with the photoevaporation mechanism.
    
    Parameters
    ----------
    tps : class setup_system.two_planet_system
        Two_planet_system object containing two planets spanning the radius 
        valley.
    '''
    assert tps.Nplanets == 2
    assert tps.planet_rocky != None
    assert tps.planet_gaseous != None

    t_young = 100 Myr
    age


    # solve for rocky planet's envelope mass fraction that maximums its mass
    # loss timescale
    args = mp_rocky, tps.planet_rocky.radius
    results = minimize(ratio_to_minimize, .01, args=args, bounds=(0,1))
    Xenv = float(results['x'])
    


# tmdot_structure()
def tmdot_to_maximize(DRrcb, Teq, Mcore, Tkh_Myr, Xiron, Xice):
    '''
    Function to compute the ratio eta*rp^3/X which must be minimized to find
    the envelope mass fraction X that results in the maximum mass loss 
    timescale for the rocky planet. See Eq 5 in Owen & Estrada 2020
    (https://arxiv.org/abs/1912.01609)
    '''
    X, f, Rplanet = ps.compute_X(DeltaRrcb, Teq, Mcore, Tkh_Myr, Xiron, Xice)
    eta = ps.compute_efficiency(Mcore, Rplanet)
    ratio_to_min = Rplanet**3 / (X * eta)   # should this be rp3*eta/X??
    return ratio_to_min

