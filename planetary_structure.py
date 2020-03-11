from imports import *


def solidradius2mass(rp, Xiron, Xice):
    '''
    Convert an input radius to into a mass assuming a solid planet using the
    MR-relation from Fortney et al 2007
    (https://arxiv.org/abs/astro-ph/0612671).
    '''
    # cannot use iron and ice together
    assert ((Xiron>0) & (Xice==0)) | ((Xiron==0) & (Xice>0))

    # solve for mass given radius by solving for the root of the MR functions
    if (Xiron > 0):
        rho0 = 1 / _mass2rad_iron(1, Xiron, 0)**3
        mp0 = rho0 * rp**3
        return fsolve(_mass2rad_iron, mp0, args=(Xiron, rp))

    else:
        rho0 = 1 / _mass2rad_ice(1, Xice, 0)**3
        mp0 = rho0 * rp**3
        return fsolve(_mass2rad_ice, mp0, args=(Xice, rp))


    

def mass2solidradius(mp, Xiron, Xice):
    '''
    Convert an input mass to into a radius assuming a solid planet using the
    MR-relation from Fortney et al 2007
    (https://arxiv.org/abs/astro-ph/0612671).
    '''
    # cannot use iron and ice together
    assert ((Xiron>0) & (Xice==0)) | ((Xiron==0) & (Xice>0))

    # solve for mass given radius by solving for the root of the MR functions
    if (Xiron > 0):
        return _mass2rad_iron(mp, Xiron)
    else:
        return _mass2rad_ice(mp, Xice)



    
def compute_rhorcb(Xenv, Rrcb, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Compute the density at the radiative-convective boundary using Eq 13 from 
    Owen & Wu 2017.
    '''
    # compute the ratio I2/I1
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    DR = Rrcb - Rcore
    assert DR >= 0    # the rcb must be above the core
    depth_env = DR/Rcore
    I1, I2 = compute_I1_I2(depth_env)
    I2_I1 = I2/I1

    # compute the density of the rcb
    a = 1/(1+alpha)
    rhorcb = (mu/Kb) * I2_I1**a
    rhorcb *= (64*np.pi*sigma * Teq**(3-alpha-beta) * Rearth2cm(Rrcb)*Myrs2sec(Tkh))**a
    rhorcb *= (3*kappa0 * Mearth2g(Mcore)*Xenv)**(-a)
 
    return float(rhorcb)




def solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Given an envelope mass fraction, calculate the radius of the 
    radiative-convective boundary and the full planet radius that includes the
    isothermal layers above the rcb.
    '''
    Rcore = mass2solidradius(Mcore, Xiron, Xice)    

    # estimate DR = Rrcb-Rcore from Eq 17 Owen & Wu 2017
    DR_guess = 2*Rcore * (Xenv/0.027)**(1/1.31) * (Mcore/5)**(-0.17)
    # solve for Rrcb
    args = Xenv, Mcore, Teq, Tkh, Xiron, Xice
    DR_solution = 10**float(fsolve(_solve_Rrcb, np.log10(DR_guess), args=args))
    Rrcb = DR_solution + Rcore

    # compute the full planet radius
    rho_rcb = compute_rhorcb(Xenv, Rrcb, Mcore, Teq, Tkh, Xiron, Xice)
    Rp_full, f = compute_full_Rp(Rrcb, Mcore, Teq, rho_rcb)

    return Rrcb, Rp_full, f




def compute_full_Rp(Rrcb, Mcore, Teq, rho_rcb):
    '''
    Compute the full planet radius that includes isothermal layers above 
    the rcb.
    '''
    # now calculate the densities at the photosphere
    a = 1/(1+alpha)
    _,rho_phot = compute_photosphere(Mcore, Rrcb, Teq)
    assert rho_phot <= rho_rcb
    
    # calculate the fractional difference between the rcb and photospheric radii
    cs2 = Kb * Teq / mu
    H = cm2Rearth(cs2 * Rearth2cm(Rrcb)**2 / (G * Mearth2g(Mcore)))
    f = float(1 + (H/Rrcb) * np.log(rho_rcb / rho_phot))
    assert f >= 1
    Rp_full = f*Rrcb

    return Rp_full, f



def compute_photosphere(Mp, Rp, Teq):
    '''Compute the pressure and density at the photosphere in cgs units.'''
    a = 1/(1+alpha)
    g = G * Mearth2g(Mp) / Rearth2cm(Rp)**2
    pressure_phot = (2*g / (3 * kappa0 * Teq**beta))**a  # cgs
    rho_phot = pressure_phot * mu / (Kb * Teq)  # cgs
    return pressure_phot, rho_phot



def compute_I1_I2(depth_env):
    '''depth_env = (R1 - R2) / R2'''
    depth_env = np.ascontiguousarray(depth_env)
    I1, I2 = np.zeros(depth_env.size), np.zeros(depth_env.size)
    for i in range(I1.size):
        I1[i] = quad(_integrand1, 1/(depth_env[i]+1), 1, args=gamma)[0]
        I2[i] = quad(_integrand2, 1/(depth_env[i]+1), 1, args=gamma)[0]
    return I1, I2



def compute_Xenv(DRrcb, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Given the difference between rcb radius and the core radius, compute the
    envelope mass fraction from Eqs 4 and 13 in Owen & Wu 2017. 
    '''
    assert DRrcb >= 0
    
    # setup variables
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    rho_core = 3*Mearth2g(Mcore) / (4*np.pi*Rearth2cm(Rcore)**3)
    cs2 = Kb * Teq / mu

    Rrcb = DRrcb + Rcore
    depth_env = DRrcb / Rcore

    # dimensionless integrals
    I1, I2 = compute_I1_I2(depth_env)
    I2_I1 = I2/I1

    # compute a proxy of the density at the rcb
    a = 1/(1+alpha)   
    rho_rcb_without_X_term = (mu/Kb) * I2_I1**a
    rho_rcb_without_X_term *= (64*np.pi*sigma*Teq**(3-alpha-beta))**a
    rho_rcb_without_X_term *= (Rearth2cm(Rrcb) * Myrs2sec(Tkh))**a
    rho_rcb_without_X_term /= (3*kappa0 * Mearth2g(Mcore))**a

    # evaluate the envelope mass fraction using Eqs 4 & 13 from Owen & Wu 2017
    b = 1/(gamma-1)
    RHS = 4*np.pi*Rearth2cm(Rrcb)**3 * rho_rcb_without_X_term
    RHS *= (grad_ab * G*Mearth2g(Mcore) / (cs2*Rearth2cm(Rrcb)))**b
    RHS *= I2
    RHS /= Mearth2g(Mcore)
    Xenv = float(RHS**(1/(1+1/(1+alpha))))
    
    return Xenv



def compute_Xenv_rad(Rp, Mcore, Teq, age_Myr, Xiron, Xice):
    '''
    Compute the envelope mass fraction in a fully radiative layer. This should
    only be used when the radius solver finds an rcb that is shallower than 
    the atmospheric scale height. 
    '''
    # check that no convective layer is present
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    lg_DRrcb0 = np.log10(Rp - Rcore)
    args = Rp, Teq, Mcore, age_Myr, Xiron, Xice
    DRrcb = 10**fsolve(_Rp_solver_function, lg_DRrcb0, args=args)
    cs2 = Kb * Teq / mu
    H = cm2Rearth(cs2 * Rearth2cm(Rp)**2 / (G * Mearth2g(Mcore)))
    if DRrcb > H:
        raise ValueError('A convective layer is present. Should use compute_Xenv instead of compute_Xenv_rad.')
    
    # setup variables
    _,rho_phot = compute_photosphere(Mcore, Rp, Teq)
    cs2 = Kb * Teq / mu
    H = cm2Rearth(cs2 * Rearth2cm(Rp)**2 / (G * Mearth2g(Mcore)))

    # calculate the mass in the radiative layer
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    if Rcore > Rp:
        raise ValueError('Core radius exceeds the planet radius. Try increasing the mass fraction of dense material (i.e. iron compared to MgS rock of rock compared to water ice.')

    rho_surf = rho_phot * np.exp((Rp-Rcore) / H)
    if rho_surf < rho_phot:
        raise ValueError('Atmospheric density at the surface is less than the photospheric density.')

    # integrate from zero -> Rp-Rcore to get mass within the radiative region
    args = rho_surf, Rearth2cm(Rcore), Rearth2cm(H)
    Menv_int = quad(_Menv_integrand_cgs, Rearth2cm(Rcore), Rearth2cm(Rp),
                    args=args)[0]

    Xenv = Menv_int / Mcore

    return Xenv



def Rp_solver(Rp_now, Mcore, Teq, age_Myr, Xiron, Xice):
    '''
    Solve for the planet structure of the gaseous planet to be consistent with 
    the observed radius.
    '''
    # 
    Rcore = mass2solidradius(Mcore,Xiron,Xice)
    if Rcore > Rp_now:
        raise ValueError('Core radius exceeds the planet radius. Try increasing the mass fraction of dense material (i.e. iron compared to MgS rock of rock compared to water ice.')
    
    lg_DRrcb0 = np.log10(Rp_now - Rcore)
    args = Rp_now, Teq, Mcore, age_Myr, Xiron, Xice
    lg_DRrcb = fsolve(_Rp_solver_function, lg_DRrcb0, args=args)
    
    # now evaluate planet structure
    DRrcb = 10**lg_DRrcb
    cs2 = Kb * Teq / mu
    H = cm2Rearth(cs2 * Rearth2cm(Rp_now)**2 / (G * Mearth2g(Mcore)))

    if (DRrcb < H):  # i.e. no convective zone
        Xenv = compute_Xenv_rad(Rp_now, Mcore, Teq, age_Myr, Xiron, Xice)
        _,Rp_full,_ = solve_radius_structure(Xenv, Mcore, Teq, age_Myr,
                                             Xiron, Xice)
        
    else:
        Xenv = compute_Xenv(DRrcb, Mcore, Teq, age_Myr, Xiron, Xice)
        _,Rp_full,_ = solve_radius_structure(Xenv, Mcore, Teq, age_Myr,
                                             Xiron, Xice)

    return Xenv, Rp_full




def _Rp_solver_function(lg_DRrcb, Rp_now, Teq, Mcore, Tkh, Xiron, Xice):
    '''Function to match the planet radius solution to the observed radius.'''
    Xenv = compute_Xenv(10**lg_DRrcb, Mcore, Teq, Tkh, Xiron, Xice)
    Rrcb,Rp_full,_ = solve_radius_structure(Xenv, Mcore, Teq, Tkh, Xiron, Xice)
    return Rp_now - Rp_full




def _mass2rad_iron(mp, Xiron, rp=0):
    '''Compute the difference between the radius and the predicted radius
    of solid rocky/iron body given its mass. Need the difference to solve for
    the root of the quadratic.'''
    Xrock = 1 - Xiron
    rpfunc = np.poly1d([0.0592 * Xrock + 0.0975,
                        0.2337 * Xrock + 0.4938,
                        0.3102 * Xrock + 0.7932])
    return rpfunc(np.log10(mp)) - rp




def _mass2rad_ice(mp, Xice, rp=0):
    '''Compute the difference between the radius and the predicted radius
    of solid icy/rocky body given its mass.'''
    rpfunc = np.poly1d([0.0912 * Xice + 0.1603,
                        0.3330 * Xice + 0.7378,
                        0.4639 * Xice + 1.1193])
    return rpfunc(np.log10(mp)) - rp




def _Menv_integrand_cgs(r, rho_surf_cgs, Rcore_cgs, H_cgs):
    K = 4*np.pi*rho_surf_cgs
    return g2Mearth(K * np.exp(-(r-Rcore_cgs) / H_cgs) * r**2)



def _solve_Rrcb(lg_DR, Xenv, Mcore, Teq, Tkh, Xiron, Xice):
    '''
    Function from Eqs 4 and 13 in Owen & Wu 2017 to solve for the radius 
    difference between the rcb and the core.
    '''
    # recover Rrcb
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    depth_env = 10**lg_DR / Rcore
    Rrcb = 10**lg_DR + Rcore
    # get density at the rcb
    rho_rcb = compute_rhorcb(Xenv, Rrcb, Mcore, Teq, Tkh, Xiron, Xice)

    # compute remaining values
    rho_core = 3*Mearth2g(Mcore) / (4*np.pi*Rearth2cm(Rcore)**3)
    _,I2 = compute_I1_I2(depth_env)
    cs2 = Kb * Teq / mu

    # estimate envelope mass fraction given the rcb properties
    # Eqs 4 Owen & Wu 2017
    Xguess = 3*(Rrcb / Rcore)**3
    Xguess *= (rho_rcb / rho_core)
    Xguess *= (grad_ab * (G*Mearth2g(Mcore))/(cs2*Rearth2cm(Rrcb)))**(1/(gamma-1))
    Xguess *= I2
 
    return Xguess - Xenv



def _integrand1(x, gamma):
    return x * (1/x-1)**(1/(gamma-1))


def _integrand2(x, gamma):
    return x**2 * (1/x-1)**(1/(gamma-1))
