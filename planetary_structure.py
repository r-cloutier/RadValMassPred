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
        rho0 = 1 / mass2rad_iron(1, Xiron, 0)**3
        mp0 = rho0 * rp**3
        return fsolve(mass2rad_iron, mp0, args=(Xiron, rp))

    else:
        rho0 = 1 / mass2rad_ice(1, Xice, 0)**3
        mp0 = rho0 * rp**3
        return fsolve(mass2rad_ice, mp0, args=(Xice, rp))



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
        return mass2rad_iron(mp, Xiron)
    else:
        return mass2rad_ice(mp, Xice)    



def mass2rad_iron(mp, Xiron, rp=0):
    '''Compute the difference between the radius and the predicted radius
    of solid rocky/iron body given its mass. Need the difference to solve for
    the root of the quadratic.'''
    Xrock = 1 - Xiron
    rpfunc = np.poly1d([0.0592 * Xrock + 0.0975,
                        0.2337 * Xrock + 0.4938,
                        0.3102 * Xrock + 0.7932])
    return rpfunc(np.log10(mp)) - rp


def mass2rad_ice(mp, Xice, rp=0):
    '''Compute the difference between the radius and the predicted radius
    of solid icy/rocky body given its mass.'''
    rpfunc = np.poly1d([0.0912 * Xice + 0.1603,
                        0.3330 * Xice + 0.7378,
                        0.4639 * Xice + 1.1193])
    return rpfunc(np.log10(mp)) - rp


def integrand1(x, gamma):
    return x * (1/x-1)**(1/(gamma-1))


def integrand2(x, gamma):
    return x**2 * (1/x-1)**(1/(gamma-1))


def compute_I2_I1(DR_ratio):
    DR_ratio = np.ascontiguousarray(DR_ratio)
    I2, ratio = np.zeros(DR_ratio.size), np.zeros(DR_ratio.size)
    for i in range(ratio.size):
        I2[i] = quad(integrand2, 1/(DR_ratio[i]+1), 1, args=gamma)[0]
        I1 = quad(integrand1, 1/(DR_ratio[i]+1), 1, args=gamma)
        ratio[i] = I2[i]/I1[0]
    return I2, ratio



def compute_rho_rcb(DRrcb, Xenv, Mcore, Teq, Tkh_Myr, Xiron, Xice):
    '''
    Compute the density of at the radiative-convective boundary. Eq 13 
    in Owen & Wu 2017 (https://arxiv.org/abs/1705.10810).
    '''
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    Rrcb = DRrcb + Rcore
    DR_ratio = DRrcb / Rcore
    _,I2_I1 = compute_I2_I1(DR_ratio)
    
    # Eq 13 in Owen & Wu 2017 for the RCB density
    a = 1 / (1+alpha)
    rho_rcb = (mu / Kb)
    rho_rcb *= I2_I1**a
    rho_rcb *= (64*np.pi*sigma * Teq**(3-alpha-beta) * \
                Rearth2cm(Rrcb) * yrs2sec(Tkh_Myr*1e6))**a
    rho_rcb *= (3*kappa0*Mearth2g(Mcore)*Xenv)**(-a)
    return rho_rcb


def compute_mass_loss_efficiency(mp, rp):
    # mass loss efficiency from Owen & Wu 2017
    vesc = np.sqrt(2*G*Mearth2g(mp)/Rearth2cm(rp))
    eta = 0.1 * (1.5e6/vesc)**2
    return eta


def compute_X(DRrcb, Teq, Mcore, Tkh_Myr, Xiron, Xice):
    '''
    Compute the envelope mass fraction at the radiative-convective boundary
    below which the entirety of the atmosphere is assumed to exist.
    '''
    # evaluate the core
    Rcore = mass2solidradius(Mcore, Xiron, Xice)
    rhocore = 3*Mearth2g(Mcore) / (4*np.pi*Rearth2cm(Rcore)**3)  # g/cm3 

    # evaulate the RCB
    cs2 = Kb*Teq / mu
    Rrcb = DRrcb + Rcore
    DR_ratio = DRrcb / Rcore

    # compute integrals
    I2, I2_I1 = compute_I2_I1(DR_ratio)

    # Eq 13 in Owen & Wu 2017 for the RCB density but without the X factor
    a = 1 / (1+alpha)
    rho_rcb_noX = (mu / Kb)
    rho_rcb_noX *= I2_I1**a
    rho_rcb_noX *= (64*np.pi*sigma * Teq**(3-alpha-beta))**a
    rho_rcb_noX *= (Rearth2cm(Rrcb) * yrs2sec(Tkh_Myr*1e6))**a
    rho_rcb_noX *= (3*kappa0 * Mearth2g(Mcore))**(-a)

    # evaluate the envelope mass fraction
    LHS = 3*(Rrcb/Rcore)**3 * rho_rcb_noX / rhocore
    LHS *= (grad_ab * G*Mearth2g(Mcore)/(cs2*Rearth2cm(Rrcb)))**(1/(gamma-1))
    LHS *= I2
    
    Xenv = LHS**(1 / (1+1/(1+alpha)))

    # compute the full planet radius w/ radiative zone over the rcb
    rho_rcb = compute_rho_rcb(DRrcb, Xenv, Mcore, Teq, Tkh_Myr,
                              Xiron, Xice)
    
    # evaluate the photosphere
    pressure_phot = (2/3 * \
                     (G*Mearth2g(Mcore) / \
                      (Rearth2cm(Rrcb)**2 * kappa0*Teq**beta)))**(1/(1+alpha))
    rho_phot = pressure_phot*mu / (Kb*Teq)

    # compute the full planet radius
    H = cs2 * Rearth2cm(Rrcb)**2 / (G*Mearth2g(Mcore))
    f = 1 + (H/Rearth2cm(Rrcb)) * np.log(rho_rcb / rho_phot)
    Rplanet = f*Rrcb
    return Xenv, f, Rplanet
