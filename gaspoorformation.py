from imports import *
import planetary_structure as ps            



def solve_taccrete_gas(Xenv, Teq, Mcore):
    '''
    Solve for the timescale to accrete the input envelope mass fraction of the
    gaseous planet. To be compared to the disk lifetime to test feasibility.
    '''
    # compute the accretion time to reach an envelope mass fraction
    # (Eq 24 Lee & Chiang 2015) 
    tacc_Myr = 1e-3 * ((Xenv/.1) * (Teq/200)**1.5 * (5/Mcore))**2.5
    return tacc_Myr
    
    

def solve_Xiron_rocky(Mp_now, Rp_now):
    '''
    Assuming that any residual gaseous envelope does not contribute to the
    rocky planet's size, caluculate the minimum iron mass fraction to be 
    consistent with the observed mass and radius. Assumes no water ice
    contribution.
    '''
    Xiron = 10**brentq(_Xiron_rock, -4, 0, args=(Mp_now, Rp_now))
    return Xiron



def _Xiron_rock(lg_Xiron, Mp_now, Rp_now):
    Xiron = 10**lg_Xiron
    Rcore = ps.mass2solidradius(Mp_now, Xiron, 0)
    return 1e3 if Rcore > Rp_now else Rcore - Rp_now



def compute_Mgas_min(Mp_rock, a_rock, a_gas):
    '''
    Compute the minimum mass of the gaseous planet in order to be consistent
    with the gas-poor formation scenerio.
    '''
    return Mp_rock * (a_rock/a_gas)**(.6)



def sample_disk_lifetime(tau=3, tmax_Myr=20):
    '''Sample a disk lifetime from the frequency distribution of disks.'''
    tarr = np.linspace(0,tmax_Myr,1000)
    dt = np.diff(tarr)[0]
    tarr += np.random.uniform(-5.*dt, .5*dt, tarr.size)
    tarr = tarr[tarr>=0]
    f_without_disk = 1 - np.exp(-tarr / tau)
    tdisk = np.random.choice(tarr, p=f_without_disk/f_without_disk.sum())
    return float(tdisk)
