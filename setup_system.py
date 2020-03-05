from imports import *

global Nsamp
Nsamp = int(1e3)


class two_planet_system:

    def __init__(self, radval_func, label='syst00'):
        '''
        Initialize a planetary system that should contain two planets 
        that span the radius valley as defined in the input function 
        radval_func.
        '''
        self.label = label
        self.radval_func = radval_func
        self.star, self.planet_rocky, self.planet_gaseous = None, None, None
        self.Nplanets = 0

        

    def add_star(self, Mssamples, Rssamples, Teffsamples, label=None):
        '''
        Add the host star given the marginalized posterior distributions 
        of its mass, radius, and effective temperature.
        '''
        starlabel = '%s_star'%self.label if label == None else label 
        self.star = star(starlabel, Mssamples, Rssamples, Teffsamples)


        
    def add_planet(self, Psamples, rpsamples, label=None, albedo=0.3,
                   radval_args=[]):
        '''
        Add a planet to the system given the marginalized posterior 
        distributions of its orbital period [days] and radius [Earth radii].
        '''
        planetlabel = '%s_planet%i'%(self.label, self.Nplanets) \
            if label == None else label
        p = planet(planetlabel, Psamples, rpsamples,
                   self.radval_func(*radval_args), albedo)
        
        # derive planet parameters if the host star is defined
        if self.star != None:
            p._compute_planet_params(self.star)
            
        # save as a rocky or gaseous planet
        if p.is_rocky:
            self.planet_rocky = p
        else:
            self.planet_gaseous = p

        # get number of planets as either 0, 1, or 2
        self.Nplanets = np.sum(np.array([self.planet_rocky,
                                         self.planet_gaseous]) != None)
            
            

class star:

    def __init__(self, label, Mssamples, Rssamples, Teffsamples):
        '''
        Initialize the host star.
        '''
        self.label = label
        self._define_stellar_params(Mssamples, Rssamples, Teffsamples)



    def _define_stellar_params(self, Mssamples, Rssamples, Teffsamples):
        '''Define stellar parameter point estimates and uncertainties.'''
        self.Mssamples = np.repeat(Mssamples, Nsamp) \
            if type(Mssamples) in [int,float] else \
               np.ascontiguousarray(Mssamples)
        self.Rssamples = np.repeat(Rssamples, Nsamp) \
            if type(Rssamples) in [int,float] else \
               np.ascontiguousarray(Rssamples)
        self.Teffsamples = np.repeat(Teffsamples, Nsamp) \
            if type(Teffsamples) in [int,float] else \
               np.ascontiguousarray(Teffsamples)

        self.mass = compute_point_estimates(self.Mssamples)
        self.radius = compute_point_estimates(self.Rssamples)
        self.teff = compute_point_estimates(self.Teffsamples)
        


class planet:

    def __init__(self, label, Psamples, rpsamples, radius_transition, albedo):
        '''
        Initialize one planet. 
        '''
        self.label = label
        self._radius_transition = float(radius_transition)
        self.albedo = float(albedo)
        self._define_planetary_params(Psamples, rpsamples)



    def _define_planetary_params(self, Psamples, rpsamples):
        self.Psamples = np.repeat(Psamples, Nsamp) \
            if type(Psamples) in [int,float] else \
               np.ascontiguousarray(Psamples)
        self.rpsamples = np.repeat(rpsamples, Nsamp) \
            if type(rpsamples) in [int,float] else \
               np.ascontiguousarray(rpsamples)
        
        self.period = compute_point_estimates(self.Psamples)
        self.radius = compute_point_estimates(self.rpsamples)

        self._check_if_rocky()


    def _check_if_rocky(self):
        '''
        Return True if the planet lies beneath the radius valley.
        Otherwise, return False.
        '''
        self.is_rocky = self.radius[0] < self._radius_transition


    def _compute_planet_params(self, star):
        '''Compute stellar-dependent planetary parameters.'''
        self.asamples = star.Mssamples**(1/3) * \
            conv.days2yrs(self.Psamples)**(2/3)
        self.Teqsamples = (1-self.albedo)**(.25) * star.Teffsamples * \
            np.sqrt(conv.Rsun2m(star.Rssamples) / (2*conv.AU2m(self.asamples)))

        self.a = compute_point_estimates(self.asamples)
        self.teq = compute_point_estimates(self.Teqsamples)




def compute_point_estimates(samples):
    v = np.percentile(samples, (16,50,84))
    return v[1], v[2]-v[1], v[1]-v[0]
