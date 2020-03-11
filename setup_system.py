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

        

    def add_star(self, Ms, Rs, Teff, age=5e3, label=None):
        '''
        Add the host star given its mass, radius, and effective temperature in 
        the form of a scalar value (i.e. zero uncertainty) or as a numpy.array
        of samples from the parameter's marginalized posterior.
        '''
        starlabel = '%s_star'%self.label if label == None else label 
        self.star = star(starlabel, Ms, Rs, Teff, age)


        
    def add_planet(self, P, rp, mp=None, Xiron=.33, Xice=0, Tkh=100,
                   radval_args=[], label=None, albedo=0.3):
        '''
        Add a planet to the system given its orbital period [days] and radius
        [Earth radii] in the form of a scalar value (i.e. zero uncertainty) or 
        as a numpy.array of samples from the parameter's marginalized posterior.
        '''
        planetlabel = '%s_planet%i'%(self.label, self.Nplanets) \
            if label == None else label
        p = planet(planetlabel, P, rp, mp, Xiron, Xice, Tkh,
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

        # check if system is complete
        # (i.e. has planets on either side of the valley)
        self._is_complete = (self.Nplanets == 2) & (self.star != None) & \
            (self.planet_rocky != None) & (self.planet_gaseous != None)
        

        

class star:

    def __init__(self, label, Ms, Rs, Teff, age):
        '''
        Initialize the host star.
        '''
        self.label = label
        self._define_stellar_params(Ms, Rs, Teff, age)



    def _define_stellar_params(self, Ms, Rs, Teff, age):
        '''Define stellar parameter point estimates and uncertainties.'''
        self.Mssamples = np.repeat(Ms, Nsamp) if type(Ms) in [int,float] else \
               np.ascontiguousarray(Ms)
        self.Rssamples = np.repeat(Rs, Nsamp) if type(Rs) in [int,float] else \
               np.ascontiguousarray(Rs)
        self.Teffsamples = np.repeat(Teff, Nsamp) \
            if type(Teff) in [int,float] else \
               np.ascontiguousarray(Teff)
        self.agesamples = np.repeat(age, Nsamp) \
            if type(age) in [int,float] else \
               np.ascontiguousarray(age)

        self.mass = compute_point_estimates(self.Mssamples)
        self.radius = compute_point_estimates(self.Rssamples)
        self.teff = compute_point_estimates(self.Teffsamples)
        self.age = compute_point_estimates(self.agesamples)
        


class planet:

    def __init__(self, label, P, rp, mp, Xiron, Xice, Tkh,
                 radius_transition, albedo):
        '''
        Initialize one planet. 
        '''
        self.label = label
        self._radius_transition = float(radius_transition)
        self.albedo = float(albedo)
        self._define_planetary_params(P, rp, mp, Xiron, Xice, Tkh)


    def _define_planetary_params(self, P, rp, mp, Xiron, Xice, Tkh):
        self.Psamples = np.repeat(P, Nsamp) if type(P) in [int,float] else \
               np.ascontiguousarray(P)
        self.rpsamples = np.repeat(rp, Nsamp) if type(rp) in [int,float] else \
               np.ascontiguousarray(rp)
        self.Xironsamples = np.repeat(Xiron, Nsamp) \
            if type(Xiron) in [int,float] else \
               np.ascontiguousarray(Xiron)
        self.Xicesamples = np.repeat(Xice, Nsamp) \
            if type(Xice) in [int,float] else \
               np.ascontiguousarray(Xice)
        self.Tkhsamples = np.repeat(Tkh, Nsamp) \
            if type(Tkh) in [int,float] else \
               np.ascontiguousarray(Tkh)
        
        self.period = compute_point_estimates(self.Psamples)
        self.radius = compute_point_estimates(self.rpsamples)
        self._check_if_rocky()
        self.Xiron = compute_point_estimates(self.Xironsamples)
        self.Xice = compute_point_estimates(self.Xicesamples)
        self.Tkh = compute_point_estimates(self.Tkhsamples)

        # define planet mass if provided
        if mp != None:
            self.mpsamples = np.repeat(mp, Nsamp) \
                if type(mp) in [int,float] else \
                   np.ascontiguousarray(mp)
            self.mass = compute_point_estimates(self.mpsamples)
            
        else:
            self.mpsamples, self.mass = None, None


    def _check_if_rocky(self):
        '''
        Return True if the planet lies beneath the radius valley.
        Otherwise, return False.
        '''
        self.is_rocky = self.radius[0] < self._radius_transition


    def _compute_planet_params(self, star):
        '''Compute stellar-dependent planetary parameters.'''
        self.asamples = star.Mssamples**(1/3) * \
            days2yrs(self.Psamples)**(2/3)
        self.Teqsamples = (1-self.albedo)**(.25) * star.Teffsamples * \
            np.sqrt(Rsun2cm(star.Rssamples) / (2*AU2cm(self.asamples)))

        self.a = compute_point_estimates(self.asamples)
        self.teq = compute_point_estimates(self.Teqsamples)



def compute_point_estimates(samples):
    v = np.percentile(samples, (16,50,84))
    return v[1], v[2]-v[1], v[1]-v[0]
