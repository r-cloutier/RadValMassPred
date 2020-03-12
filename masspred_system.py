from imports import *
import planetary_structure as ps
import photoevaporation as phev


class two_planet_system:

    def __init__(self, radval_func, Nsamp=1e3, label='syst00'):
        '''
        Initialize a planetary system that should contain two planets 
        that span the radius valley as defined in the input function 
        radval_func.
        '''
        self.label = label
        self._Nsamp = int(Nsamp)
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
        self.star = star(starlabel, self._Nsamp, Ms, Rs, Teff, age)


        
    def add_planet(self, P, rp, mp, Xiron=.33, Xice=0, Tkh=100,
                   radval_args=[], label=None, albedo=0.3):
        '''
        Add a planet to the system given its orbital period [days] and radius
        [Earth radii] in the form of a scalar value (i.e. zero uncertainty) or 
        as a numpy.array of samples from the parameter's marginalized posterior.
        '''
        planetlabel = '%s_planet%i'%(self.label, self.Nplanets) \
            if label == None else label
        p = planet(planetlabel, self._Nsamp, P, rp, mp, Xiron, Xice, Tkh,
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



    def compute_Mgas_min_photoevaporation(self, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the photoevaporation scenerio. 

        The minimum gaseous planet mass comes from equating the maximum mass 
        loss time of the rocky planet (i.e. it just lost its primordial H/He 
        envelope) to the minimum mass loss timescale for the gaseous planet.
        '''
        assert self._is_complete  # system is setup for calculations

        # sample gaseous planet minimum masses
        N = int(size)
        self.planet_gaseous.Mmin_samples = np.zeros(N)
        self.planet_gaseous.Xenvmax_samples = np.zeros(N)
        self.planet_gaseous.tmdotmax_samples = np.zeros(N)
        self.planet_gaseous.depthenvmax_samples = np.zeros(N)
        self.planet_rocky.Xenvmax_samples = np.zeros(N)
        self.planet_rocky.tmdotmax_samples = np.zeros(N)
        self.planet_rocky.depthenvmax_samples = np.zeros(N)

        progress_bar = initialize_progressbar(N, "\nComputing the gaseous planet's minimum mass (%i realizations)\n"%N)
        for i in range(N):

            progress_bar.update(i+1)
            
            # solve for Xenv that maximizes the rocky planet mass loss timescale
            N = int(size)
            args = sample(self.planet_rocky.Psamples), \
                sample(self.star.Mssamples), \
                sample(self.planet_rocky.mpsamples), \
                sample(self.planet_rocky.Teqsamples), \
                sample(self.planet_rocky.Tkhsamples), \
                sample(self.planet_rocky.Xironsamples), \
                sample(self.planet_rocky.Xicesamples)

            if value_errors:
                self.planet_rocky.Xenvmax_samples[i] = phev.compute_Xmax_rock(*args)
                # compute rocky planet mass loss timescale
                p = phev.compute_tmdot(self.planet_rocky.Xenvmax_samples[i], *args)
                self.planet_rocky.tmdotmax_samples[i] = p[0]
                self.planet_rocky.depthenvmax_samples[i] = p[1]
            
            else:
                try:
                    self.planet_rocky.Xenvmax_samples[i] = \
                        phev.compute_Xmax_rock(*args)
                    # compute rocky planet mass loss timescale
                    p = phev.compute_tmdot(self.planet_rocky.Xenvmax_samples[i],
                                           *args)
                    self.planet_rocky.tmdotmax_samples[i] = p[0]
                    self.planet_rocky.depthenvmax_samples[i] = p[1]
                
                except (ValueError, AssertionError):
                    self.planet_rocky.Xenvmax_samples[i] = np.nan
                    self.planet_rocky.tmdotmax_samples[i] = np.nan
                    self.planet_rocky.depthenvmax_samples[i] = np.nan


            # get minimum gaseous core mass for which Xenv < 1
            args = sample(self.planet_gaseous.Psamples), \
                sample(self.star.Mssamples), \
                sample(self.planet_gaseous.Teqsamples), \
                sample(self.planet_gaseous.Tkhsamples), \
                sample(self.planet_gaseous.Xironsamples), \
                sample(self.planet_gaseous.Xicesamples)
            Xenv_min = 1e-8
            Mcore_gas_min = phev.compute_Mcore_min_no_self_gravity(Xenv_min, *args)
    
            # check that a solution exists
            # get minimum gaseous core mass such that the gaseous planet's maximum
            # tmdot > rocky planet's maximum tmdot
            # i.e. increase minimum gas planet core mass until its maxmimum tmdot
            # exceeds that of the rocky planet
            Mcore_gas_min /= 1.1
            args = list(np.insert(args, 2, Mcore_gas_min))
            tmdotmax_gas = 0
            
            while (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) &  (Mcore_gas_min < self.planet_gaseous.mass[0]):
                
                # increase minimum gas planet core mass
                Mcore_gas_min *= 1.1
                args[2] = Mcore_gas_min

                if value_errors:
                    self.planet_gaseous.Xenvmax_samples[i] = \
                        phev.compute_Xmax_gas(*tuple(args))
                    # compute gaseous planet mass loss timescale
                    p = phev.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i],
                                           *tuple(args))
                    self.planet_gaseous.tmdotmax_samples[i] = p[0]
                    self.planet_gaseous.depthenvmax_samples[i] = p[1]
                    
                else:
                    try:
                        self.planet_gaseous.Xenvmax_samples[i] = \
                            phev.compute_Xmax_gas(*tuple(args))
                        # compute gaseous planet mass loss timescale
                        p = phev.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i],
                                               *tuple(args))
                        self.planet_gaseous.tmdotmax_samples[i] = p[0]
                        self.planet_gaseous.depthenvmax_samples[i] = p[1]

                    except (ValueError, AssertionError):
                        self.planet_gaseous.Xenvmax_samples[i] = np.nan
                        self.planet_gaseous.tmdotmax_samples[i] = np.nan
                        self.planet_gaseous.depthenvmax_samples[i] = np.nan

     
            # ensure that a solution exists
            if (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) | (Mcore_gas_min > self.planet_gaseous.mass[0]):
                raise ValueError("No solution exists because the gaseous planet's maximum mass loss timescale is less than the rocky planet's maximum mass loss timescale.")


            # just solved for minimum gaseous core mass to have a longer mass loss
            # time than the rocky planet
            # set maximum gaseous core mass for a pure iron ball
            Mcore_gas_max = ps.solidradius2mass(self.planet_gaseous.radius[0],1,0)
    
            # solve for the minimum gaseous planet mass
            args = sample(self.planet_gaseous.Psamples), \
                sample(self.star.Mssamples), \
                sample(self.planet_gaseous.Teqsamples), \
                sample(self.planet_gaseous.Tkhsamples), \
                sample(self.planet_gaseous.Xironsamples), \
                sample(self.planet_gaseous.Xicesamples), \
                sample(self.planet_gaseous.rpsamples), \
                sample(self.star.agesamples), \
                self.planet_rocky.tmdotmax_samples[i]
            if value_errors:
                Mgas_min = 10**brentq(phev._Mp_gas_to_solve,
                                      np.log10(Mcore_gas_min),
                                      np.log10(Mcore_gas_max),
                                      args=args)
                self.planet_gaseous.Mmin_samples[i] = Mgas_min

            else:
                try:
                    Mgas_min = 10**brentq(phev._Mp_gas_to_solve,
                                          np.log10(Mcore_gas_min),
                                          np.log10(Mcore_gas_max),
                                          args=args)
                    self.planet_gaseous.Mmin_samples[i] = Mgas_min

                except (ValueError, AssertionError):
                    self.planet_gaseous.Mmin_samples[i] = np.nan


        close_progressbar(progress_bar)
                    
        # gather point estimates
        self.planet_gaseous.Mmin = \
            compute_point_estimates(self.planet_gaseous.Mmin_samples)
        self.planet_gaseous.Xenvmax = \
            compute_point_estimates(self.planet_gaseous.Xenvmax_samples)
        self.planet_gaseous.tmdotmax = \
            compute_point_estimates(self.planet_gaseous.tmdotmax_samples)
        self.planet_gaseous.depthenvmax = \
            compute_point_estimates(self.planet_gaseous.depthenvmax_samples)
        self.planet_rocky.Xenvmax = \
            compute_point_estimates(self.planet_rocky.Xenvmax_samples)
        self.planet_rocky.tmdotmax = \
            compute_point_estimates(self.planet_rocky.tmdotmax_samples)
        self.planet_rocky.depthenvmax = \
            compute_point_estimates(self.planet_rocky.depthenvmax_samples)

                    
        

class star:

    def __init__(self, label, Nsamp, Ms, Rs, Teff, age):
        '''
        Initialize the host star.
        '''
        self.label = label
        self._Nsamp = int(Nsamp)
        self._define_stellar_params(Ms, Rs, Teff, age)


    def _define_stellar_params(self, Ms, Rs, Teff, age):
        '''Define stellar parameter point estimates and uncertainties.'''
        self.Mssamples = np.repeat(Ms, self._Nsamp) if type(Ms) in [int,float] else \
               np.ascontiguousarray(Ms)
        self.Rssamples = np.repeat(Rs, self._Nsamp) if type(Rs) in [int,float] else \
               np.ascontiguousarray(Rs)
        self.Teffsamples = np.repeat(Teff, self._Nsamp) \
            if type(Teff) in [int,float] else \
               np.ascontiguousarray(Teff)
        self.agesamples = np.repeat(age, self._Nsamp) \
            if type(age) in [int,float] else \
               np.ascontiguousarray(age)

        assert self.Mssamples.size == self._Nsamp
        assert self.Rssamples.size == self._Nsamp
        assert self.Teffsamples.size == self._Nsamp
        assert self.agesamples.size == self._Nsamp
        
        self.mass = compute_point_estimates(self.Mssamples)
        self.radius = compute_point_estimates(self.Rssamples)
        self.teff = compute_point_estimates(self.Teffsamples)
        self.age = compute_point_estimates(self.agesamples)

        


class planet:

    def __init__(self, label, Nsamp, P, rp, mp, Xiron, Xice, Tkh,
                 radius_transition, albedo):
        '''
        Initialize one planet. 
        '''
        self.label = label
        self._Nsamp = int(Nsamp)
        self._radius_transition = float(radius_transition)
        self.albedo = float(albedo)
        self._define_planetary_params(P, rp, mp, Xiron, Xice, Tkh)

        

    def _define_planetary_params(self, P, rp, mp, Xiron, Xice, Tkh):
        self.Psamples = np.repeat(P, self._Nsamp) if type(P) in [int,float] else \
               np.ascontiguousarray(P)
        self.rpsamples = np.repeat(rp, self._Nsamp) if type(rp) in [int,float] else \
               np.ascontiguousarray(rp)
        self.Xironsamples = np.repeat(Xiron, self._Nsamp) \
            if type(Xiron) in [int,float] else \
               np.ascontiguousarray(Xiron)
        assert np.all(self.Xironsamples >= 0) & np.all(self.Xironsamples <= 1)
        self.Xicesamples = np.repeat(Xice, self._Nsamp) \
            if type(Xice) in [int,float] else \
               np.ascontiguousarray(Xice)
        assert np.all(self.Xicesamples >= 0) & np.all(self.Xicesamples <= 1)
        self.Tkhsamples = np.repeat(Tkh, self._Nsamp) \
            if type(Tkh) in [int,float] else \
               np.ascontiguousarray(Tkh)

        assert self.Psamples.size == self._Nsamp
        assert self.rpsamples.size == self._Nsamp
        assert self.Xironsamples.size == self._Nsamp
        assert self.Xicesamples.size == self._Nsamp
        assert self.Tkhsamples.size == self._Nsamp

        self.period = compute_point_estimates(self.Psamples)
        self.radius = compute_point_estimates(self.rpsamples)
        self._check_if_rocky()
        self.Xiron = compute_point_estimates(self.Xironsamples)
        self.Xice = compute_point_estimates(self.Xicesamples)
        self.Tkh = compute_point_estimates(self.Tkhsamples)

        # define planet mass if provided
        if np.all(mp != None):
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
    v = np.nanpercentile(samples, (16,50,84))
    return v[1], v[2]-v[1], v[1]-v[0]


def sample(samples, size=1):
    '''Sample randomly from an input distribution.'''
    return np.random.choice(samples) if size == 1 \
        else np.random.choice(samples, size=int(N))


def initialize_progressbar(N, message=None):
    # if desired, print message at the beginning of the loop
    if (message != None) & (type(message) == str): print(message)
    
    # define progress bar if available
    try:
        progress_bar = progressbar.ProgressBar(maxval=int(N),
                                            widgets=[progressbar.Bar('=','[', ']'),
                                                     ' ', progressbar.Percentage()])
        progress_bar.start()
        return progress_bar 

    except NameError:
        warnings.warn('Progress bar cannot be initialized because the "progressbar" package is not available. Try running pip install progressbar.')
        return None


    
def close_progressbar(bar):
    dt = bar.seconds_elapsed
    print('Time elapsed = %.1f seconds (%.2f minutes).'%(dt,dt/60))
    bar.finish()
