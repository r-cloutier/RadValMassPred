from imports import *
import planetary_structure as ps
import photoevaporation as phev
import corepoweredmassloss as cpml
import gaspoorformation as gpf


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
        self._is_complete = False
       
 

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
            delattr(p, 'is_rocky')
            self.planet_rocky = p
        else:
            delattr(p, 'is_rocky')
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
        kwargs = {'value_errors': value_errors, 'size':size}
        self.photoevaporation = photoevaporation(self, **kwargs) 


        
    def compute_Mgas_min_corepoweredmassloss(self, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the core-powered mass loss scenerio. 

        The minimum gaseous planet mass comes from equating the maximum mass 
        loss time of the rocky planet (i.e. it just lost its primordial H/He 
        envelope) to the minimum mass loss timescale for the gaseous planet.
        '''
        kwargs = {'value_errors': value_errors, 'size':size}
        self.corepoweredmassloss = corepoweredmassloss(self, **kwargs)

        

    def compute_Mgas_min_gaspoorformation(self, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the gas-poor formation scenerio. 

        The minimum gaseous planet mass comes from its envelope mass fractions must 
        exceed that of the rocky planet.
        '''
        kwargs = {'value_errors': value_errors, 'size':size}
        self.gaspoorformation = gaspoorformation(self, **kwargs)



    def dump_pickle(self, fname, overwrite=False):
        '''Pickle the object.'''
        if (overwrite) | (not os.path.exists(fname)):
            f = open(fname, 'wb')
            pickle.dump(self, f)
            f.close()




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
        self.Teff = compute_point_estimates(self.Teffsamples)
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
            self.mpsamples = np.repeat(mp, self._Nsamp) \
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
        self.Teq = compute_point_estimates(self.Teqsamples)




class photoevaporation:

    def __init__(self, tps, value_errors=True, size=1):
        '''Class to make calculations based on the photoevaporation model.'''
        self.star = copy.copy(tps.star)
        self.planet_rocky = copy.copy(tps.planet_rocky)
        self.planet_gaseous = copy.copy(tps.planet_gaseous)
        #self = copy.copy(tps)
        
        # run the minimum mass calculation
        kwargs = {'value_errors': value_errors, 'size':size}
        self._compute_Mgas_min_photoevaporation(tps, **kwargs)
        

    def _compute_Mgas_min_photoevaporation(self, tps, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the photoevaporation scenerio. 

        The minimum gaseous planet mass comes from equating the maximum mass 
        loss time of the rocky planet (i.e. it just lost its primordial H/He 
        envelope) to the minimum mass loss timescale for the gaseous planet.
        '''
        assert tps._is_complete  # system is setup for calculations

        # sample gaseous planet minimum masses
        N = int(size)
        self.planet_gaseous.Mmin_solution_samples = np.zeros(N)
        self.planet_gaseous.Xenv_solution_samples = np.zeros(N)
        self.planet_gaseous.Rcore_solution_samples = np.zeros(N)
        self.planet_gaseous.Rrcb_solution_samples = np.zeros(N)
        self.planet_gaseous.Rpfull_solution_samples = np.zeros(N)
        self.planet_gaseous.tmdot_solution_samples = np.zeros(N)
        self.planet_gaseous.is_consistent_photoevap  = np.zeros(N)
        
        self.planet_gaseous.Mcorerangemin_samples = np.zeros(N)
        self.planet_gaseous.Mcorerangemax_samples = np.zeros(N)
        self.planet_gaseous.Xenvmax_samples = np.zeros(N)
        self.planet_gaseous.Rcoremax_samples = np.zeros(N)        
        self.planet_gaseous.Rrcbmax_samples = np.zeros(N)
        self.planet_gaseous.Rpfullmax_samples = np.zeros(N)
        self.planet_gaseous.tmdotmax_samples = np.zeros(N)
        self.planet_gaseous.depthenvmax_samples = np.zeros(N)

        self.planet_rocky.Xenvmax_samples = np.zeros(N)
        self.planet_rocky.Rcore_samples = np.zeros(N)
        self.planet_rocky.Rrcbmax_samples = np.zeros(N)
        self.planet_rocky.Rpfullmax_samples = np.zeros(N)
        self.planet_rocky.tmdotmax_samples = np.zeros(N)
        self.planet_rocky.depthenvmax_samples = np.zeros(N)


        # Monte-Carlo sample over parameters to get distribution of solutions 
        progress_bar = initialize_progressbar(N, "\nComputing the gaseous planet's minimum mass under photoevaporation (%i realizations)\n"%N)
        for i in range(N):

            progress_bar.update(i+1)

            # sample values for this realization
            Msi = sample(self.star.Mssamples)
            agei = sample(self.star.agesamples)
            Pi_rock = sample(self.planet_rocky.Psamples)
            rpi_rock = sample(self.planet_rocky.rpsamples)
            mpi_rock = sample(self.planet_rocky.mpsamples)
            Teqi_rock = sample(self.planet_rocky.Teqsamples)
            Tkhi_rock = sample(self.planet_rocky.Tkhsamples)
            Xironi_rock = sample(self.planet_rocky.Xironsamples)
            Xicei_rock = sample(self.planet_rocky.Xicesamples)
            Pi_gas = sample(self.planet_gaseous.Psamples)
            rpi_gas = sample(self.planet_gaseous.rpsamples)
            mpi_gas = sample(self.planet_gaseous.mpsamples)
            Teqi_gas = sample(self.planet_gaseous.Teqsamples)
            Tkhi_gas = sample(self.planet_gaseous.Tkhsamples)
            Xironi_gas = sample(self.planet_gaseous.Xironsamples)
            Xicei_gas = sample(self.planet_gaseous.Xicesamples)
            
            # solve for Xenv that maximizes the rocky planet mass loss timescale
            args = Pi_rock, Msi, mpi_rock, Teqi_rock, Tkhi_rock, Xironi_rock, Xicei_rock

            # compute the rocky planet core radius
            self.planet_rocky.Rcore_samples[i] = ps.mass2solidradius(args[2], *args[5:])
            
            if value_errors:
                # solve for maximum tmdot
                self.planet_rocky.Xenvmax_samples[i] = phev.compute_Xmax_rock(*args)
                # get planet structure
                p = ps.solve_radius_structure(self.planet_rocky.Xenvmax_samples[i],
                                              *args[2:])
                self.planet_rocky.Rrcbmax_samples[i] = p[0]
                self.planet_rocky.Rpfullmax_samples[i] = p[1]
                # compute rocky planet mass loss timescale
                p = phev.compute_tmdot(self.planet_rocky.Xenvmax_samples[i], *args)
                self.planet_rocky.tmdotmax_samples[i] = p[0]
                self.planet_rocky.depthenvmax_samples[i] = p[1]
        
            else:
                try:
                    # solve for maximum tmdot
                    self.planet_rocky.Xenvmax_samples[i] = \
                        phev.compute_Xmax_rock(rpi_rock, *args)
                    # get planet structure
                    p=ps.solve_radius_structure(self.planet_rocky.Xenvmax_samples[i],
                                                *args[2:])
                    self.planet_rocky.Rrcbmax_samples[i] = p[0]
                    self.planet_rocky.Rpfullmax_samples[i] = p[1]
                    # compute rocky planet mass loss timescale
                    ## TEMP
                    ##p = phev.compute_tmdot(self.planet_rocky.Xenvmax_samples[i], *args)
                    ##self.planet_rocky.tmdotmax_samples[i] = p[0]
                    ##self.planet_rocky.depthenvmax_samples[i] = p[1]
                    eta = phev.compute_mass_loss_efficiency(mpi_rock, self.planet_rocky.Rpfullmax_samples[i])
                    sma = Msi**(1/3) * (Pi_rock/365.25)**(2/3)
                    tmdot = self.planet_rocky.Xenvmax_samples[i] * mpi_rock**2 * AU2cm(sma)**2 * eta / Rearth2cm(self.planet_rocky.Rpfullmax_samples[i])**3
                    self.planet_rocky.tmdotmax_samples[i] = tmdot
                    self.planet_rocky.depthenvmax_samples[i] = 1

                except (ValueError, AssertionError):
                    self.planet_rocky.Xenvmax_samples[i] = np.nan
                    self.planet_rocky.Rrcbmax_samples[i] = np.nan
                    self.planet_rocky.Rpfullmax_samples[i] = np.nan
                    self.planet_rocky.tmdotmax_samples[i] = np.nan
                    self.planet_rocky.depthenvmax_samples[i] = np.nan


            # get minimum gaseous core mass for which Xenv < 1
            args = Pi_gas, Msi, Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas
            Xenv_min = 1e-8
            Mcore_gas_min = phev.compute_Mcore_min_no_self_gravity(Xenv_min, *args)
            self.planet_gaseous.Mcorerangemin_samples[i] = np.copy(Mcore_gas_min)

            # check that a solution exists
            # get minimum gaseous core mass such that the gaseous planet's maximum
            # tmdot > rocky planet's maximum tmdot
            # i.e. increase minimum gas planet core mass until its maxmimum tmdot
            # exceeds that of the rocky planet
            Mcore_gas_min /= 1.1
            args = list(np.insert(args, 2, Mcore_gas_min))
            self.planet_gaseous.tmdotmax_samples[i] = 0
            
            while (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) & (Mcore_gas_min < self.planet_gaseous.mass[0]):
                
                # increase minimum gas planet core mass
                Mcore_gas_min *= 1.1
                args[2] = Mcore_gas_min

                if value_errors:
                    self.planet_gaseous.Xenvmax_samples[i] = \
                        phev.compute_Xmax_gas(rpi_gas, *tuple(args))
                    # get planet structure
                    p=ps.solve_radius_structure(self.planet_gaseous.Xenvmax_samples[i],
                                                *args[2:])
                    self.planet_gaseous.Rrcbmax_samples[i] = p[0]
                    self.planet_gaseous.Rpfullmax_samples[i] = p[1]
                    # compute gaseous planet mass loss timescale
                    p = phev.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i],
                                           *tuple(args))
                    self.planet_gaseous.tmdotmax_samples[i] = p[0]
                    self.planet_gaseous.depthenvmax_samples[i] = p[1]
                    
                else:
                    try:
                        self.planet_gaseous.Xenvmax_samples[i] = \
                            phev.compute_Xmax_gas(rpi_gas, *tuple(args))
                        # get planet structure
                        p=ps.solve_radius_structure(self.planet_gaseous.Xenvmax_samples[i],
                                                    *args[2:])
                        self.planet_gaseous.Rrcbmax_samples[i] = p[0]
                        self.planet_gaseous.Rpfullmax_samples[i] = p[1]
                        # compute gaseous planet mass loss timescale
                        #p = phev.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i],
                        #                       *tuple(args))
                        #self.planet_gaseous.tmdotmax_samples[i] = p[0]
                        #self.planet_gaseous.depthenvmax_samples[i] = p[1]
                        eta = phev.compute_mass_loss_efficiency(mpi_rock, self.planet_gaseous.Rpfullmax_samples[i])
                        sma = Msi**(1/3) * (Pi_gas/365.25)**(2/3)
                        tmdot = self.planet_gaseous.Xenvmax_samples[i] * mpi_gas**2 * AU2cm(sma)**2 * eta / Rearth2cm(self.planet_gaseous.Rpfullmax_samples[i])**3
                        self.planet_gaseous.tmdotmax_samples[i] = tmdot
                        self.planet_gaseous.depthenvmax_samples[i] = 1

                    except (ValueError, AssertionError):
                        self.planet_gaseous.Xenvmax_samples[i] = np.nan
                        self.planet_gaseous.Rrcbmax_samples[i] = np.nan
                        self.planet_gaseous.Rpfullmax_samples[i] = np.nan
                        self.planet_gaseous.tmdotmax_samples[i] = np.nan
                        self.planet_gaseous.depthenvmax_samples[i] = np.nan

     
            # ensure that a solution exists
            if value_errors:
                if (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) | (self.planet_gaseous.Mcorerangemin_samples[i] > mpi_gas):
                    raise ValueError("No solution exists because the gaseous planet's maximum mass loss timescale is less than the rocky planet's maximum mass loss timescale.")
                else:
                    self.planet_gaseous.Mcorerangemin_samples[i] = Mcore_gas_min

            else:
                if (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) | (self.planet_gaseous.Mcorerangemin_samples[i] > mpi_gas):
                    self.planet_gaseous.Mcorerangemin_samples[i] = np.nan
                else:
                    self.planet_gaseous.Mcorerangemin_samples[i] = Mcore_gas_min
                        
            # just solved for minimum gaseous core mass to have a longer mass loss
            # time than the rocky planet
            # set maximum gaseous core mass for a pure iron ball
            self.planet_gaseous.Mcorerangemax_samples[i] = 100#\
            #    ps.solidradius2mass(rpi_gas, Xironi_gas, 0)
            #self.planet_gaseous.Mcorerangemax_samples[i] = \
            #    phev.compute_Mcore_max(rpi_gas, Teqi_gas, agei, Xironi_gas, Xicei_gas)

            # solve for the minimum gaseous planet mass
            args = Pi_gas, Msi, Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas, rpi_gas, \
                agei, self.planet_rocky.tmdotmax_samples[i]
            if value_errors:
                vmin = self.planet_gaseous.Mcorerangemin_samples[i]
                vmax = self.planet_gaseous.Mcorerangemax_samples[i]
                Mgas_min = 10**brentq(phev._Mp_gas_to_solve, np.log10(vmin),
                                      np.log10(vmax), args=args)
                self.planet_gaseous.Mmin_solution_samples[i] = Mgas_min

            else:
                try:
                    vmin = self.planet_gaseous.Mcorerangemin_samples[i]
                    vmax = self.planet_gaseous.Mcorerangemax_samples[i]
                    Mgas_min = 10**brentq(phev._Mp_gas_to_solve, np.log10(vmin),
                                          np.log10(vmax), args=args)
                    self.planet_gaseous.Mmin_solution_samples[i] = Mgas_min

                except (ValueError, AssertionError):
                    self.planet_gaseous.Mmin_solution_samples[i] = np.nan

                    
            # solve for the envelope mass fraction and planet structure
            # given the minimum mass
            if value_errors:
                # get Xenv
                args = rpi_gas, self.planet_gaseous.Mmin_solution_samples[i], Teqi_gas, \
                    agei, Xironi_gas, Xicei_gas
                p1 = ps.Rp_solver_gas(*args)
                # solve radius structure
                args = self.planet_gaseous.Xenv_solution_samples[i], \
                    self.planet_gaseous.Mmin_solution_samples[i], \
                    Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas
                p2 = ps.solve_radius_structure(*args)
                p3 = phev.compute_tmdot(self.planet_gaseous.Xenv_solution_samples[i],
                                        Pi_gas, Msi,
                                        self.planet_gaseous.Mmin_solution_samples[i],
                                        Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas)
                
            else:
                try:
                    # get Xenv
                    args = rpi_gas, self.planet_gaseous.Mmin_solution_samples[i], \
                        Teqi_gas, agei, Xironi_gas, Xicei_gas
                    p1 = ps.Rp_solver_gas(*args)
                    # solve radius structure
                    args = self.planet_gaseous.Xenv_solution_samples[i], \
                        self.planet_gaseous.Mmin_solution_samples[i], \
                        Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas
                    p2 = ps.solve_radius_structure(*args)
                    p3 = phev.compute_tmdot(self.planet_gaseous.Xenv_solution_samples[i],
                                            Pi_gas, Msi,
                                            self.planet_gaseous.Mmin_solution_samples[i],
                                            Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas)
                except (ValueError, AssertionError):
                    p1 = np.repeat(np.nan, 2)
                    p2 = np.repeat(np.nan, 2)
                    p3 = np.repeat(np.nan, 2)

            # save solution results
            self.planet_gaseous.Xenv_solution_samples[i] = p1[0]
            self.planet_gaseous.Rcore_solution_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenv_solution_samples[i], Xironi_gas, Xicei_gas)
            self.planet_gaseous.Rrcb_solution_samples[i] = p2[0]
            self.planet_gaseous.Rpfull_solution_samples[i] = p2[1]
            self.planet_gaseous.tmdot_solution_samples[i] = p3[0]

            # is the minimum mass consistent with the photoevaporation model?
            if np.isfinite(self.planet_gaseous.Mmin_solution_samples[i]):
                self.planet_gaseous.is_consistent_photoevap[i] = \
                    self.planet_gaseous.Mmin_solution_samples[i] < mpi_gas
            else:
                self.planet_gaseous.is_consistent_photoevap[i] = np.nan
                
            # compute gaseous planet core radius
            self.planet_gaseous.Rcoremax_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenvmax_samples[i], *args[-2:])
            self.planet_gaseous.Rcore_solution_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenv_solution_samples[i], *args[-2:])

        
        close_progressbar(progress_bar)
                    
        # gather point estimates
        self.planet_gaseous.Mmin_solution = \
            compute_point_estimates(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.Xenv_solution = \
            compute_point_estimates(self.planet_gaseous.Xenv_solution_samples)
        self.planet_gaseous.Rcore_solution = \
            compute_point_estimates(self.planet_gaseous.Rcore_solution_samples)
        self.planet_gaseous.Rrcb_solution = \
            compute_point_estimates(self.planet_gaseous.Rrcb_solution_samples)
        self.planet_gaseous.Rpfull_solution = \
            compute_point_estimates(self.planet_gaseous.Rpfull_solution_samples)
        self.planet_gaseous.tmdot_solution = \
            compute_point_estimates(self.planet_gaseous.tmdot_solution_samples)
        g = np.isfinite(self.planet_gaseous.is_consistent_photoevap)
        self.planet_gaseous.frac_consistent_photoevap = self.planet_gaseous.is_consistent_photoevap[g].sum() / g.sum()
        self.planet_gaseous.success_photoevap_samples = np.isfinite(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.frac_success_photoevap = self.planet_gaseous.success_photoevap_samples.mean()
        self.planet_gaseous.Mcorerangemin = \
            compute_point_estimates(self.planet_gaseous.Mcorerangemin_samples)
        self.planet_gaseous.Mcorerangemax = \
            compute_point_estimates(self.planet_gaseous.Mcorerangemax_samples)
        self.planet_gaseous.Xenvmax = \
            compute_point_estimates(self.planet_gaseous.Xenvmax_samples)
        self.planet_gaseous.Rcoremax = \
            compute_point_estimates(self.planet_gaseous.Rcoremax_samples)
        self.planet_gaseous.Rrcbmax = \
            compute_point_estimates(self.planet_gaseous.Rrcbmax_samples)
        self.planet_gaseous.Rpfullmax = \
            compute_point_estimates(self.planet_gaseous.Rpfullmax_samples)
        self.planet_gaseous.tmdotmax = \
            compute_point_estimates(self.planet_gaseous.tmdotmax_samples)
        self.planet_gaseous.depthenvmax = \
            compute_point_estimates(self.planet_gaseous.depthenvmax_samples)
        self.planet_rocky.Xenvmax = \
            compute_point_estimates(self.planet_rocky.Xenvmax_samples)
        self.planet_rocky.Rcore = \
            compute_point_estimates(self.planet_rocky.Rcore_samples)
        self.planet_rocky.Rrcbmax = \
            compute_point_estimates(self.planet_rocky.Rrcbmax_samples)
        self.planet_rocky.Rpfullmax = \
            compute_point_estimates(self.planet_rocky.Rpfullmax_samples)
        self.planet_rocky.tmdotmax = \
            compute_point_estimates(self.planet_rocky.tmdotmax_samples)
        self.planet_rocky.depthenvmax = \
            compute_point_estimates(self.planet_rocky.depthenvmax_samples)

        


class corepoweredmassloss:

    def __init__(self, tps, value_errors=True, size=1):
        '''Class to make calculations based on the core-powered mass loss model.'''
        self.star = copy.copy(tps.star)
        self.planet_rocky = copy.copy(tps.planet_rocky)
        self.planet_gaseous = copy.copy(tps.planet_gaseous)
        
        # run the minimum mass calculation
        kwargs = {'value_errors': value_errors, 'size':size}
        self._compute_Mgas_min_corepoweredmassloss(tps, **kwargs)


    def _compute_Mgas_min_corepoweredmassloss(self, tps, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the core-powered mass loss scenerio. 

        The minimum gaseous planet mass comes from equating the maximum mass 
        loss time of the rocky planet (i.e. it just lost its primordial H/He 
        envelope) to the minimum mass loss timescale for the gaseous planet.
        '''
        assert tps._is_complete  # system is setup for calculations

        # sample gaseous planet minimum masses
        N = int(size)
        self.planet_gaseous.Mmin_solution_samples = np.zeros(N)
        self.planet_gaseous.Xenv_solution_samples = np.zeros(N)
        self.planet_gaseous.Rcore_solution_samples = np.zeros(N)
        self.planet_gaseous.Rrcb_solution_samples = np.zeros(N)
        self.planet_gaseous.Rpfull_solution_samples = np.zeros(N)
        self.planet_gaseous.tmdot_solution_samples = np.zeros(N)
        self.planet_gaseous.is_consistent_corepoweredmassloss  = np.zeros(N)
        
        self.planet_gaseous.Mcorerangemin_samples = np.zeros(N)
        self.planet_gaseous.Mcorerangemax_samples = np.zeros(N)
        self.planet_gaseous.Xenvmax_samples = np.zeros(N)
        self.planet_gaseous.Rcoremax_samples = np.zeros(N)        
        self.planet_gaseous.Rrcbmax_samples = np.zeros(N)
        self.planet_gaseous.Rpfullmax_samples = np.zeros(N)
        self.planet_gaseous.tmdotmax_samples = np.zeros(N)
        
        self.planet_rocky.Xenvmax_samples = np.zeros(N)
        self.planet_rocky.Rcore_samples = np.zeros(N)
        self.planet_rocky.Rrcbmax_samples = np.zeros(N)
        self.planet_rocky.Rpfullmax_samples = np.zeros(N)
        self.planet_rocky.tmdotmax_samples = np.zeros(N)

        # Monte-Carlo sample over parameters to get distribution of solutions 
        progress_bar = initialize_progressbar(N, "\nComputing the gaseous planet's minimum mass under core-powered mass loss (%i realizations)\n"%N)
        for i in range(N):

            progress_bar.update(i+1)

            # sample values for this realization
            Msi = sample(self.star.Mssamples)
            agei = sample(self.star.agesamples)
            Pi_rock = sample(self.planet_rocky.Psamples)
            rpi_rock = sample(self.planet_rocky.rpsamples)
            mpi_rock = sample(self.planet_rocky.mpsamples)
            Teqi_rock = sample(self.planet_rocky.Teqsamples)
            Tkhi_rock = sample(self.planet_rocky.Tkhsamples)
            Xironi_rock = sample(self.planet_rocky.Xironsamples)
            Xicei_rock = sample(self.planet_rocky.Xicesamples)
            Pi_gas = sample(self.planet_gaseous.Psamples)
            rpi_gas = sample(self.planet_gaseous.rpsamples)
            mpi_gas = sample(self.planet_gaseous.mpsamples)
            Teqi_gas = sample(self.planet_gaseous.Teqsamples)
            Tkhi_gas = sample(self.planet_gaseous.Tkhsamples)
            Xironi_gas = sample(self.planet_gaseous.Xironsamples)
            Xicei_gas = sample(self.planet_gaseous.Xicesamples)
            
            # solve for Xenv that maximizes the rocky planet mass loss timescale
            args = mpi_rock, Teqi_rock, Tkhi_rock, Xironi_rock, Xicei_rock

            # compute the rocky planet core radius
            self.planet_rocky.Rcore_samples[i] = ps.mass2solidradius(args[0], *args[3:])
            
            if value_errors:
                # solve for maximum tmdot
                self.planet_rocky.Xenvmax_samples[i] = cpml.compute_Xmax(*args)
                # get planet structure
                p = ps.solve_radius_structure(self.planet_rocky.Xenvmax_samples[i], *args)
                self.planet_rocky.Rrcbmax_samples[i] = p[0]
                self.planet_rocky.Rpfullmax_samples[i] = p[1]
                # compute rocky planet mass loss timescale
                self.planet_rocky.tmdotmax_samples[i] = \
                    cpml.compute_tmdot(self.planet_rocky.Xenvmax_samples[i], *args)
    
            else:
                try:
                    # solve for maximum tmdot
                    self.planet_rocky.Xenvmax_samples[i] = cpml.compute_Xmax(*args)
                    # get planet structure
                    p=ps.solve_radius_structure(self.planet_rocky.Xenvmax_samples[i],*args)
                    self.planet_rocky.Rrcbmax_samples[i] = p[0]
                    self.planet_rocky.Rpfullmax_samples[i] = p[1]
                    # compute rocky planet mass loss timescale
                    self.planet_rocky.tmdotmax_samples[i] = \
                        cpml.compute_tmdot(self.planet_rocky.Xenvmax_samples[i], *args)

                except (ValueError, AssertionError):
                    self.planet_rocky.Xenvmax_samples[i] = np.nan
                    self.planet_rocky.Rrcbmax_samples[i] = np.nan
                    self.planet_rocky.Rpfullmax_samples[i] = np.nan
                    self.planet_rocky.tmdotmax_samples[i] = np.nan


            # check that a solution exists
            # get minimum gaseous core mass such that the gaseous planet's maximum
            # tmdot > rocky planet's maximum tmdot
            # i.e. increase minimum gas planet core mass until its maxmimum tmdot
            # exceeds that of the rocky planet
            Mcore_gas_min = .1
            self.planet_gaseous.Mcorerangemin_samples[i] = np.copy(Mcore_gas_min)
            Mcore_gas_min /= 1.1
            args = [Mcore_gas_min, Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas]
            self.planet_gaseous.tmdotmax_samples[i] = 0

            while (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) & (Mcore_gas_min < self.planet_gaseous.mass[0]):
                
                # increase minimum gas planet core mass
                Mcore_gas_min *= 1.1
                args[0] = Mcore_gas_min

                if value_errors:
                    self.planet_gaseous.Xenvmax_samples[i]= cpml.compute_Xmax(*tuple(args))
                    # get planet structure
                    p=ps.solve_radius_structure(self.planet_gaseous.Xenvmax_samples[i],
                                                *tuple(args))
                    self.planet_gaseous.Rrcbmax_samples[i] = p[0]
                    self.planet_gaseous.Rpfullmax_samples[i] = p[1]
                    # compute gaseous planet mass loss timescale
                    self.planet_gaseous.tmdotmax_samples[i] = cpml.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i], *tuple(args))
                
                else:
                    try:
                        self.planet_gaseous.Xenvmax_samples[i] = \
                            cpml.compute_Xmax(*tuple(args))
                        # get planet structure
                        p=ps.solve_radius_structure(self.planet_gaseous.Xenvmax_samples[i],
                                                    *tuple(args))
                        self.planet_gaseous.Rrcbmax_samples[i] = p[0]
                        self.planet_gaseous.Rpfullmax_samples[i] = p[1]
                        self.planet_gaseous.tmdotmax_samples[i] = cpml.compute_tmdot(self.planet_gaseous.Xenvmax_samples[i], *tuple(args))

                    except (ValueError, AssertionError):
                        self.planet_gaseous.Xenvmax_samples[i] = np.nan
                        self.planet_gaseous.Rrcbmax_samples[i] = np.nan
                        self.planet_gaseous.Rpfullmax_samples[i] = np.nan
                        self.planet_gaseous.tmdotmax_samples[i] = np.nan
                        
     
            # ensure that a solution exists
            if value_errors:
                if (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) | (self.planet_gaseous.Mcorerangemin_samples[i] > mpi_gas):
                    raise ValueError("No solution exists because the gaseous planet's maximum mass loss timescale is less than the rocky planet's maximum mass loss timescale.")
                else:
                    self.planet_gaseous.Mcorerangemin_samples[i] = Mcore_gas_min

            else:
                if (self.planet_gaseous.tmdotmax_samples[i] < self.planet_rocky.tmdotmax_samples[i]) | (self.planet_gaseous.Mcorerangemin_samples[i] > mpi_gas):
                    self.planet_gaseous.Mcorerangemin_samples[i] = np.nan
                else:
                    self.planet_gaseous.Mcorerangemin_samples[i] = Mcore_gas_min


            # just solved for minimum gaseous core mass to have a longer mass loss
            # time than the rocky planet
            # set maximum gaseous core mass
            self.planet_gaseous.Mcorerangemax_samples[i] = 100
            
            # solve for the minimum gaseous planet mass
            args = Teqi_gas, Xironi_gas, Xicei_gas, rpi_gas, agei, \
                self.planet_rocky.tmdotmax_samples[i]
            vmin = self.planet_gaseous.Mcorerangemin_samples[i]
            vmax = self.planet_gaseous.Mcorerangemax_samples[i]

            if value_errors:
                Mgas_min = 10**brentq(cpml._Mp_gas_to_solve, np.log10(vmin),
                                      np.log10(vmax), args=args)
                self.planet_gaseous.Mmin_solution_samples[i] = Mgas_min

            else:
                try:
                    Mgas_min = 10**brentq(cpml._Mp_gas_to_solve, np.log10(vmin),
                                          np.log10(vmax), args=args)
                    self.planet_gaseous.Mmin_solution_samples[i] = Mgas_min

                except (ValueError, AssertionError):
                    self.planet_gaseous.Mmin_solution_samples[i] = np.nan

                    
            # solve for the envelope mass fraction and planet structure
            # given the minimum mass
            args = rpi_gas, self.planet_gaseous.Mmin_solution_samples[i], Teqi_gas, agei, \
                Xironi_gas, Xicei_gas
            
            if value_errors:
                # get Xenv
                p1 = ps.Rp_solver_gas(*args)
                # solve radius structure
                args = self.planet_gaseous.Xenv_solution_samples[i], \
                    self.planet_gaseous.Mmin_solution_samples[i], \
                    Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas
                p2 = ps.solve_radius_structure(*args)
                p3 = cpml.compute_tmdot(self.planet_gaseous.Xenv_solution_samples[i],
                                        self.planet_gaseous.Mmin_solution_samples[i],
                                        Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas)
                
            else:
                try:
                    # get Xenv
                    p1 = ps.Rp_solver_gas(*args)
                    # solve radius structure
                    args = self.planet_gaseous.Xenv_solution_samples[i], \
                        self.planet_gaseous.Mmin_solution_samples[i], \
                        Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas
                    p2 = ps.solve_radius_structure(*args)
                    p3 = cpml.compute_tmdot(self.planet_gaseous.Xenv_solution_samples[i],
                                            self.planet_gaseous.Mmin_solution_samples[i],
                                            Teqi_gas, Tkhi_gas, Xironi_gas, Xicei_gas)
                except (ValueError, AssertionError):
                    p1 = np.repeat(np.nan, 2)
                    p2 = np.repeat(np.nan, 2)
                    p3 = np.nan

            # save solution results
            self.planet_gaseous.Xenv_solution_samples[i] = p1[0]
            self.planet_gaseous.Rcore_solution_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenv_solution_samples[i], Xironi_gas, Xicei_gas)
            self.planet_gaseous.Rrcb_solution_samples[i] = p2[0]
            self.planet_gaseous.Rpfull_solution_samples[i] = p2[1]
            self.planet_gaseous.tmdot_solution_samples[i] = p3

            # is the minimum mass consistent with the core-powered mass loss model?
            if np.isfinite(self.planet_gaseous.Mmin_solution_samples[i]):
                self.planet_gaseous.is_consistent_corepoweredmassloss[i] = \
                    self.planet_gaseous.Mmin_solution_samples[i] < mpi_gas
            else:
                self.planet_gaseous.is_consistent_corepoweredmassloss[i] = np.nan
                
            # compute gaseous planet core radius
            self.planet_gaseous.Rcoremax_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenvmax_samples[i], *args[-2:])
            self.planet_gaseous.Rcore_solution_samples[i] = \
                ps.mass2solidradius(self.planet_gaseous.Mmin_solution_samples[i]/self.planet_gaseous.Xenv_solution_samples[i], *args[-2:])

        
        close_progressbar(progress_bar)
                    
        # gather point estimates
        self.planet_gaseous.Mmin_solution = \
            compute_point_estimates(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.Xenv_solution = \
            compute_point_estimates(self.planet_gaseous.Xenv_solution_samples)
        self.planet_gaseous.Rcore_solution = \
            compute_point_estimates(self.planet_gaseous.Rcore_solution_samples)
        self.planet_gaseous.Rrcb_solution = \
            compute_point_estimates(self.planet_gaseous.Rrcb_solution_samples)
        self.planet_gaseous.Rpfull_solution = \
            compute_point_estimates(self.planet_gaseous.Rpfull_solution_samples)
        self.planet_gaseous.tmdot_solution = \
            compute_point_estimates(self.planet_gaseous.tmdot_solution_samples)
        g = np.isfinite(self.planet_gaseous.is_consistent_corepoweredmassloss)
        self.planet_gaseous.frac_consistent_corepoweredmassloss = self.planet_gaseous.is_consistent_corepoweredmassloss[g].sum() / g.sum()
        self.planet_gaseous.success_corepoweredmassloss_samples = np.isfinite(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.frac_success_corepoweredmassloss = self.planet_gaseous.success_corepoweredmassloss_samples.mean()
        self.planet_gaseous.Mcorerangemin = \
            compute_point_estimates(self.planet_gaseous.Mcorerangemin_samples)
        self.planet_gaseous.Mcorerangemax = \
            compute_point_estimates(self.planet_gaseous.Mcorerangemax_samples)
        self.planet_gaseous.Xenvmax = \
            compute_point_estimates(self.planet_gaseous.Xenvmax_samples)
        self.planet_gaseous.Rcoremax = \
            compute_point_estimates(self.planet_gaseous.Rcoremax_samples)
        self.planet_gaseous.Rrcbmax = \
            compute_point_estimates(self.planet_gaseous.Rrcbmax_samples)
        self.planet_gaseous.Rpfullmax = \
            compute_point_estimates(self.planet_gaseous.Rpfullmax_samples)
        self.planet_gaseous.tmdotmax = \
            compute_point_estimates(self.planet_gaseous.tmdotmax_samples)
        self.planet_rocky.Xenvmax = \
            compute_point_estimates(self.planet_rocky.Xenvmax_samples)
        self.planet_rocky.Rcore = \
            compute_point_estimates(self.planet_rocky.Rcore_samples)
        self.planet_rocky.Rrcbmax = \
            compute_point_estimates(self.planet_rocky.Rrcbmax_samples)
        self.planet_rocky.Rpfullmax = \
            compute_point_estimates(self.planet_rocky.Rpfullmax_samples)
        self.planet_rocky.tmdotmax = \
            compute_point_estimates(self.planet_rocky.tmdotmax_samples)
        

        
class gaspoorformation:

    def __init__(self, tps, value_errors=True, size=1):
        '''Class to make calculations based on the gas-poor formation model.'''
        self.star = copy.copy(tps.star)
        self.planet_rocky = copy.copy(tps.planet_rocky)
        self.planet_gaseous = copy.copy(tps.planet_gaseous)
        
        # run the minimum mass calculation
        kwargs = {'value_errors': value_errors, 'size':size}
        self._compute_Mgas_min_gaspoorformation(tps, **kwargs)


    def _compute_Mgas_min_gaspoorformation(self, tps, value_errors=True, size=1):
        '''
        Compute the minimum mass of the gaseous planet in order to be 
        consistent with the gas-poor formation scenerio. 

        The minimum gaseous planet mass comes from its envelope mass fractions must 
        exceed that of the rocky planet.
        '''
        assert tps._is_complete  # system is setup for calculations

        # sample gaseous planet minimum masses
        N = int(size)
        self.planet_gaseous.Mmin_solution_samples = np.zeros(N)
        self.planet_gaseous.Xenv_solution_samples = np.zeros(N)
        self.planet_gaseous.taccrete_samples = np.zeros(N)
        self.planet_gaseous.tdisk_samples = np.zeros(N)
        self.planet_gaseous.is_consistent_gaspoorformation = np.zeros(N)
        
        
        progress_bar = initialize_progressbar(N, "\nComputing the gaseous planet's minimum mass under gas-poor formation (%i realizations)\n"%N)
        for i in range(N):

            progress_bar.update(i+1)

            # sample values for this realization
            agei = sample(self.star.agesamples)
            ai_rock = sample(self.planet_rocky.asamples)
            Teqi_rock = sample(self.planet_rocky.Teqsamples)
            rpi_rock = sample(self.planet_rocky.rpsamples)
            mpi_rock = sample(self.planet_rocky.mpsamples)
            Xironi_rock = sample(self.planet_rocky.Xironsamples)
            Xicei_rock = sample(self.planet_rocky.Xicesamples)
            ai_gas = sample(self.planet_gaseous.asamples)
            Teqi_gas = sample(self.planet_gaseous.Teqsamples)
            rpi_gas = sample(self.planet_gaseous.rpsamples)
            mpi_gas = sample(self.planet_gaseous.mpsamples)
            Xironi_gas = sample(self.planet_gaseous.Xironsamples)
            Xicei_gas = sample(self.planet_gaseous.Xicesamples)
            
            # compare solid masses to derive minimum gaseous planet mass
            args = mpi_rock, ai_rock, ai_gas
            if value_errors:
                self.planet_gaseous.Mmin_solution_samples[i]= gpf.compute_Mgas_min(*args)
                argsX = rpi_gas, mpi_gas, Teqi_gas, agei, Xironi_gas, Xicei_gas
                self.planet_gaseous.Xenv_solution_samples[i],_ = ps.Rp_solver_gas(*argsX)

            else:
                try:
                    self.planet_gaseous.Mmin_solution_samples[i]= \
                        gpf.compute_Mgas_min(*args)
                    argsX = rpi_gas, mpi_gas, Teqi_gas, agei, Xironi_gas, Xicei_gas
                    self.planet_gaseous.Xenv_solution_samples[i],_ = ps.Rp_solver_gas(*argsX)
                    
                except (ValueError, AssertionError):
                    self.planet_gaseous.Mmin_solution_samples[i] = np.nan
                    self.planet_gaseous.Xenv_solution_samples[i] = np.nan

            # check that the gaseous planet can accrete enough gas to explain its
            # radius before disk dispersal
            self.planet_gaseous.tdisk_samples[i] = gpf.sample_disk_lifetime()
            args = self.planet_gaseous.Xenv_solution_samples[i], Teqi_gas, mpi_gas
            self.planet_gaseous.taccrete_samples[i] = gpf.solve_taccrete_gas(*args)
            self.planet_gaseous.is_consistent_gaspoorformation[i] = \
                    self.planet_gaseous.taccrete_samples[i] <= \
                    self.planet_gaseous.tdisk_samples[i]

            # is the minimum mass consistent with the core-powered mass loss model?
            if np.isfinite(self.planet_gaseous.Mmin_solution_samples[i]):
                self.planet_gaseous.is_consistent_gaspoorformation[i] *= \
                    self.planet_gaseous.Mmin_solution_samples[i] < mpi_gas
            else:
                self.planet_gaseous.is_consistent_gaspoorformation[i] *= np.nan

        close_progressbar(progress_bar)
                    
        # gather point estimates
        self.planet_gaseous.Mmin_solution = \
            compute_point_estimates(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.Xenv_solution = \
            compute_point_estimates(self.planet_gaseous.Xenv_solution_samples)
        self.planet_gaseous.taccrete = \
            compute_point_estimates(self.planet_gaseous.taccrete_samples)
        self.planet_gaseous.tdisk = \
            compute_point_estimates(self.planet_gaseous.tdisk_samples)
        g = np.isfinite(self.planet_gaseous.is_consistent_gaspoorformation)
        self.planet_gaseous.frac_consistent_gaspoorformation = self.planet_gaseous.is_consistent_gaspoorformation[g].sum() / g.sum()
        self.planet_gaseous.success_gaspoorformation_samples = np.isfinite(self.planet_gaseous.Mmin_solution_samples)
        self.planet_gaseous.frac_success_gaspoorformation = self.planet_gaseous.success_gaspoorformation_samples.mean()

        
    
        

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


def load_pickle(fname):
    f = open(fname, 'rb')
    tps = pickle.load(f)
    f.close()
    return tps
