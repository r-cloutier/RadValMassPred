from imports import *


def plot_Mmin_histograms(two_planet_system, outfile='', **kwargs):
    '''Plot the distribution of minimum masses for each physical model and 
    compare it to the gaseous planet's measured mass.'''
    self = copy.copy(two_planet_system)
    tps_mechanism = self.photoevaporation, self.corepoweredmassloss, self.gaspoorformation
    label_mechanism = 'Photoevaporation', 'Core-powered mass loss', 'Gas-poor formation'
    
    # customize plots
    if 'bins' in kwargs.keys():
        bins = kwargs['bins']
        log = not np.all(np.isclose(np.diff(bins), np.diff(bins)[0], rtol=1e-4))
    else:
        bins, log = 20, False
    
    fig, axs = plt.subplots(1, 3, sharex=True, figsize=(11,4))
    for i in range(3):

        # plot minimum masses
        try:
            g = np.isfinite(tps_mechanism[i].planet_gaseous.Mmin_solution_samples)
            y, x_edges = np.histogram(tps_mechanism[i].planet_gaseous.Mmin_solution_samples[g], bins=bins)
            x_edges = _get_bin_edges(x_edges, log)
            _=axs[i].step(x_edges, y, ls='-', color='k', lw=2,
                          label='Calcuated\nminimum mass')
        except AttributeError:
            pass
            
        # plot measured gaseous planet mass
        y2, x_edges = np.histogram(self.planet_gaseous.mpsamples, bins=bins)
        x_edges = _get_bin_edges(x_edges, log)
        _=axs[i].step(x_edges, y2*y.max()/y2.max(), ls='-', color='b', lw=2,
                      label='Measured mass')

        # customize the plot
        consistency = tps_mechanism[i].planet_gaseous.frac_consistent
        axs[i].set_title('%s\n(consistency rate = %i%%)'%(label_mechanism[i],
                                                          consistency*1e2),
                         fontsize=11)
        axs[i].set_xlabel('Gaseous planet mass [M$_{\oplus}$]', fontsize=12)
        if i == 0 : axs[i].set_ylabel('Number of realizations', fontsize=12)
        if log: axs[i].set_xscale('log')
        if i == 0: axs[i].legend(fontsize=10)

    fig.subplots_adjust(bottom=.12, top=.9, right=.99, left=.06)
    # save file if desired
    if outfile != '':
        plt.savefig(outfile)



def _get_bin_edges(bin_edges, log=False):
    if log:
        return 10**(np.log10(bin_edges)[:-1] + \
                    .5 * np.diff(np.log10(bin_edges))[0])
    else:
        return bin_edges[:-1] + .5 * np.diff(bin_edges)[0] 
