from imports import *


def plot_Mmin_histograms(two_planet_system, outfile='', **kwargs):
    '''Plot the distribution of minimum masses for each physical model and 
    compare it to the gaseous planet's measured mass.'''
    # get physical mechanisms to plot
    self, Npanels = copy.copy(two_planet_system), 0
    tps_mechanism, label_mechanism = [], []
    if hasattr(self, 'photoevaporation'):
        Npanels += 1
        tps_mechanism.append(self.photoevaporation)
        label_mechanism.append('Photoevaporation')
    if hasattr(self, 'corepoweredmassloss'):
        Npanels += 1
        tps_mechanism.append(self.corepoweredmassloss)
        label_mechanism.append('Core-powered mass loss')
    if hasattr(self, 'gaspoorformation'):
        Npanels += 1
        tps_mechanism.append(self.gaspoorformation)
        label_mechanism.append('Gas-poor formation')
    if Npanels == 0:
        raise ValueError('No physical models have been run on the input system.')

    # customize plots
    if 'bins' in kwargs.keys():
        bins = kwargs['bins']
        log = not np.all(np.isclose(np.diff(bins), np.diff(bins)[0], rtol=1e-4))
    else:
        bins, log = 20, False
    
    fig, axs = plt.subplots(1, Npanels, sharex=True, figsize=(11,4))
    for i in range(Npanels):

        # plot minimum masses
        g = np.isfinite(tps_mechanism[i].planet_gaseous.Mmin_solution_samples)
        y, x_edges = np.histogram(tps_mechanism[i].planet_gaseous.Mmin_solution_samples[g], bins=bins)
        x_edges = _get_bin_edges(x_edges, log)
        _=axs[i].step(x_edges, y, ls='-', color='k', lw=2,
                      label='Calcuated\nminimum mass')

        # plot measured gaseous planet mass
        y2, x_edges = np.histogram(self.planet_gaseous.mpsamples, bins=bins)
        x_edges = _get_bin_edges(x_edges, log)
        _=axs[i].step(x_edges, y2*y.max()/y2.max(), ls='-', color='b', lw=2,
                      label='Measured mass')

        # customize the plot
        consistency = tps_mechanism[i].planet_gaseous.frac_consistent
        axs[i].set_title('%s (%.2f consistency rate)'%(label_mechanism[i],
                                                       consistency),fontsize=11)
        axs[i].set_xlabel('Gaseous planet mass [M$_{\oplus}$]', fontsize=12)
        if i == 0 : axs[i].set_ylabel('Number of realizations', fontsize=12)
        if log: axs[i].set_xscale('log')
        if i == 0: axs[i].legend(fontsize=10)

    fig.subplots_adjust(bottom=.12, top=.94, right=.99, left=.06)
    # save file if desired
    if outfile != '':
        plt.savefig(outfile)



def _get_bin_edges(bin_edges, log=False):
    if log:
        return 10**(np.log10(bin_edges)[:-1] + \
                    .5 * np.diff(np.log10(bin_edges))[0])
    else:
        return bin_edges[:-1] + .5 * np.diff(bin_edges)[0] 
