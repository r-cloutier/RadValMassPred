from imports import *


def plot_Mmin_histograms(two_planet_system, outfile='', Nbins=15, logx=False):
    '''Plot the distribution of minimum masses for each physical model and
    compare it to the gaseous planet's measured mass.'''
    # add available mechanisms
    self = copy.copy(two_planet_system)
    labels = {'photoevaporation': ['Photoevaporation','#a50f15'],
              'corepoweredmassloss': ['Core-powered mass loss','#08519c'],
              'gaspoorformation': ['Gas-poor formation','#238b45']}
    tps_mechanism, label_mechanism, colour_mechanism = [], [], []
    for s in labels.keys():
        try:
            tps_mechanism.append(getattr(self, s))
            label_mechanism.append(labels[s][0])
            colour_mechanism.append(labels[s][1])
        except AttributeError:
            pass

    Nmech = len(tps_mechanism)
    assert len(label_mechanism) == Nmech
    
    # customize binning
    if hasattr(Nbins, '__len__'):
        assert len(Nbins)+1 >= Nmech
    else:
        Nbins = np.repeat(Nbins, Nmech+1)

    fig, axs = plt.subplots(1, Nmech, sharex=True, figsize=(4.5*Nmech,3.7))
    axs = [axs] if Nmech == 1 else axs
    for i in range(Nmech):

        # plot minimum masses for this mechanism if available
        g = np.isfinite(tps_mechanism[i].planet_gaseous.Mmin_solution_samples)
        y, x_edges = np.histogram(tps_mechanism[i].planet_gaseous.Mmin_solution_samples[g], bins=Nbins[i+1])
        x_edges = _get_bin_edges(x_edges, logx)
        axs[i].plot(x_edges, y, ls='-', color=colour_mechanism[i], lw=2.5, drawstyle='steps-mid',
                    label='Model\nminimum\nmass')
        axs[i].fill_between(x_edges, np.zeros(Nbins[i+1]), y, color=colour_mechanism[i], 
                            alpha=.2, step='mid')
        axs[i].axvline(tps_mechanism[i].planet_gaseous.Mmin_solution[0], ls=':', lw=.9, 
                       color=colour_mechanism[i])
        
        # plot measured gaseous planet mass
        y2, x_edges = np.histogram(self.planet_gaseous.mpsamples, bins=Nbins[0])
        x_edges = _get_bin_edges(x_edges, logx)
        axs[i].plot(x_edges, y2*y.max()/y2.max(), ls='--', color='k', lw=2.5, drawstyle='steps-mid',
                    label='Measured\nmass')
        axs[i].fill_between(x_edges, np.zeros(Nbins[0]), y2*y.max()/y2.max(), color='k', 
                            alpha=.2, step='mid')
        axs[i].axvline(two_planet_system.planet_gaseous.mass[0], ls=':', lw=.8, color='k')

        # customize the plot
        consistency = tps_mechanism[i].planet_gaseous.frac_consistent
        axs[i].set_title('%s\n(consistency rate = %i%%)'%(label_mechanism[i],
                                                          consistency*1e2),
                         fontsize=10)
        axs[i].set_xlabel('Gaseous planet mass [M$_{\oplus}$]', fontsize=10)
        axs[i].set_ylim((0, axs[i].get_ylim()[1]))
        xmin, xmax = axs[i].get_xlim()
        axs[i].set_xlim((xmin, xmax*1.2))
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend(handles[::-1], labels[::-1], fontsize=8)
        if i == 0 : axs[i].set_ylabel('Number of realizations', fontsize=10)
        if logx: axs[i].set_xscale('log')
        
    fig.subplots_adjust(bottom=.12, top=.9, right=.97, left=.12)
    # save file if desired
    if outfile != '':
        plt.savefig(outfile)
        
        
def _get_bin_edges(bin_edges, log=False):
    if log:
        return 10**(np.log10(bin_edges)[:-1] + \
                    .5 * np.diff(np.log10(bin_edges))[0])
    else:
        return bin_edges[:-1] + .5 * np.diff(bin_edges)[0]
