from imports import *


def plot_Mmin_histograms_width(two_planet_system, outfile='', binwidth=0.1, logx=False):
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
    mplims = np.zeros((Nmech+1,2))
    mplims[0] = self.planet_gaseous.mpsamples.min(), self.planet_gaseous.mpsamples.max()
    for i in range(Nmech): 
        mplims[i+1] = np.nanmin(tps_mechanism[i].planet_gaseous.Mmin_solution_samples), \
                      np.nanmax(tps_mechanism[i].planet_gaseous.Mmin_solution_samples)
    bins = np.arange(mplims[:,0].min(), mplims[:,1].max(), binwidth)

    # plot
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8,4))
    #axs = [axs] if Nmech == 1 else axs
    ymaxes, Mmin_meds = np.zeros(Nmech), np.zeros(Nmech)
    for i in range(Nmech):

        # plot minimum masses for this mechanism if available
        g = np.isfinite(tps_mechanism[i].planet_gaseous.Mmin_solution_samples)
        y, x_edges = np.histogram(tps_mechanism[i].planet_gaseous.Mmin_solution_samples[g], bins=bins)
        ymaxes[i] = y.max()
        x_edges = _get_bin_edges(x_edges, logx)
        axs.plot(x_edges, y, ls='-', color=colour_mechanism[i], lw=2.5, drawstyle='steps-mid',
                 label='%s\nminimum mass (%.2f)'%(label_mechanism[i], 
                                                  tps_mechanism[i].planet_gaseous.frac_consistent))
        axs.fill_between(x_edges, np.zeros(y.size), y, color=colour_mechanism[i], alpha=.2, step='mid')
        Mmin_meds[i] = tps_mechanism[i].planet_gaseous.Mmin_solution[0]
        axs.axvline(Mmin_meds[i], ls=':', lw=1.6, color=colour_mechanism[i])
    
    # plot measured gaseous planet mass
    y2, x_edges = np.histogram(self.planet_gaseous.mpsamples, bins=bins)
    x_edges = _get_bin_edges(x_edges, logx)
    axs.plot(x_edges, y2*ymaxes.max()/y2.max(), ls='--', color='k', lw=2.5, drawstyle='steps-mid',
             label='Measured mass')
    axs.fill_between(x_edges, np.zeros(y2.size), y2*ymaxes.max()/y2.max(), color='k', 
                     alpha=.2, step='mid')
    axs.axvline(self.planet_gaseous.mass[0], ls=':', lw=1.6, color='k')

    # customize the plot
    axs.set_ylabel('Number of realizations', fontsize=12)
    axs.set_xlabel('Gaseous planet mass [M$_{\oplus}$]', fontsize=12)
    axs.set_ylim((0, axs.get_ylim()[1]))
    xmin, xmax = axs.get_xlim()
    axs.set_xlim((xmin, xmax*1.2))
    handles, labels = axs.get_legend_handles_labels()
    s = np.append(Nmech, np.argsort(Mmin_meds))
    axs.legend(np.array(handles)[s], np.array(labels)[s], fontsize=10, loc='upper right')
    if logx: axs.set_xscale('log')
    
    fig.subplots_adjust(bottom=.14, top=.97, right=.97, left=.08)
    # save file if desired
    if outfile != '':
        plt.savefig(outfile)

    
        
def _get_bin_edges(bin_edges, log=False):
    if log:
        return 10**(np.log10(bin_edges)[:-1] + \
                    .5 * np.diff(np.log10(bin_edges))[0])
    else:
        return bin_edges[:-1] + .5 * np.diff(bin_edges)[0]
