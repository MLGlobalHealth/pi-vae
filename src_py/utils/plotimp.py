import matplotlib as mpl
import seaborn as sns

def set_plot_props():
    sns.set_style('darkgrid')
    # sns.set_style("ticks")
    mpl.rc('axes.spines', right=False, top=False)
    mpl.rc('axes', labelsize=20)
    mpl.rc('xtick', labelsize=16, top=False)
    mpl.rc('xtick.minor', visible=False)
    mpl.rc('ytick', labelsize=16, right=False)
    mpl.rc('ytick.minor', visible=False)
    mpl.rc('savefig', bbox='tight', format='pdf')
    mpl.rc('figure', figsize=(6, 4))
    mpl.rc('legend',fontsize=16)