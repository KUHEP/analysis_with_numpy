from __future__ import print_function
import ROOT as rt
import root_numpy as rnp
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

rt.TH1.AddDirectory(rt.kFALSE)

def dict_of_hists(in_file_, hists_to_take_ = ['Mu_MS']):
    """
    Inputs:
    @in_file_ : root file with histograms

    Return:
    list of histograms (only taking specified variables) scaled to 1.0
    """
    open_file =  rt.TFile(in_file_, 'r')

    hists = OrderedDict()
    hists['10'] = []
    hists['20'] = []
    hists['30'] = []
    hists['40'] = []
    hists['50'] = []
    hists['60'] = []
    hists['70'] = []
    hists['80'] = []
    #hists['100'] = []
    #hists['125'] = []
    #hists['150'] = []
    #hists['175'] = []
    #hists['200'] = []
   
    for key in open_file.GetListOfKeys():
        key_name = key.GetName()

        sample = key_name.split('__')[0]
        mass_point = key_name.split('__')[1]
        hist_name = key_name.split('__')[2]

        if hist_name not in hists_to_take_: continue

        stop_mass = mass_point.split('_')[1]
        lsp_mass = mass_point.split('_')[2]
       
        if int(stop_mass) > 802: continue
       
        last_stop_digit = stop_mass[-1]
        last_lsp_digit = lsp_mass[-1]

        if last_stop_digit == '4' or last_stop_digit == '6':
            stop_mass = stop_mass[:-1]+'5'
        elif last_stop_digit == '2':
            stop_mass = stop_mass[:-1]+'0'

        if last_lsp_digit == '4' or last_lsp_digit == '6':
            lsp_mass = lsp_mass[:-1]+'5'
        elif last_lsp_digit == '2':
            lsp_mass = lsp_mass[:-1]+'0'

        delta_m = str(int(stop_mass) - int(lsp_mass))

        if delta_m in hists.keys():
            hists[delta_m].append(((stop_mass, lsp_mass, sample), open_file.Get(key_name)))

    return hists
        

def sorting_to_arrays(hist_dict_):

    array_dict = OrderedDict()

    for delta_m in hist_dict_:
        if delta_m not in array_dict.keys():
            array_dict[delta_m] = OrderedDict()

        for hist_tuple in hist_dict_[delta_m]:
            sample = hist_tuple[0][2]

            if sample not in array_dict[delta_m].keys():
                array_dict[delta_m][sample] = []
            
            hist_tuple[1].Rebin(8)
         
            try:
                hist_tuple[1].Scale(1./hist_tuple[1].Integral())
            except:
                print(sample, hist_tuple[0], "skipped with no events")
                continue

            tmp_array = rnp.hist2array(hist_tuple[1], return_edges=True)

            array_dict[delta_m][sample].append(((hist_tuple[0][0],hist_tuple[0][1]), tmp_array))

    return array_dict


def make_grid(arrays_, sample_to_plot_='SMS-T2bW_dM'):

    n_rows_all = [len(arrays_[dm][sample_to_plot_]) for dm in arrays_ if sample_to_plot_ in arrays_[dm].keys()]
    n_rows = max(n_rows_all)
    n_cols = len(arrays_)

    x_values = []
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    
    for idm, dm in enumerate(arrays_):
        if sample_to_plot_ not in arrays_[dm].keys(): continue
        for imass, mass in enumerate(arrays_[dm][sample_to_plot_]):
            mass_text = mass[0][0]
            y_values = mass[1][0]
            if not x_values:
                x_tmp = mass[1][1][0]
                for i in xrange(len(x_tmp)):
                    if i+1 == len(x_tmp): continue
                    x_values.append((x_tmp[i+1] + x_tmp[i]) / 2.)
                
            ax[imass, idm].plot(x_values, y_values, 'b')
            ax[imass, idm].text(max(x_values)/2., 0.02, mass_text,  fontsize=8)
            ax[imass, idm].tick_params(axis='both', which='major', labelsize=4)

    #fig.xlabel('\Delta M')
    #fig.ylabel('M_{Stop}')

    fig.savefig("test.pdf")

if __name__ == "__main__":
   hists = dict_of_hists('hists_signal_mass_s_sep_leps_risr_0p8_16Sep19.root')

   arrays = sorting_to_arrays(hists)
   #print(arrays['10'].keys())
   make_grid(arrays) 

   
