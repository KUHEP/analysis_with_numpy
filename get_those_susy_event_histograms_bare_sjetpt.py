#!/usr/bin/env python

"""
New thing to try out, doing analysis with numpy, converting back to ROOT to do histogramming
Creator: Erich Schmitz
Date: Feb 22, 2019
"""

import ROOT as rt
import numpy as np
import root_numpy as rnp
import numpy.lib.recfunctions as rfc
import os
from get_those_tree_objects_with_numpy import *
from collections import OrderedDict
import time
rt.gROOT.SetBatch()
rt.TH1.AddDirectory(rt.kFALSE)

########## get_histograms function template ############

########################################################


def get_histograms(list_of_files_, variable_list_, cuts_to_apply_=None):
    
    hist = OrderedDict()
    counts = OrderedDict()
    for sample in list_of_files_:
        hist[sample] = OrderedDict()
        counts[sample] = OrderedDict()
        for tree_name in list_of_files_[sample]['trees']:
            print '\nReserving Histograms for:', sample, tree_name 
            hist[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve histograms
#            hist[sample][tree_name]['MET'] = rt.TH1D('MET_'+sample+'_'+tree_name, 'E_{T}^{miss} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['pt_jet'] = rt.TH1D('pt_jet_'+sample+'_'+tree_name, 'pt jet', 256, 0, 1000)
            hist[sample][tree_name]['N_jet'] = rt.TH1D('pt_jet_'+sample+'_'+tree_name, 'n jet', 50, 0, 50)
            hist[sample][tree_name]['pt_N_jet'] = rt.TH2D('pt_jet_'+sample+'_'+tree_name, 'pt n jet', 256, 0, 1000, 50, 0, 50)
            hist[sample][tree_name]['S_jet_pt'] = rt.TH1D('S_jet_pt_'+sample+'_'+tree_name, 's pt', 256, 0, 1000)
            hist[sample][tree_name]['ISR_jet_pt'] = rt.TH1D('ISR_jet_pt_'+sample+'_'+tree_name, 'isr pt', 256, 0, 1000)
            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                met = np.array(sample_array[sample][tree_name]['MET'])
                risr = np.array(sample_array[sample][tree_name]['RISR'])
                ptisr = np.array(sample_array[sample][tree_name]['PTISR'])
                weight = np.array(sample_array[sample][tree_name]['weight'])
                pt_jet = np.array(sample_array[sample][tree_name]['PT_jet'])
                isr_index_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                s_index_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                mini_lep = np.array(sample_array[sample][tree_name]['MiniIso_lep'])

                risr = np.array([entry[:3] for entry in risr])
                ptisr = np.array([entry[:3] for entry in ptisr])
                isr_index_jet = np.array([entry[:3] for entry in isr_index_jet])
                s_index_jet = np.array([entry[:3] for entry in s_index_jet])

                risr = risr[:, 1]
                ptisr = ptisr[:, 1]
                isr_index_jet = isr_index_jet[:, 1]
                s_index_jet = s_index_jet[:, 1]

                pt_s_jet = np.array([ jet[index] for jet, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = np.array([ jet[index] for jet, index in zip(pt_jet, isr_index_jet)])

                n_jet = np.array([len(jets) for jets in pt_jet])
                max_n_jet = np.amax(n_jet)
                jet_weight = np.array([np.array([np.float64(event)]*len(jets)) for jets, event in zip(pt_jet, weight)]) 
                n_jets_plural = np.array([np.array([np.float64(event)]*len(jets)) for jets, event in zip(pt_jet, n_jet)]) 
                pt_jet = np.array([np.pad(jets, (0, max_n_jet - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                n_jets_plural = np.array([np.pad(jets, (0, max_n_jet - len(jets)), 'constant', constant_values=np.nan) for jets in n_jets_plural]) 

                s_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, s_index_jet)])
                isr_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, isr_index_jet)])

                s_len_jet = np.array([len(jets) for jets in pt_s_jet])
                max_n_s_jets = np.amax(s_len_jet)
                pt_s_jet = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_s_jet]) 
                s_jet_weight = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in s_jet_weight]) 
                isr_len_jet = np.array([len(jets) for jets in pt_isr_jet])
                max_n_isr_jets = np.amax(isr_len_jet)
                pt_isr_jet = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_isr_jet])
                isr_jet_weight = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in isr_jet_weight]) 
 
                ge_1_mini_lep = np.array([True if len(mini[mini>0.1]) > 0 else False for mini in mini_lep])
                risr_0p8 = risr > 0.8
                met_200 = met > 200
                ptisr_200 = ptisr > 200

                evt_selection_mask = np.all([met_200, risr_0p8, ptisr_200, ge_1_mini_lep], axis=0)

                pt_s_jet = pt_s_jet[evt_selection_mask]
                pt_isr_jet = pt_isr_jet[evt_selection_mask]
                s_jet_weight = s_jet_weight[evt_selection_mask]
                isr_jet_weight = isr_jet_weight[evt_selection_mask]

                pt_jet  = pt_jet[evt_selection_mask]
                n_jets_plural = n_jets_plural[evt_selection_mask]
 
                s_jet_weight = s_jet_weight[~np.isnan(s_jet_weight)]
                pt_s_jet = pt_s_jet[~np.isnan(pt_s_jet)]
                isr_jet_weight = isr_jet_weight[~np.isnan(isr_jet_weight)]
                pt_isr_jet = pt_isr_jet[~np.isnan(pt_isr_jet)]
                pt_jet = pt_jet[~np.isnan(pt_jet)]
                n_jets_plural = n_jets_plural[~np.isnan(n_jets_plural)]
                
                if not np.any(evt_selection_mask): 
                    print 'finished filling'
                    continue
                
                
#                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)
                #rnp.fill_hist(hist[sample][tree_name]['S_jet_pt'], np.concatenate(pt_s_jet), np.concatenate(s_jet_weight))
                #rnp.fill_hist(hist[sample][tree_name]['ISR_jet_pt'], np.concatenate(pt_isr_jet), np.concatenate(isr_jet_weight))
                #rnp.fill_hist(hist[sample][tree_name]['S_jet_pt'], pt_s_jet, s_jet_weight)
                #rnp.fill_hist(hist[sample][tree_name]['ISR_jet_pt'], pt_isr_jet, isr_jet_weight)
                #rnp.fill_hist(hist[sample][tree_name]['pt_jet'], np.concatenate(pt_jet), np.concatenate(jet_weight))
                #rnp.fill_hist(hist[sample][tree_name]['N_jet'], n_jet, weight)
                #rnp.fill_hist(hist[sample][tree_name]['pt_N_jet'], np.swapaxes([np.concatenate(pt_jet), np.concatenate(n_jets_plural)],0,1), np.concatenate(jet_weight))
                
                rnp.fill_hist(hist[sample][tree_name]['S_jet_pt'], pt_s_jet)
                rnp.fill_hist(hist[sample][tree_name]['ISR_jet_pt'], pt_isr_jet)
                rnp.fill_hist(hist[sample][tree_name]['pt_jet'], pt_jet)
                rnp.fill_hist(hist[sample][tree_name]['N_jet'], n_jet)
                rnp.fill_hist(hist[sample][tree_name]['pt_N_jet'], np.swapaxes([pt_jet, n_jets_plural],0,1))
                
 

                print 'finished filling'
    return hist


             

if __name__ == "__main__":

    signals = { 
#    'SMS-T2-4bd_420' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_',
#],
    'SMS-T2-4bd_490' : [
                    #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_v2',
],
              }
    backgrounds = {
    'TTJets_2017' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X'],
    'WJets_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
                  }
    variables = ['MET', 'RISR', 'PTISR', 'PT_jet', 'index_jet_S', 'index_jet_ISR', 'MiniIso_lep', 'weight']

    start_b = time.time()    
#    background_list = process_the_samples(backgrounds, None, None)
#    hist_background = get_histograms(background_list, variables, None)

#    write_hists_to_file(hist_background, './output_background_bare_pt_hists.root') 
    stop_b = time.time()

    signal_list = process_the_samples(signals, None, None)
    hist_signal = get_histograms(signal_list, variables, None)

    write_hists_to_file(hist_signal, './output_signal_jetpt_hists_initselection.root')  
    stop_s = time.time()

    print "background: ", stop_b - start_b
    print "signal:     ", stop_s - stop_b
    print "total:      ", stop_s - start_b
 
    print 'finished writing'
