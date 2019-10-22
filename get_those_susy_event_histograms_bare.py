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
            hist[sample][tree_name]['ID_lep_precut'] = rt.TH1D('ID_lep_precut_'+sample+'_'+tree_name, 'ID_lep', 10, 0, 10)
            hist[sample][tree_name]['ID_lep_medium'] = rt.TH1D('ID_lep_medium_'+sample+'_'+tree_name, 'ID_lep', 10, 0, 10)
            hist[sample][tree_name]['ID_lep_tight'] = rt.TH1D('ID_lep_tight_'+sample+'_'+tree_name, 'ID_lep', 10, 0, 10)
            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                #met = np.array(sample_array[sample][tree_name]['MET'])
                #weight = np.array(sample_array[sample][tree_name]['weight'])
                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])

                len_lep = np.array([len(leps) for leps in id_lep])
                max_n_leps = np.amax(len_lep)
 
                id_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=0) for leps in id_lep]) 
                 
                medium_id_mask = id_lep >= 3
                tight_id_mask = id_lep >= 4

                medium_id_lep = id_lep[medium_id_mask]
                tight_id_lep = id_lep[tight_id_mask]

                id_lep = id_lep[~np.isnan(id_lep)] 
                #if not np.any(evt_selection_mask): 
                #    print 'finished filling'
                #    continue
                
                
#                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)
                rnp.fill_hist(hist[sample][tree_name]['ID_lep_precut'], id_lep)
                rnp.fill_hist(hist[sample][tree_name]['ID_lep_medium'], medium_id_lep)
                rnp.fill_hist(hist[sample][tree_name]['ID_lep_tight'], tight_id_lep)

                print 'finished filling'
    return hist


             

if __name__ == "__main__":

    signals = { 
    #'SMS-T2bW' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2bW_v2/root/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_TuneCUETP']
    'SMS-T2-4bd_420' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
],
    'SMS-T2-4bd_490' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
],
              }
    backgrounds = {
    'TTJets' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ttbar_2017/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_TuneCP5'],
    'ST' : [
              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_t-channel_antitop_5f_TuneCP5_PSweights_13TeV-powheg-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8_Fall17',
              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_Fall17',
              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_Fall17',
           ],
    'WJets_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/Wjets_2017/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
],
                  }
    variables = ['MET', 'ID_lep', 'weight']

    start_b = time.time()    
    background_list = process_the_samples(backgrounds, 3, None)
    hist_background = get_histograms(background_list, variables, None)

    write_hists_to_file(hist_background, './output_background_bare_hists.root') 
    stop_b = time.time()

    signal_list = process_the_samples(signals, None, None)
    hist_signal = get_histograms(signal_list, variables, None)

    write_hists_to_file(hist_signal, './output_signal_bare_hists.root')  
    stop_s = time.time()

    print "background: ", stop_b - start_b
    print "signal:     ", stop_s - stop_b
    print "total:      ", stop_s - start_b
 
    print 'finished writing'
