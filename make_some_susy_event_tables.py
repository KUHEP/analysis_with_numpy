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


def make_tables(list_of_files_, variable_list_, cuts_to_apply_=None):
    
    hist = OrderedDict()
    hist_w = OrderedDict()
    counts = OrderedDict()
    for sample in list_of_files_:
        hist[sample] = OrderedDict()
        hist_w[sample] = OrderedDict()
        counts[sample] = OrderedDict()
        for tree_name in list_of_files_[sample]['trees']:
            print '\nReserving Tables for:', sample, tree_name 
            hist[sample][tree_name] = OrderedDict()
            hist_w[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve tables

            hist[sample][tree_name]['total'] = 0.
            hist_w[sample][tree_name]['total_w'] = 0.
            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                met = np.array(sample_array[sample][tree_name]['MET'])
                weight = np.array(sample_array[sample][tree_name]['weight'])

                hist[sample][tree_name]['total'] += len(met)
                hist_w[sample][tree_name]['total_w'] += np.sum(weight) 

                print 'finished filling'
    return hist, hist_w


             

if __name__ == "__main__":
    signals = { 
    'SMS-T2-4bd_490_I' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_SMS_I/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'SMS-T2-4bd_490_IV' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_SMS_IV/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
              }
    backgrounds = {
#    'TTJets_2017_I' : ['/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root'],
    'WJets_2017_I' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
],
#    'WJets_2017_IV' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_WJets_IV/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#],
#    'ZJets_2017' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-100To200_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-200To400_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-400To600_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-600To800_13TeV-madgraph_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZJetsToNuNu_HT-800To1200_13TeV-madgraph_Fall17_94X.root',
#],
#    'DY_M50_2017' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'WW_2017' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWTo4Q_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWToLNuQQ_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#],
#    'ZZ_2017' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZZTo2L2Nu_13TeV_powheg_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZZTo2Q2Nu_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZZTo4L_13TeV_powheg_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#],
#    'WZ_2017' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_v2_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X.root',
#                    '/home/t3-ku/crogan/NTUPLES/NANO/Fall17_94X/WZ_TuneCP5_13TeV-pythia8_Fall17_94X.root',
#],
                  }

    variables = ['MET', 'weight']

    start_b = time.time()    
    sample_list = process_the_samples(backgrounds, None, None)
    sample_arrays, sample_w_arrays = make_tables(sample_list, variables, None)

    write_table(sample_arrays, sample_w_arrays, './output_table_backgrounds_raw.txt')  
    stop_b = time.time()

    print "total: ", stop_b - start_b
 
    print 'finished writing'
