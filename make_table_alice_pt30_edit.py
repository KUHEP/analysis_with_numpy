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
from get_numpy import *
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
            hist[sample][tree_name]['2l_0b_met_300'] = 0.
            hist[sample][tree_name]['2l_1b_met_300'] = 0.
            hist[sample][tree_name]['2l_2b_met_300'] = 0.
            hist[sample][tree_name]['1l_0b_met_300'] = 0.
            hist[sample][tree_name]['1l_1b_met_300'] = 0.
            hist[sample][tree_name]['1l_2b_met_300'] = 0.
            hist[sample][tree_name]['2l_0b_met_200'] = 0.
            hist[sample][tree_name]['2l_1b_met_200'] = 0.
            hist[sample][tree_name]['2l_2b_met_200'] = 0.
            hist[sample][tree_name]['1l_0b_met_200'] = 0.
            hist[sample][tree_name]['1l_1b_met_200'] = 0.
            hist[sample][tree_name]['1l_2b_met_200'] = 0.
# put in alice's list
            hist[sample][tree_name]['CR1aX'] = 0.
            hist[sample][tree_name]['CR1aY'] = 0.

            hist_w[sample][tree_name]['total_w'] = 0.
            hist_w[sample][tree_name]['2l_0b_met_300_w'] = 0.
            hist_w[sample][tree_name]['2l_1b_met_300_w'] = 0.
            hist_w[sample][tree_name]['2l_2b_met_300_w'] = 0.
            hist_w[sample][tree_name]['1l_0b_met_300_w'] = 0.
            hist_w[sample][tree_name]['1l_1b_met_300_w'] = 0.
            hist_w[sample][tree_name]['1l_2b_met_300_w'] = 0.
            hist_w[sample][tree_name]['2l_0b_met_200_w'] = 0.
            hist_w[sample][tree_name]['2l_1b_met_200_w'] = 0.
            hist_w[sample][tree_name]['2l_2b_met_200_w'] = 0.
            hist_w[sample][tree_name]['1l_0b_met_200_w'] = 0.
            hist_w[sample][tree_name]['1l_1b_met_200_w'] = 0.
            hist_w[sample][tree_name]['1l_2b_met_200_w'] = 0.
            hist_w[sample][tree_name]['CR1aX_w'] = 0.
            hist_w[sample][tree_name]['CR1aY_w'] = 0.

            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                base_risr = np.array(sample_array[sample][tree_name]['RISR'])
                base_ptisr = np.array(sample_array[sample][tree_name]['PTISR'])
                base_ptcm = np.array(sample_array[sample][tree_name]['PTCM'])
                pt_jet = np.array(sample_array[sample][tree_name]['PT_jet'])
                flavor_jet = np.array(sample_array[sample][tree_name]['Flavor_jet'])
                phi_jet = np.array(sample_array[sample][tree_name]['Phi_jet'])
                base_isr_index_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                base_s_index_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                bjet_tag = np.array(sample_array[sample][tree_name]['Btag_jet'])

                pt_lep = np.array(sample_array[sample][tree_name]['PT_lep'])
                ch_lep = np.array(sample_array[sample][tree_name]['Charge_lep'])
                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])
                eta_lep = np.array(sample_array[sample][tree_name]['Eta_lep'])
                phi_lep = np.array(sample_array[sample][tree_name]['Phi_lep'])
                miniiso_lep = np.array(sample_array[sample][tree_name]['MiniIso_lep'])
                pdgid_lep = np.array(sample_array[sample][tree_name]['PDGID_lep'])
                base_isr_index_lep = np.array(sample_array[sample][tree_name]['index_lep_ISR'])
                base_s_index_lep = np.array(sample_array[sample][tree_name]['index_lep_S'])

                risr = np.array([entry[:3] for entry in base_risr])
                ptisr = np.array([entry[:3] for entry in base_ptisr])
                ptcm = np.array([entry[:3] for entry in base_ptcm])
                #dphi = np.array([entry[:3] for entry in base_dphi])
                isr_index_jet = np.array([entry[:3] for entry in base_isr_index_jet])
                s_index_jet = np.array([entry[:3] for entry in base_s_index_jet])
                isr_index_lep = np.array([entry[:3] for entry in base_isr_index_lep])
                s_index_lep = np.array([entry[:3] for entry in base_s_index_lep])
                
                risr = risr[:, 2]
                ptisr = ptisr[:, 2]
                ptcm = ptcm[:, 2]
                #dphi = dphi[:, 2]
                isr_index_jet = isr_index_jet[:, 2]
                s_index_jet = s_index_jet[:, 2]
                isr_index_lep = isr_index_lep[:, 2]
                s_index_lep = s_index_lep[:, 2]

                #risr = risr[:, 1]
                #ptisr = ptisr[:, 1]
                #ptcm = ptcm[:, 1]
                #isr_index_jet = isr_index_jet[:, 1]
                #s_index_jet = s_index_jet[:, 1]
                #isr_index_lep = isr_index_lep[:, 1]
                #s_index_lep = s_index_lep[:, 1]
                #dphi = dphi[:, 1]

                # risr_lepV_jetI = risr[:,0]
                # risr_lepV_jetA = risr[:,1]
                # risr_lepA_jetA = risr[:,2]

                met = np.array(sample_array[sample][tree_name]['MET'])
                weight = np.array(sample_array[sample][tree_name]['weight'])
                weight = 137. * weight
                phi_met = np.array(sample_array[sample][tree_name]['MET_phi'])

###########################################################################################################
                mask_pt_jet_30 = np.array([jet > 30 for jet in pt_jet])
                
                pt_s_jet = np.array([jets[index] for jets, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = np.array([jets[index] for jets, index in zip(pt_jet, isr_index_jet)])

                pt_jet = np.array([jet[jet > 30] for jet in pt_jet])
                bjet_tag = np.array([tag[mask] for tag, mask in zip(bjet_tag, mask_pt_jet_30)])
                phi_jet = 
###########################################################################################################
                len_jet = np.array([len(jets) for jets in pt_jet])
                max_n_jets = np.amax(len_jet)

                ht = np.array([np.sum(pt) for pt in pt_jet])

                pt_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                bjet_tag = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in bjet_tag]) 
                phi_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in phi_jet]) 

                len_lep = np.array([len(leps) for leps in pt_lep])
                max_n_leps = np.amax(len_lep)

                pt_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_lep]) 
                phi_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in phi_lep]) 
                miniiso_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in miniiso_lep]) 

                pt_mini_lep = np.array([pt[mini<0.1] for pt, mini in zip(pt_lep, miniiso_lep)])
                pt_5_mini_lep = np.array([ pt[pt>30] for pt in pt_mini_lep])
                pt_20_lep = np.array([ pt[pt>20] for pt in pt_lep])

                dphi_lmet = phi_lep[:,0] - phi_met
                dphi_lmet = np.array([ phi + 2*np.pi if phi < -np.pi else phi for phi in dphi_lmet])
                dphi_lmet = np.array([ phi - 2*np.pi if phi > np.pi else phi for phi in dphi_lmet])
                mt = np.sqrt(2 * pt_lep[:,0] * met * (1-np.cos(dphi_lmet)))

                print '\ncreating masks'
                print '-> bjet masks'
                loose_mask = bjet_tag > 0.5426
                medium_mask = bjet_tag > 0.8484
                tight_mask = bjet_tag > 0.9535

                medium_s_mask = np.array([tag[index] > 0.8484 for tag, index in zip(bjet_tag, s_index_jet)])

                print 'jet masks'                
                no_more_than_2_60_jets = np.array([True if len(jets[jets>60]) <= 2 else False for jets in pt_jet])

                lead_jet_100 = np.array([True if jets[0] > 100 else False for jets in pt_jet])

                print 'bjet event masks'
                has_2_loose = np.array([True if len(mask[mask]) >= 2 else False for mask in loose_mask])
                has_2_medium = np.array([True if (len(mask[mask]) >= 2 and mask[0] is not True) or (len(mask[mask]) >= 3 and (mask[0] is True)) else False for mask in medium_mask])
                has_2_s_medium = np.array([True if len(mask[mask]) >= 2 else False for mask in medium_s_mask])
                has_2_tight = np.array([True if len(mask[mask]) >= 2 else False for mask in tight_mask])

                has_1_loose = np.array([np.any(event) for event in loose_mask])
                has_1_medium = np.array([np.any(event[1:]) for event in medium_mask])
                has_1_s_medium = np.array([np.any(event) for event in medium_s_mask])
                has_1_tight = np.array([np.any(event) for event in tight_mask])

                has_no_loose = np.array([False if np.any(mask) else True for mask in loose_mask])
                has_no_medium = np.array([False if np.any(mask[1:]) else True for mask in medium_mask])
                has_no_s_medium = np.array([False if np.any(mask) else True for mask in medium_s_mask])
                has_no_tight = np.array([False if np.any(mask) else True for mask in tight_mask])

                print 'lepton masks'
                only_1_pt5_mini_lep = np.array([True if len(lep) == 1 else False for lep in pt_5_mini_lep])
                only_2_pt5_mini_lep = np.array([True if len(lep) == 2 else False for lep in pt_5_mini_lep])

                only_2_pt5_mini_opp_lep = np.array([True if len(lep) == 2 and len(charge[charge>0])>0 and len(charge[charge<0])>0 else False for lep, charge in zip(pt_5_mini_lep, ch_lep)])
                ct1 = mp.amin(np.swapaxes([met,ht-100.],0,1), axis=0)

                met_300 = met > 300
                met_200 = met > 200
                CT1Y = ct1 > 400
                CT1X = ct1 < 400

                ht_300 = ht > 300

                risr_0p5 = risr > 0.5


                one_pt5_mini_lep_0_s_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_no_s_medium, risr_0p5, met_200]), axis=0)
                two_pt5_mini_lep_0_s_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_no_s_medium, risr_0p5, met_200]), axis=0)
                one_pt5_mini_lep_1_s_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_1_s_medium, risr_0p5, met_200]), axis=0)
                two_pt5_mini_lep_1_s_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_1_s_medium, risr_0p5, met_200]), axis=0)
                one_pt5_mini_lep_2_s_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_2_s_medium, risr_0p5, met_200]), axis=0)
                two_pt5_mini_lep_2_s_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_2_s_medium, risr_0p5, met_200]), axis=0)

                one_pt5_mini_lep_1_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_1_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
                two_pt5_mini_lep_1_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_1_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
                one_pt5_mini_lep_0_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_no_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
#                two_pt5_mini_lep_0_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_no_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
#                one_pt5_mini_lep_2_med_mask = np.all(np.array([only_1_pt5_mini_lep, has_2_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
#                two_pt5_mini_lep_2_med_mask = np.all(np.array([only_2_pt5_mini_lep, has_2_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300]), axis=0)
# alice's control region masks
                control_r_1aX_mask = np.all(np.array([only_1_pt5_mini_lep, has_no_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300, CT1X]), axis=0)
                control_r_1aY_mask = np.all(np.array([only_1_pt5_mini_lep, has_no_medium, ht_300, no_more_than_2_60_jets, lead_jet_100, met_300, CT1Y]), axis=0)
 

                hist[sample][tree_name]['total'] += len(met)
#                hist[sample][tree_name]['2l_0b_met_300'] += len(met[two_pt5_mini_lep_0_med_mask])
#                hist[sample][tree_name]['2l_1b_met_300'] += len(met[two_pt5_mini_lep_1_med_mask])
#                hist[sample][tree_name]['2l_2b_met_300'] += len(met[two_pt5_mini_lep_2_med_mask])
                hist[sample][tree_name]['1l_0b_met_300'] += len(met[one_pt5_mini_lep_0_med_mask])
                hist[sample][tree_name]['1l_1b_met_300'] += len(met[one_pt5_mini_lep_1_med_mask])
#                hist[sample][tree_name]['1l_2b_met_300'] += len(met[one_pt5_mini_lep_2_med_mask])
#                hist[sample][tree_name]['2l_0b_met_200'] += len(met[two_pt5_mini_lep_0_s_med_mask])
#                hist[sample][tree_name]['2l_1b_met_200'] += len(met[two_pt5_mini_lep_1_s_med_mask])
#                hist[sample][tree_name]['2l_2b_met_200'] += len(met[two_pt5_mini_lep_2_s_med_mask])
                hist[sample][tree_name]['1l_0b_met_200'] += len(met[one_pt5_mini_lep_0_s_med_mask])
                hist[sample][tree_name]['1l_1b_met_200'] += len(met[one_pt5_mini_lep_1_s_med_mask])
#                hist[sample][tree_name]['1l_2b_met_200'] += len(met[one_pt5_mini_lep_2_s_med_mask])
                hist[sample][tree_name]['CR1aX'] += len(met[control_r_1aX_mask])
                hist[sample][tree_name]['CR1aY'] += len(met[control_r_1aY_mask])

                
                hist_w[sample][tree_name]['total_w'] += np.sum(weight) 
#                hist_w[sample][tree_name]['2l_0b_met_300_w'] += np.sum(weight[two_pt5_mini_lep_0_med_mask])
#                hist_w[sample][tree_name]['2l_1b_met_300_w'] += np.sum(weight[two_pt5_mini_lep_1_med_mask])
#                hist_w[sample][tree_name]['2l_2b_met_300_w'] += np.sum(weight[two_pt5_mini_lep_2_med_mask])
                hist_w[sample][tree_name]['1l_0b_met_300_w'] += np.sum(weight[one_pt5_mini_lep_0_med_mask])
                hist_w[sample][tree_name]['1l_1b_met_300_w'] += np.sum(weight[one_pt5_mini_lep_1_med_mask])
#                hist_w[sample][tree_name]['1l_2b_met_300_w'] += np.sum(weight[one_pt5_mini_lep_2_med_mask])
#                hist_w[sample][tree_name]['2l_0b_met_200_w'] += np.sum(weight[two_pt5_mini_lep_0_s_med_mask])
#                hist_w[sample][tree_name]['2l_1b_met_200_w'] += np.sum(weight[two_pt5_mini_lep_1_s_med_mask])
#                hist_w[sample][tree_name]['2l_2b_met_200_w'] += np.sum(weight[two_pt5_mini_lep_2_s_med_mask])
                hist_w[sample][tree_name]['1l_0b_met_200_w'] += np.sum(weight[one_pt5_mini_lep_0_s_med_mask])
                hist_w[sample][tree_name]['1l_1b_met_200_w'] += np.sum(weight[one_pt5_mini_lep_1_s_med_mask])
#                hist_w[sample][tree_name]['1l_2b_met_200_w'] += np.sum(weight[one_pt5_mini_lep_2_s_med_mask])
                hist_w[sample][tree_name]['CR1aX_w'] += np.sum(weight[control_r_1aX_mask])
                hist_w[sample][tree_name]['CR1aY_w'] += np.sum(weight[control_r_1aY_mask])


                print 'finished filling'
    return hist, hist_w


             

if __name__ == "__main__":

    samples = { 
    #'SMS-T2bW' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2bW/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_TuneCUETP'],
#    'SMS-T2-4bd_420' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
#],
    'SMS-T2-4bd_490' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/SMS-T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
],
#    'TTJets_2017' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ttbar_2017/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_TuneCP5'],
    'WJets_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/Wjets_2017/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17',
],
#    'ST_2017' : [
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_t-channel_antitop_5f_TuneCP5_PSweights_13TeV-powheg-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_Fall17',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples/ST_2017/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_Fall17',
#              ],
                  }
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'weight', 'MiniIso_lep','Phi_jet','Phi_lep','Eta_lep','PTISR','PTCM','MET_phi']

    start_b = time.time()    
    sample_list = process_the_samples(samples, None, None)
    sample_arrays, sample_w_arrays = make_tables(sample_list, variables, None)

    write_table(sample_arrays, './output_table_analysis_comparison.txt')  
    write_table(sample_w_arrays, './output_table_weighted.txt')  
    stop_b = time.time()

    print "total: ", stop_b - start_b
 
    print 'finished writing'
