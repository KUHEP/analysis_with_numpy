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
    
    hist_s = OrderedDict()
    hist_isr = OrderedDict()
    hist_s_w = OrderedDict()
    hist_isr_w = OrderedDict()
    counts = OrderedDict()
    for sample in list_of_files_:
        hist_s[sample] = OrderedDict()
        hist_isr[sample] = OrderedDict()
        hist_s_w[sample] = OrderedDict()
        hist_isr_w[sample] = OrderedDict()
        counts[sample] = OrderedDict()
        for tree_name in list_of_files_[sample]['trees']:
            print '\nReserving Tables for:', sample, tree_name 
            hist_s[sample][tree_name] = OrderedDict()
            hist_isr[sample][tree_name] = OrderedDict()
            hist_s_w[sample][tree_name] = OrderedDict()
            hist_isr_w[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve tables

            hist_s[sample][tree_name]['total'] = 0.
            hist_s[sample][tree_name]['2l_0sb'] = 0.
            hist_s[sample][tree_name]['2l_greateq_2sb'] = 0.
            hist_s[sample][tree_name]['2l_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['2l_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['2l_0sj'] = 0.
            hist_s[sample][tree_name]['2l_1sj'] = 0.
            hist_s[sample][tree_name]['2l_greateq_1sj'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_0sb'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_greateq_2sb'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_0sj'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sj'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_greateq_1sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_0sb'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_greateq_2sb'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_0sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_greateq_1sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_0sb'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_greateq_2sb'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_0sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_greateq_1sj'] = 0.

            hist_isr[sample][tree_name]['total'] = 0.
            hist_isr[sample][tree_name]['2l_0isrb'] = 0.
            hist_isr[sample][tree_name]['2l_greateq_2isrb'] = 0.
            hist_isr[sample][tree_name]['2l_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['2l_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['2l_0isrj'] = 0.
            hist_isr[sample][tree_name]['2l_1isrj'] = 0.
            hist_isr[sample][tree_name]['2l_greateq_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_greateq_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_greateq_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_greateq_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_greateq_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_greateq_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_greateq_1isrj'] = 0.


            hist_s_w[sample][tree_name]['total_w'] = 0.
            hist_s_w[sample][tree_name]['2l_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['2l_greateq_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['2l_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['2l_greateq_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_greateq_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_greateq_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_greateq_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_greateq_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_greateq_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_greateq_1sj_w'] = 0.

            hist_isr_w[sample][tree_name]['total_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_greateq_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_greateq_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_greateq_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_greateq_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_greateq_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_greateq_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_greateq_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_greateq_1isrj_w'] = 0.
            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                risr = np.array(sample_array[sample][tree_name]['RISR'])
                ptisr = np.array(sample_array[sample][tree_name]['PTISR'])

                pt_jet = np.array(sample_array[sample][tree_name]['PT_jet'])
                isr_index_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                s_index_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                bjet_tag = np.array(sample_array[sample][tree_name]['Btag_jet'])

                pt_lep = np.array(sample_array[sample][tree_name]['PT_lep'])
                mini_lep = np.array(sample_array[sample][tree_name]['MiniIso_lep'])
                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])

                risr = np.array([entry[:2] for entry in risr])
                ptisr = np.array([entry[:2] for entry in ptisr])
                #ptcm = np.array([entry[:2] for entry in ptcm])
                #dphi = np.array([entry[:2] for entry in dphi])
                isr_index_jet = np.array([entry[:2] for entry in isr_index_jet])
                s_index_jet = np.array([entry[:2] for entry in s_index_jet])

                risr = risr[:, 1]
                ptisr = ptisr[:, 1]
                #ptcm = ptcm[:, 2]
                #dphi = dphi[:, 2]
                isr_index_jet = isr_index_jet[:, 1]
                s_index_jet = s_index_jet[:, 1]
                #isr_index_lep = isr_index_lep[:, 2]
                #s_index_lep = s_index_lep[:, 2]

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

                #if 'SMS' in sample:
                #    weight = 1000. * weight

                len_jet = np.array([len(jets) for jets in pt_jet])
                max_n_jets = np.amax(len_jet)

                pt_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                bjet_tag = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in bjet_tag]) 
                pt_s_jet = np.array([jets[index] for jets, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = np.array([jets[index] for jets, index in zip(pt_jet, isr_index_jet)])
               

                ################  Choosing Lepton ID #####################
                ################        Medium       #####################
                #pt_lep = np.array([pt[lid>=3] for pt, lid in zip(pt_lep, id_lep)])
                #mini_lep = np.array([mini[lid>=3] for mini, lid in zip(mini_lep, id_lep)])
                ################        Tight        #####################
                pt_lep = np.array([pt[lid>=4] for pt, lid in zip(pt_lep, id_lep)])
                mini_lep = np.array([mini[lid>=4] for mini, lid in zip(mini_lep, id_lep)])
                ##########################################################
 
                len_lep = np.array([len(leps) for leps in pt_lep])
                max_n_leps = np.amax(len_lep)
                pt_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_lep]) 
                mini_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in mini_lep]) 

                pt_5_lep = np.array([ pt[np.logical_and(pt>5, mini<0.1)] for pt, mini in zip(pt_lep, mini_lep)])

                pt_less_12_lep = np.array([ pt[np.logical_and(pt<12, mini<0.1)] for pt, mini in zip(pt_lep, mini_lep)])

                pt_great_12_lep = np.array([ pt[np.logical_and(pt>12, mini<0.1)] for pt, mini in zip(pt_lep, mini_lep)])
                pt_great_50_lep = np.array([ pt[np.logical_and(pt>50, mini<0.1)] for pt, mini in zip(pt_lep, mini_lep)])
                
                print '\ncreating masks and weights'
                print '-> bjet masks'
                medium_s_mask = np.array([tag[index] > 0.8484 for tag, index in zip(bjet_tag, s_index_jet)])
                medium_isr_mask = np.array([tag[index] > 0.8484 for tag, index in zip(bjet_tag, isr_index_jet)])
              

 
                has_0_s_jets = np.array([True if len(jet) < 1 else False for jet in s_index_jet])
                has_1_s_jets = np.array([True if len(jet) == 1 else False for jet in s_index_jet])
                has_greateq_1_s_jets = np.array([True if len(jet) >= 1 else False for jet in s_index_jet])
                  
                has_0_isr_jets = np.array([True if len(jet) < 1 else False for jet in isr_index_jet])
                has_1_isr_jets = np.array([True if len(jet) == 1 else False for jet in isr_index_jet])
                has_greateq_1_isr_jets = np.array([True if len(jet) >= 1 else False for jet in isr_index_jet])
                  
                has_2_medium = np.array([True if len(mask[mask]) >= 2 else False for mask in medium_s_mask])
                has_2_isr_medium = np.array([True if len(mask[mask]) >= 2 else False for mask in medium_isr_mask])

                has_1_less_pt30_medium = np.array([True if len(event[np.logical_and(pt<30, event)]) == 1 else False for event, pt in zip(medium_s_mask, pt_s_jet)])
                has_1_great_pt30_medium = np.array([True if len(event[np.logical_and(pt>30, event)]) == 1 else False for event, pt in zip(medium_s_mask, pt_s_jet)])

                has_1_less_pt30_isr_medium = np.array([True if len(event[np.logical_and(pt<30, event)]) == 1 else False for event, pt in zip(medium_isr_mask, pt_isr_jet)])
                has_1_great_pt30_isr_medium = np.array([True if len(event[np.logical_and(pt>30, event)]) == 1 else False for event, pt in zip(medium_isr_mask, pt_isr_jet)])

                has_no_medium = np.array([False if np.any(mask) else True for mask in medium_s_mask])
                has_no_isr_medium = np.array([False if np.any(mask) else True for mask in medium_isr_mask])


                print '-> lepton masks'
                only_2_lep = np.array([True if len(lep) == 2 else False for lep in pt_5_lep])

                only_1_less_pt12_lep = np.array([True if len(lep) == 1 and not two_leps else False for lep, two_leps in zip(pt_less_12_lep, only_2_lep)])
                only_1_pt12to50_lep = np.array([True if len(lep[lep<50]) == 1 and not two_leps else False for lep, two_leps in zip(pt_great_12_lep, only_2_lep)])
                only_1_great_pt50_lep = np.array([True if len(lep) == 1 and not two_leps else False for lep, two_leps in zip(pt_great_50_lep, only_2_lep)])

                met_200 = met > 200

                risr_0p8 = risr > 0.95

                ptisr_200 = ptisr > 200


                print 'incrementing tables'
                two_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_no_medium], axis=0)
                two_l_greateq_2sb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_2_medium], axis=0)
                two_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_less_pt30_medium], axis=0)
                two_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_great_pt30_medium], axis=0)
                two_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_0_s_jets], axis=0)
                two_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_s_jets], axis=0)
                two_l_greateq_1sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_greateq_1_s_jets], axis=0)

                one_less_pt12_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_no_medium], axis=0)
                one_less_pt12_l_greateq_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_2_medium], axis=0)
                one_less_pt12_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_less_pt30_medium], axis=0)
                one_less_pt12_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_great_pt30_medium], axis=0)
                one_less_pt12_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_0_s_jets], axis=0)
                one_less_pt12_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_s_jets], axis=0)
                one_less_pt12_l_greateq_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_greateq_1_s_jets], axis=0)

                one_pt12to50_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_no_medium], axis=0)
                one_pt12to50_l_greateq_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_2_medium], axis=0)
                one_pt12to50_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_less_pt30_medium], axis=0)
                one_pt12to50_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_great_pt30_medium], axis=0)
                one_pt12to50_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_0_s_jets], axis=0)
                one_pt12to50_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_s_jets], axis=0)
                one_pt12to50_l_greateq_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_greateq_1_s_jets], axis=0)

                one_great_pt50_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_no_medium], axis=0)
                one_great_pt50_l_greateq_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_2_medium], axis=0)
                one_great_pt50_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_less_pt30_medium], axis=0)
                one_great_pt50_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_great_pt30_medium], axis=0)
                one_great_pt50_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_0_s_jets], axis=0)
                one_great_pt50_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_s_jets], axis=0)
                one_great_pt50_l_greateq_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_greateq_1_s_jets], axis=0)

                two_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_no_isr_medium], axis=0)
                two_l_greateq_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_2_isr_medium], axis=0)
                two_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_less_pt30_isr_medium], axis=0)
                two_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_great_pt30_isr_medium], axis=0)
                two_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_0_isr_jets], axis=0)
                two_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_isr_jets], axis=0)
                two_l_greateq_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_greateq_1_isr_jets], axis=0)

                one_less_pt12_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_no_isr_medium], axis=0)
                one_less_pt12_l_greateq_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_2_isr_medium], axis=0)
                one_less_pt12_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_less_pt30_isr_medium], axis=0)
                one_less_pt12_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_great_pt30_isr_medium], axis=0)
                one_less_pt12_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_0_isr_jets], axis=0)
                one_less_pt12_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_isr_jets], axis=0)
                one_less_pt12_l_greateq_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_greateq_1_isr_jets], axis=0)

                one_pt12to50_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_no_isr_medium], axis=0)
                one_pt12to50_l_greateq_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_2_isr_medium], axis=0)
                one_pt12to50_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_less_pt30_isr_medium], axis=0)
                one_pt12to50_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_great_pt30_isr_medium], axis=0)
                one_pt12to50_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_0_isr_jets], axis=0)
                one_pt12to50_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_isr_jets], axis=0)
                one_pt12to50_l_greateq_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_greateq_1_isr_jets], axis=0)

                one_great_pt50_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_no_isr_medium], axis=0)
                one_great_pt50_l_greateq_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_2_isr_medium], axis=0)
                one_great_pt50_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_less_pt30_isr_medium], axis=0)
                one_great_pt50_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_great_pt30_isr_medium], axis=0)
                one_great_pt50_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_0_isr_jets], axis=0)
                one_great_pt50_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_isr_jets], axis=0)
                one_great_pt50_l_greateq_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_greateq_1_isr_jets], axis=0)


                hist_s[sample][tree_name]['total'] += len(met)
                hist_s[sample][tree_name]['2l_0sb'] += len(met[two_l_0sb])
                hist_s[sample][tree_name]['2l_greateq_2sb'] += len(met[two_l_greateq_2sb])
                hist_s[sample][tree_name]['2l_1sb_less_pt30'] += len(met[two_l_1sb_less_pt30])
                hist_s[sample][tree_name]['2l_1sb_great_pt30'] += len(met[two_l_1sb_great_pt30])
                hist_s[sample][tree_name]['2l_0sj'] += len(met[two_l_0sj])
                hist_s[sample][tree_name]['2l_1sj'] += len(met[two_l_1sj])
                hist_s[sample][tree_name]['2l_greateq_1sj'] += len(met[two_l_greateq_1sj])
                
                hist_s[sample][tree_name]['1l_less_pt12_0sb'] += len(met[one_less_pt12_l_0sb])
                hist_s[sample][tree_name]['1l_less_pt12_greateq_2sb'] += len(met[one_less_pt12_l_greateq_2sb])
                hist_s[sample][tree_name]['1l_less_pt12_1sb_less_pt30'] += len(met[one_less_pt12_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_less_pt12_1sb_great_pt30'] += len(met[one_less_pt12_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_less_pt12_0sj'] += len(met[one_less_pt12_l_0sj])
                hist_s[sample][tree_name]['1l_less_pt12_1sj'] += len(met[one_less_pt12_l_1sj])
                hist_s[sample][tree_name]['1l_less_pt12_greateq_1sj'] += len(met[one_less_pt12_l_greateq_1sj])
                
                hist_s[sample][tree_name]['1l_pt12to50_0sb'] += len(met[one_pt12to50_l_0sb])
                hist_s[sample][tree_name]['1l_pt12to50_greateq_2sb'] += len(met[one_pt12to50_l_greateq_2sb])
                hist_s[sample][tree_name]['1l_pt12to50_1sb_less_pt30'] += len(met[one_pt12to50_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_pt12to50_1sb_great_pt30'] += len(met[one_pt12to50_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_pt12to50_0sj'] += len(met[one_pt12to50_l_0sj])
                hist_s[sample][tree_name]['1l_pt12to50_1sj'] += len(met[one_pt12to50_l_1sj])
                hist_s[sample][tree_name]['1l_pt12to50_greateq_1sj'] += len(met[one_pt12to50_l_greateq_1sj])
                
                hist_s[sample][tree_name]['1l_great_pt50_0sb'] += len(met[one_great_pt50_l_0sb])
                hist_s[sample][tree_name]['1l_great_pt50_greateq_2sb'] += len(met[one_great_pt50_l_greateq_2sb])
                hist_s[sample][tree_name]['1l_great_pt50_1sb_less_pt30'] += len(met[one_great_pt50_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_great_pt50_1sb_great_pt30'] += len(met[one_great_pt50_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_great_pt50_0sj'] += len(met[one_great_pt50_l_0sj])
                hist_s[sample][tree_name]['1l_great_pt50_1sj'] += len(met[one_great_pt50_l_1sj])
                hist_s[sample][tree_name]['1l_great_pt50_greateq_1sj'] += len(met[one_great_pt50_l_greateq_1sj])
                
                hist_s_w[sample][tree_name]['total_w'] += np.sum(weight) 
                hist_s_w[sample][tree_name]['2l_0sb_w'] += np.sum(weight[two_l_0sb])
                hist_s_w[sample][tree_name]['2l_greateq_2sb_w'] += np.sum(weight[two_l_greateq_2sb])
                hist_s_w[sample][tree_name]['2l_1sb_less_pt30_w'] += np.sum(weight[two_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['2l_1sb_great_pt30_w'] += np.sum(weight[two_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['2l_0sj_w'] += np.sum(weight[two_l_0sj])
                hist_s_w[sample][tree_name]['2l_1sj_w'] += np.sum(weight[two_l_1sj])
                hist_s_w[sample][tree_name]['2l_greateq_1sj_w'] += np.sum(weight[two_l_greateq_1sj])

                hist_s_w[sample][tree_name]['1l_less_pt12_0sb_w'] += np.sum(weight[one_less_pt12_l_0sb])
                hist_s_w[sample][tree_name]['1l_less_pt12_greateq_2sb_w'] += np.sum(weight[one_less_pt12_l_greateq_2sb])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sb_less_pt30_w'] += np.sum(weight[one_less_pt12_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sb_great_pt30_w'] += np.sum(weight[one_less_pt12_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_less_pt12_0sj_w'] += np.sum(weight[one_less_pt12_l_0sj])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sj_w'] += np.sum(weight[one_less_pt12_l_1sj])
                hist_s_w[sample][tree_name]['1l_less_pt12_greateq_1sj_w'] += np.sum(weight[one_less_pt12_l_greateq_1sj])
                
                hist_s_w[sample][tree_name]['1l_pt12to50_0sb_w'] += np.sum(weight[one_pt12to50_l_0sb])
                hist_s_w[sample][tree_name]['1l_pt12to50_greateq_2sb_w'] += np.sum(weight[one_pt12to50_l_greateq_2sb])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sb_less_pt30_w'] += np.sum(weight[one_pt12to50_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sb_great_pt30_w'] += np.sum(weight[one_pt12to50_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_pt12to50_0sj_w'] += np.sum(weight[one_pt12to50_l_0sj])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sj_w'] += np.sum(weight[one_pt12to50_l_1sj])
                hist_s_w[sample][tree_name]['1l_pt12to50_greateq_1sj_w'] += np.sum(weight[one_pt12to50_l_greateq_1sj])
                
                hist_s_w[sample][tree_name]['1l_great_pt50_0sb_w'] += np.sum(weight[one_great_pt50_l_0sb])
                hist_s_w[sample][tree_name]['1l_great_pt50_greateq_2sb_w'] += np.sum(weight[one_great_pt50_l_greateq_2sb])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sb_less_pt30_w'] += np.sum(weight[one_great_pt50_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sb_great_pt30_w'] += np.sum(weight[one_great_pt50_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_great_pt50_0sj_w'] += np.sum(weight[one_great_pt50_l_0sj])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sj_w'] += np.sum(weight[one_great_pt50_l_1sj])
                hist_s_w[sample][tree_name]['1l_great_pt50_greateq_1sj_w'] += np.sum(weight[one_great_pt50_l_greateq_1sj])

                hist_isr[sample][tree_name]['total'] += len(met)
                hist_isr[sample][tree_name]['2l_0isrb'] += len(met[two_l_0isrb])
                hist_isr[sample][tree_name]['2l_greateq_2isrb'] += len(met[two_l_greateq_2isrb])
                hist_isr[sample][tree_name]['2l_1isrb_less_pt30'] += len(met[two_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['2l_1isrb_great_pt30'] += len(met[two_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['2l_0isrj'] += len(met[two_l_0isrj])
                hist_isr[sample][tree_name]['2l_1isrj'] += len(met[two_l_1isrj])
                hist_isr[sample][tree_name]['2l_greateq_1isrj'] += len(met[two_l_greateq_1isrj])
                
                hist_isr[sample][tree_name]['1l_less_pt12_0isrb'] += len(met[one_less_pt12_l_0isrb])
                hist_isr[sample][tree_name]['1l_less_pt12_greateq_2isrb'] += len(met[one_less_pt12_l_greateq_2isrb])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrb_less_pt30'] += len(met[one_less_pt12_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrb_great_pt30'] += len(met[one_less_pt12_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_less_pt12_0isrj'] += len(met[one_less_pt12_l_0isrj])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrj'] += len(met[one_less_pt12_l_1isrj])
                hist_isr[sample][tree_name]['1l_less_pt12_greateq_1isrj'] += len(met[one_less_pt12_l_greateq_1isrj])
                
                hist_isr[sample][tree_name]['1l_pt12to50_0isrb'] += len(met[one_pt12to50_l_0isrb])
                hist_isr[sample][tree_name]['1l_pt12to50_greateq_2isrb'] += len(met[one_pt12to50_l_greateq_2isrb])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrb_less_pt30'] += len(met[one_pt12to50_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrb_great_pt30'] += len(met[one_pt12to50_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_pt12to50_0isrj'] += len(met[one_pt12to50_l_0isrj])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrj'] += len(met[one_pt12to50_l_1isrj])
                hist_isr[sample][tree_name]['1l_pt12to50_greateq_1isrj'] += len(met[one_pt12to50_l_greateq_1isrj])
                
                hist_isr[sample][tree_name]['1l_great_pt50_0isrb'] += len(met[one_great_pt50_l_0isrb])
                hist_isr[sample][tree_name]['1l_great_pt50_greateq_2isrb'] += len(met[one_great_pt50_l_greateq_2isrb])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrb_less_pt30'] += len(met[one_great_pt50_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrb_great_pt30'] += len(met[one_great_pt50_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_great_pt50_0isrj'] += len(met[one_great_pt50_l_0isrj])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrj'] += len(met[one_great_pt50_l_1isrj])
                hist_isr[sample][tree_name]['1l_great_pt50_greateq_1isrj'] += len(met[one_great_pt50_l_greateq_1isrj])
                
                hist_isr_w[sample][tree_name]['total_w'] += np.sum(weight) 
                hist_isr_w[sample][tree_name]['2l_0isrb_w'] += np.sum(weight[two_l_0isrb])
                hist_isr_w[sample][tree_name]['2l_greateq_2isrb_w'] += np.sum(weight[two_l_greateq_2isrb])
                hist_isr_w[sample][tree_name]['2l_1isrb_less_pt30_w'] += np.sum(weight[two_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['2l_1isrb_great_pt30_w'] += np.sum(weight[two_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['2l_0isrj_w'] += np.sum(weight[two_l_0isrj])
                hist_isr_w[sample][tree_name]['2l_1isrj_w'] += np.sum(weight[two_l_1isrj])
                hist_isr_w[sample][tree_name]['2l_greateq_1isrj_w'] += np.sum(weight[two_l_greateq_1isrj])

                hist_isr_w[sample][tree_name]['1l_less_pt12_0isrb_w'] += np.sum(weight[one_less_pt12_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_less_pt12_greateq_2isrb_w'] += np.sum(weight[one_less_pt12_l_greateq_2isrb])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_less_pt30_w'] += np.sum(weight[one_less_pt12_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_great_pt30_w'] += np.sum(weight[one_less_pt12_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_less_pt12_0isrj_w'] += np.sum(weight[one_less_pt12_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrj_w'] += np.sum(weight[one_less_pt12_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_less_pt12_greateq_1isrj_w'] += np.sum(weight[one_less_pt12_l_greateq_1isrj])
                
                hist_isr_w[sample][tree_name]['1l_pt12to50_0isrb_w'] += np.sum(weight[one_pt12to50_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_pt12to50_greateq_2isrb_w'] += np.sum(weight[one_pt12to50_l_greateq_2isrb])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_less_pt30_w'] += np.sum(weight[one_pt12to50_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_great_pt30_w'] += np.sum(weight[one_pt12to50_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_pt12to50_0isrj_w'] += np.sum(weight[one_pt12to50_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrj_w'] += np.sum(weight[one_pt12to50_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_pt12to50_greateq_1isrj_w'] += np.sum(weight[one_pt12to50_l_greateq_1isrj])
                
                hist_isr_w[sample][tree_name]['1l_great_pt50_0isrb_w'] += np.sum(weight[one_great_pt50_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_great_pt50_greateq_2isrb_w'] += np.sum(weight[one_great_pt50_l_greateq_2isrb])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_less_pt30_w'] += np.sum(weight[one_great_pt50_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_great_pt30_w'] += np.sum(weight[one_great_pt50_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_great_pt50_0isrj_w'] += np.sum(weight[one_great_pt50_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrj_w'] += np.sum(weight[one_great_pt50_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_great_pt50_greateq_1isrj_w'] += np.sum(weight[one_great_pt50_l_greateq_1isrj])

                print 'finished filling'
    return hist_s, hist_s_w, hist_isr, hist_isr_w


             

if __name__ == "__main__":

    samples = { 
    'SMS-T2-4bd_420' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_SMS_T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_/',
],
    'SMS-T2-4bd_490' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_SMS_T2-4bd/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_/',
],
    'TTJets_2017' : [
                     '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
#    'ST_2017' : [
#              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8_Fall17_94X',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8_Fall17_94X',
#              ],
    'WJets_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
#'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
],
    'ZJets_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-100To200_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-200To400_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-400To600_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-600To800_13TeV-madgraph_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZJetsToNuNu_HT-800To1200_13TeV-madgraph_Fall17_94X/',
],
    'DY_M50_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X/',
#'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_Fall17_94X/',
],
    'WW_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWTo2L2Nu_NNPDF31_TuneCP5_PSweights_13TeV-powheg-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWTo4Q_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWToLNuQQ_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X/',
],
    'ZZ_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZZTo2L2Nu_13TeV_powheg_pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZZTo2Q2Nu_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZZTo4L_13TeV_powheg_pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X/',
],
    'WZ_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_v2_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/WZ_TuneCP5_13TeV-pythia8_Fall17_94X/',
],
    'QCD_2017' : [
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT100to200_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO_test/Fall17_94X_mybkg/QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8_Fall17_94X/',
],

                  }
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'PTISR', 'MiniIso_lep', 'weight']

    start_b = time.time()    
    sample_list = process_the_samples(samples, None, None)
    sample_arrays, sample_w_arrays, isr_arrays, isr_w_arrays = make_tables(sample_list, variables, None)

    write_table(sample_arrays, sample_w_arrays, './output_table_nano_samples_tightleps_regions_s_jets_low_pt_30Jul19.txt')  
    write_table(isr_arrays, isr_w_arrays, './output_table_nano_samples_tightleps_regions_isr_jets_low_pt_30Jul19.txt')  
    stop_b = time.time()

    print "total: ", stop_b - start_b
 
    print 'finished writing'
