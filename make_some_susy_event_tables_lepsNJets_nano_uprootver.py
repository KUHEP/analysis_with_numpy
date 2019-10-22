#!/usr/bin/env python

"""
New thing to try out, doing analysis with numpy, converting back to ROOT to do histogramming
Creator: Erich Schmitz
Date: Feb 22, 2019
"""

import ROOT as rt
import numpy as np
import uproot as ur
import awkward as aw
#import root_numpy as rnp
import numpy.lib.recfunctions as rfc
import os
from file_table_functions import *
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
            print('\nReserving Tables for:', sample, tree_name)
            hist_s[sample][tree_name] = OrderedDict()
            hist_isr[sample][tree_name] = OrderedDict()
            hist_s_w[sample][tree_name] = OrderedDict()
            hist_isr_w[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve tables

            hist_s[sample][tree_name]['total'] = 0.
            hist_s[sample][tree_name]['2l'] = 0.
            hist_s[sample][tree_name]['2l_0sb'] = 0.
            hist_s[sample][tree_name]['2l_ge_2sb'] = 0.
            hist_s[sample][tree_name]['2l_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['2l_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['2l_0sj'] = 0.
            hist_s[sample][tree_name]['2l_1sj'] = 0.
            hist_s[sample][tree_name]['2l_ge_1sj'] = 0.
            hist_s[sample][tree_name]['1l'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_0sb'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_ge_2sb'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_0sj'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_1sj'] = 0.
            hist_s[sample][tree_name]['1l_less_pt12_ge_1sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_0sb'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_ge_2sb'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_0sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_1sj'] = 0.
            hist_s[sample][tree_name]['1l_pt12to50_ge_1sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_0sb'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_ge_2sb'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sb_less_pt30'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sb_great_pt30'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_0sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_1sj'] = 0.
            hist_s[sample][tree_name]['1l_great_pt50_ge_1sj'] = 0.

            hist_isr[sample][tree_name]['total'] = 0.
            hist_isr[sample][tree_name]['2l_0isrb'] = 0.
            hist_isr[sample][tree_name]['2l_ge_2isrb'] = 0.
            hist_isr[sample][tree_name]['2l_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['2l_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['2l_0isrj'] = 0.
            hist_isr[sample][tree_name]['2l_1isrj'] = 0.
            hist_isr[sample][tree_name]['2l_ge_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_ge_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_less_pt12_ge_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_ge_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_pt12to50_ge_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_0isrb'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_ge_2isrb'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrb_less_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrb_great_pt30'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_0isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_1isrj'] = 0.
            hist_isr[sample][tree_name]['1l_great_pt50_ge_1isrj'] = 0.


            hist_s_w[sample][tree_name]['total_w'] = 0.
            hist_s_w[sample][tree_name]['2l_w'] = 0.
            hist_s_w[sample][tree_name]['2l_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['2l_ge_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['2l_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['2l_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['2l_ge_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_ge_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_less_pt12_ge_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_ge_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_pt12to50_ge_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_0sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_ge_2sb_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sb_less_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sb_great_pt30_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_0sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_1sj_w'] = 0.
            hist_s_w[sample][tree_name]['1l_great_pt50_ge_1sj_w'] = 0.

            hist_isr_w[sample][tree_name]['total_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_ge_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['2l_ge_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_ge_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_less_pt12_ge_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_ge_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_pt12to50_ge_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_0isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_ge_2isrb_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_less_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_great_pt30_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_0isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_1isrj_w'] = 0.
            hist_isr_w[sample][tree_name]['1l_great_pt50_ge_1isrj_w'] = 0.

        i_entries = 0    
        for itree, in_tree in enumerate(list_of_files_[sample]['trees']):
            for events in ur.tree.iterate(list_of_files_[sample]['files'], in_tree, branches=variable_list_, entrysteps=np.inf):
                print('\nGetting Histograms for:', sample, tree_name)
                print('tree: ', itree+1)
                #i_entries += 10000
                #print(i_entries)

                print('converting objects to jaggeds')
                for b in events:
                    if 'ObjectArray' in str(type(events[b])):
                        events[b] = aw.fromiter(events[b])
                met = events[b'MET']
                weight = events[b'weight']
                weight = 137. * weight

                if 'SMS-T2-4bd_490' in sample:
                    weight = np.array([(137000.*0.51848) / 1207007. for w in weight])

                risr = events[b'RISR']
                ptisr = events[b'PTISR']

                pt_jet = events[b'PT_jet']
                isr_index_jet = events[b'index_jet_ISR']
                s_index_jet = events[b'index_jet_S']
                btag_jet = events[b'Btag_jet']

                pt_lep = events[b'PT_lep']
                mini_lep = events[b'MiniIso_lep']
                id_lep = events[b'ID_lep']

                risr = risr[:, 1]
                ptisr = ptisr[:, 1]
                #ptcm = ptcm[:, 1]
                isr_index_jet = isr_index_jet[:, 1]
                s_index_jet = s_index_jet[:, 1]
                #isr_index_lep = isr_index_lep[:, 1]
                #s_index_lep = s_index_lep[:, 1]
                #dphi = dphi[:, 1]

                print('makeing variables')
                len_jet = pt_jet.stops - pt_jet.starts
                max_n_jets = np.amax(len_jet)

                pt_s_jet = pt_jet[s_index_jet]
                pt_isr_jet = pt_jet[isr_index_jet]
                btag_s_jet = btag_jet[s_index_jet]
                btag_isr_jet = btag_jet[isr_index_jet]

                ################  Choosing Lepton ID #####################
                ################        Medium       #####################
                #pt_lep = pt_lep[id_lep>=3]
                #mini_lep = mini_lep[id_lep>=3]
                #pt_lep = np.array([pt[lid>=3] for pt, lid in zip(pt_lep, id_lep)])
                #mini_lep = np.array([mini[lid>=3] for mini, lid in zip(mini_lep, id_lep)])
                ################        Tight        #####################
                #pt_lep = np.array([pt[lid>=4] for pt, lid in zip(pt_lep, id_lep)])
                #mini_lep = np.array([mini[lid>=4] for mini, lid in zip(mini_lep, id_lep)])
                ##########################################################
 
                len_lep = pt_lep.stops - pt_lep.starts
                max_n_leps = np.amax(len_lep)

                pt_mini_lep = pt_lep[mini_lep < 0.1]

                pt_less_12_lep = pt_mini_lep[pt_mini_lep<12]
                pt_12to50_lep = pt_mini_lep[np.logical_and(pt_mini_lep>12, pt_mini_lep<=50)]
                pt_great_50_lep = pt_mini_lep[pt_mini_lep>50]
                
                print('\ncreating masks and weights')
                print('-> bjet masks')
                medium_s_jet = pt_s_jet[btag_s_jet > 0.8484]
                medium_isr_jet = pt_isr_jet[btag_isr_jet > 0.8484]
              

                print('making masks')
                print('-> s jets')
                has_0_s_jets = (s_index_jet.stops - s_index_jet.starts) < 1
                has_1_s_jets = (s_index_jet.stops - s_index_jet.starts) == 1
                has_ge_1_s_jets = (s_index_jet.stops - s_index_jet.starts) >= 1
 
                print('-> isr jets')
                has_0_isr_jets = (isr_index_jet.stops - isr_index_jet.starts) < 1
                has_1_isr_jets = (isr_index_jet.stops - isr_index_jet.starts) == 1
                has_ge_1_isr_jets = (isr_index_jet.stops - isr_index_jet.starts) >= 1
                 
                print('-> b jets')
                has_2_medium = (medium_s_jet.stops - medium_s_jet.starts) >= 2
                has_2_isr_medium = (medium_isr_jet.stops - medium_isr_jet.starts) >= 2

                has_1_less_pt30_medium = (medium_s_jet[medium_s_jet < 30].stops - medium_s_jet[medium_s_jet < 30].starts) == 1
                has_1_great_pt30_medium = (medium_s_jet[medium_s_jet > 30].stops - medium_s_jet[medium_s_jet > 30].starts) == 1

                has_1_less_pt30_isr_medium = (medium_isr_jet[medium_isr_jet < 30].stops - medium_isr_jet[medium_isr_jet < 30].starts) == 1
                has_1_great_pt30_isr_medium = (medium_isr_jet[medium_isr_jet > 30].stops - medium_isr_jet[medium_isr_jet > 30].starts) == 1

                has_no_medium = (medium_s_jet.stops - medium_s_jet.starts) == 0
                has_no_isr_medium = (medium_isr_jet.stops - medium_isr_jet.starts) == 0

                print('-> leptons')
                only_2_lep = (pt_mini_lep.stops - pt_mini_lep.starts) == 2
                only_1_lep = np.logical_and((pt_mini_lep.stops - pt_mini_lep.starts) == 1, only_2_lep == False)

                only_1_less_pt12_lep = np.logical_and((pt_less_12_lep.stops - pt_less_12_lep.starts) == 1, only_2_lep == False)
                only_1_pt12to50_lep = np.logical_and((pt_12to50_lep.stops - pt_less_12_lep.starts) == 1, only_2_lep == False)
                only_1_great_pt50_lep = np.logical_and((pt_great_50_lep.stops - pt_great_50_lep.starts) == 1, only_2_lep == False)

                print('-> events variables')
                met_200 = met > 200

                risr_0p8 = risr > 0.95

                ptisr_200 = ptisr > 200


                print('incrementing tables')
                #two_l = np.all([met_200, risr_0p8, ptisr_200, only_2_lep], axis=0)
                two_l = np.all([only_2_lep], axis=0)
                two_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_no_medium], axis=0)
                two_l_ge_2sb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_2_medium], axis=0)
                two_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_less_pt30_medium], axis=0)
                two_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_great_pt30_medium], axis=0)
                two_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_0_s_jets], axis=0)
                two_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_s_jets], axis=0)
                two_l_ge_1sj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_ge_1_s_jets], axis=0)
                one_l = np.all([met_200, risr_0p8, ptisr_200, only_1_lep], axis=0)
                one_less_pt12_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_no_medium], axis=0)
                one_less_pt12_l_ge_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_2_medium], axis=0)
                one_less_pt12_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_less_pt30_medium], axis=0)
                one_less_pt12_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_great_pt30_medium], axis=0)
                one_less_pt12_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_0_s_jets], axis=0)
                one_less_pt12_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_s_jets], axis=0)
                one_less_pt12_l_ge_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_ge_1_s_jets], axis=0)

                one_pt12to50_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_no_medium], axis=0)
                one_pt12to50_l_ge_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_2_medium], axis=0)
                one_pt12to50_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_less_pt30_medium], axis=0)
                one_pt12to50_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_great_pt30_medium], axis=0)
                one_pt12to50_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_0_s_jets], axis=0)
                one_pt12to50_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_s_jets], axis=0)
                one_pt12to50_l_ge_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_ge_1_s_jets], axis=0)

                one_great_pt50_l_0sb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_no_medium], axis=0)
                one_great_pt50_l_ge_2sb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_2_medium], axis=0)
                one_great_pt50_l_1sb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_less_pt30_medium], axis=0)
                one_great_pt50_l_1sb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_great_pt30_medium], axis=0)
                one_great_pt50_l_0sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_0_s_jets], axis=0)
                one_great_pt50_l_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_s_jets], axis=0)
                one_great_pt50_l_ge_1sj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_ge_1_s_jets], axis=0)

                two_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_no_isr_medium], axis=0)
                two_l_ge_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_2_isr_medium], axis=0)
                two_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_less_pt30_isr_medium], axis=0)
                two_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_great_pt30_isr_medium], axis=0)
                two_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_0_isr_jets], axis=0)
                two_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_1_isr_jets], axis=0)
                two_l_ge_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_2_lep, has_ge_1_isr_jets], axis=0)

                one_less_pt12_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_no_isr_medium], axis=0)
                one_less_pt12_l_ge_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_2_isr_medium], axis=0)
                one_less_pt12_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_less_pt30_isr_medium], axis=0)
                one_less_pt12_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_great_pt30_isr_medium], axis=0)
                one_less_pt12_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_0_isr_jets], axis=0)
                one_less_pt12_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_1_isr_jets], axis=0)
                one_less_pt12_l_ge_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_less_pt12_lep, has_ge_1_isr_jets], axis=0)

                one_pt12to50_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_no_isr_medium], axis=0)
                one_pt12to50_l_ge_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_2_isr_medium], axis=0)
                one_pt12to50_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_less_pt30_isr_medium], axis=0)
                one_pt12to50_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_great_pt30_isr_medium], axis=0)
                one_pt12to50_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_0_isr_jets], axis=0)
                one_pt12to50_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_1_isr_jets], axis=0)
                one_pt12to50_l_ge_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_pt12to50_lep, has_ge_1_isr_jets], axis=0)

                one_great_pt50_l_0isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_no_isr_medium], axis=0)
                one_great_pt50_l_ge_2isrb = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_2_isr_medium], axis=0)
                one_great_pt50_l_1isrb_less_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_less_pt30_isr_medium], axis=0)
                one_great_pt50_l_1isrb_great_pt30 = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_great_pt30_isr_medium], axis=0)
                one_great_pt50_l_0isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_0_isr_jets], axis=0)
                one_great_pt50_l_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_1_isr_jets], axis=0)
                one_great_pt50_l_ge_1isrj = np.all([met_200, risr_0p8, ptisr_200, only_1_great_pt50_lep, has_ge_1_isr_jets], axis=0)


                hist_s[sample][tree_name]['total'] += len(met)
                hist_s[sample][tree_name]['2l'] += len(met[two_l])
                hist_s[sample][tree_name]['2l_0sb'] += len(met[two_l_0sb])
                hist_s[sample][tree_name]['2l_ge_2sb'] += len(met[two_l_ge_2sb])
                hist_s[sample][tree_name]['2l_1sb_less_pt30'] += len(met[two_l_1sb_less_pt30])
                hist_s[sample][tree_name]['2l_1sb_great_pt30'] += len(met[two_l_1sb_great_pt30])
                hist_s[sample][tree_name]['2l_0sj'] += len(met[two_l_0sj])
                hist_s[sample][tree_name]['2l_1sj'] += len(met[two_l_1sj])
                hist_s[sample][tree_name]['2l_ge_1sj'] += len(met[two_l_ge_1sj])
                
                hist_s[sample][tree_name]['1l'] += len(met[one_l])
                hist_s[sample][tree_name]['1l_less_pt12_0sb'] += len(met[one_less_pt12_l_0sb])
                hist_s[sample][tree_name]['1l_less_pt12_ge_2sb'] += len(met[one_less_pt12_l_ge_2sb])
                hist_s[sample][tree_name]['1l_less_pt12_1sb_less_pt30'] += len(met[one_less_pt12_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_less_pt12_1sb_great_pt30'] += len(met[one_less_pt12_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_less_pt12_0sj'] += len(met[one_less_pt12_l_0sj])
                hist_s[sample][tree_name]['1l_less_pt12_1sj'] += len(met[one_less_pt12_l_1sj])
                hist_s[sample][tree_name]['1l_less_pt12_ge_1sj'] += len(met[one_less_pt12_l_ge_1sj])
                
                hist_s[sample][tree_name]['1l_pt12to50_0sb'] += len(met[one_pt12to50_l_0sb])
                hist_s[sample][tree_name]['1l_pt12to50_ge_2sb'] += len(met[one_pt12to50_l_ge_2sb])
                hist_s[sample][tree_name]['1l_pt12to50_1sb_less_pt30'] += len(met[one_pt12to50_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_pt12to50_1sb_great_pt30'] += len(met[one_pt12to50_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_pt12to50_0sj'] += len(met[one_pt12to50_l_0sj])
                hist_s[sample][tree_name]['1l_pt12to50_1sj'] += len(met[one_pt12to50_l_1sj])
                hist_s[sample][tree_name]['1l_pt12to50_ge_1sj'] += len(met[one_pt12to50_l_ge_1sj])
                
                hist_s[sample][tree_name]['1l_great_pt50_0sb'] += len(met[one_great_pt50_l_0sb])
                hist_s[sample][tree_name]['1l_great_pt50_ge_2sb'] += len(met[one_great_pt50_l_ge_2sb])
                hist_s[sample][tree_name]['1l_great_pt50_1sb_less_pt30'] += len(met[one_great_pt50_l_1sb_less_pt30])
                hist_s[sample][tree_name]['1l_great_pt50_1sb_great_pt30'] += len(met[one_great_pt50_l_1sb_great_pt30])
                hist_s[sample][tree_name]['1l_great_pt50_0sj'] += len(met[one_great_pt50_l_0sj])
                hist_s[sample][tree_name]['1l_great_pt50_1sj'] += len(met[one_great_pt50_l_1sj])
                hist_s[sample][tree_name]['1l_great_pt50_ge_1sj'] += len(met[one_great_pt50_l_ge_1sj])
                
                hist_s_w[sample][tree_name]['total_w'] += np.sum(weight) 
                hist_s_w[sample][tree_name]['2l_w'] += np.sum(weight[two_l])
                hist_s_w[sample][tree_name]['2l_0sb_w'] += np.sum(weight[two_l_0sb])
                hist_s_w[sample][tree_name]['2l_ge_2sb_w'] += np.sum(weight[two_l_ge_2sb])
                hist_s_w[sample][tree_name]['2l_1sb_less_pt30_w'] += np.sum(weight[two_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['2l_1sb_great_pt30_w'] += np.sum(weight[two_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['2l_0sj_w'] += np.sum(weight[two_l_0sj])
                hist_s_w[sample][tree_name]['2l_1sj_w'] += np.sum(weight[two_l_1sj])
                hist_s_w[sample][tree_name]['2l_ge_1sj_w'] += np.sum(weight[two_l_ge_1sj])

                hist_s_w[sample][tree_name]['1l_w'] += np.sum(weight[one_l])
                hist_s_w[sample][tree_name]['1l_less_pt12_0sb_w'] += np.sum(weight[one_less_pt12_l_0sb])
                hist_s_w[sample][tree_name]['1l_less_pt12_ge_2sb_w'] += np.sum(weight[one_less_pt12_l_ge_2sb])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sb_less_pt30_w'] += np.sum(weight[one_less_pt12_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sb_great_pt30_w'] += np.sum(weight[one_less_pt12_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_less_pt12_0sj_w'] += np.sum(weight[one_less_pt12_l_0sj])
                hist_s_w[sample][tree_name]['1l_less_pt12_1sj_w'] += np.sum(weight[one_less_pt12_l_1sj])
                hist_s_w[sample][tree_name]['1l_less_pt12_ge_1sj_w'] += np.sum(weight[one_less_pt12_l_ge_1sj])
                
                hist_s_w[sample][tree_name]['1l_pt12to50_0sb_w'] += np.sum(weight[one_pt12to50_l_0sb])
                hist_s_w[sample][tree_name]['1l_pt12to50_ge_2sb_w'] += np.sum(weight[one_pt12to50_l_ge_2sb])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sb_less_pt30_w'] += np.sum(weight[one_pt12to50_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sb_great_pt30_w'] += np.sum(weight[one_pt12to50_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_pt12to50_0sj_w'] += np.sum(weight[one_pt12to50_l_0sj])
                hist_s_w[sample][tree_name]['1l_pt12to50_1sj_w'] += np.sum(weight[one_pt12to50_l_1sj])
                hist_s_w[sample][tree_name]['1l_pt12to50_ge_1sj_w'] += np.sum(weight[one_pt12to50_l_ge_1sj])
                
                hist_s_w[sample][tree_name]['1l_great_pt50_0sb_w'] += np.sum(weight[one_great_pt50_l_0sb])
                hist_s_w[sample][tree_name]['1l_great_pt50_ge_2sb_w'] += np.sum(weight[one_great_pt50_l_ge_2sb])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sb_less_pt30_w'] += np.sum(weight[one_great_pt50_l_1sb_less_pt30])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sb_great_pt30_w'] += np.sum(weight[one_great_pt50_l_1sb_great_pt30])
                hist_s_w[sample][tree_name]['1l_great_pt50_0sj_w'] += np.sum(weight[one_great_pt50_l_0sj])
                hist_s_w[sample][tree_name]['1l_great_pt50_1sj_w'] += np.sum(weight[one_great_pt50_l_1sj])
                hist_s_w[sample][tree_name]['1l_great_pt50_ge_1sj_w'] += np.sum(weight[one_great_pt50_l_ge_1sj])

                hist_isr[sample][tree_name]['total'] += len(met)
                hist_isr[sample][tree_name]['2l_0isrb'] += len(met[two_l_0isrb])
                hist_isr[sample][tree_name]['2l_ge_2isrb'] += len(met[two_l_ge_2isrb])
                hist_isr[sample][tree_name]['2l_1isrb_less_pt30'] += len(met[two_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['2l_1isrb_great_pt30'] += len(met[two_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['2l_0isrj'] += len(met[two_l_0isrj])
                hist_isr[sample][tree_name]['2l_1isrj'] += len(met[two_l_1isrj])
                hist_isr[sample][tree_name]['2l_ge_1isrj'] += len(met[two_l_ge_1isrj])
                
                hist_isr[sample][tree_name]['1l_less_pt12_0isrb'] += len(met[one_less_pt12_l_0isrb])
                hist_isr[sample][tree_name]['1l_less_pt12_ge_2isrb'] += len(met[one_less_pt12_l_ge_2isrb])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrb_less_pt30'] += len(met[one_less_pt12_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrb_great_pt30'] += len(met[one_less_pt12_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_less_pt12_0isrj'] += len(met[one_less_pt12_l_0isrj])
                hist_isr[sample][tree_name]['1l_less_pt12_1isrj'] += len(met[one_less_pt12_l_1isrj])
                hist_isr[sample][tree_name]['1l_less_pt12_ge_1isrj'] += len(met[one_less_pt12_l_ge_1isrj])
                
                hist_isr[sample][tree_name]['1l_pt12to50_0isrb'] += len(met[one_pt12to50_l_0isrb])
                hist_isr[sample][tree_name]['1l_pt12to50_ge_2isrb'] += len(met[one_pt12to50_l_ge_2isrb])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrb_less_pt30'] += len(met[one_pt12to50_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrb_great_pt30'] += len(met[one_pt12to50_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_pt12to50_0isrj'] += len(met[one_pt12to50_l_0isrj])
                hist_isr[sample][tree_name]['1l_pt12to50_1isrj'] += len(met[one_pt12to50_l_1isrj])
                hist_isr[sample][tree_name]['1l_pt12to50_ge_1isrj'] += len(met[one_pt12to50_l_ge_1isrj])
                
                hist_isr[sample][tree_name]['1l_great_pt50_0isrb'] += len(met[one_great_pt50_l_0isrb])
                hist_isr[sample][tree_name]['1l_great_pt50_ge_2isrb'] += len(met[one_great_pt50_l_ge_2isrb])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrb_less_pt30'] += len(met[one_great_pt50_l_1isrb_less_pt30])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrb_great_pt30'] += len(met[one_great_pt50_l_1isrb_great_pt30])
                hist_isr[sample][tree_name]['1l_great_pt50_0isrj'] += len(met[one_great_pt50_l_0isrj])
                hist_isr[sample][tree_name]['1l_great_pt50_1isrj'] += len(met[one_great_pt50_l_1isrj])
                hist_isr[sample][tree_name]['1l_great_pt50_ge_1isrj'] += len(met[one_great_pt50_l_ge_1isrj])
                
                hist_isr_w[sample][tree_name]['total_w'] += np.sum(weight) 
                hist_isr_w[sample][tree_name]['2l_0isrb_w'] += np.sum(weight[two_l_0isrb])
                hist_isr_w[sample][tree_name]['2l_ge_2isrb_w'] += np.sum(weight[two_l_ge_2isrb])
                hist_isr_w[sample][tree_name]['2l_1isrb_less_pt30_w'] += np.sum(weight[two_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['2l_1isrb_great_pt30_w'] += np.sum(weight[two_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['2l_0isrj_w'] += np.sum(weight[two_l_0isrj])
                hist_isr_w[sample][tree_name]['2l_1isrj_w'] += np.sum(weight[two_l_1isrj])
                hist_isr_w[sample][tree_name]['2l_ge_1isrj_w'] += np.sum(weight[two_l_ge_1isrj])

                hist_isr_w[sample][tree_name]['1l_less_pt12_0isrb_w'] += np.sum(weight[one_less_pt12_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_less_pt12_ge_2isrb_w'] += np.sum(weight[one_less_pt12_l_ge_2isrb])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_less_pt30_w'] += np.sum(weight[one_less_pt12_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrb_great_pt30_w'] += np.sum(weight[one_less_pt12_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_less_pt12_0isrj_w'] += np.sum(weight[one_less_pt12_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_less_pt12_1isrj_w'] += np.sum(weight[one_less_pt12_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_less_pt12_ge_1isrj_w'] += np.sum(weight[one_less_pt12_l_ge_1isrj])
                
                hist_isr_w[sample][tree_name]['1l_pt12to50_0isrb_w'] += np.sum(weight[one_pt12to50_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_pt12to50_ge_2isrb_w'] += np.sum(weight[one_pt12to50_l_ge_2isrb])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_less_pt30_w'] += np.sum(weight[one_pt12to50_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrb_great_pt30_w'] += np.sum(weight[one_pt12to50_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_pt12to50_0isrj_w'] += np.sum(weight[one_pt12to50_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_pt12to50_1isrj_w'] += np.sum(weight[one_pt12to50_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_pt12to50_ge_1isrj_w'] += np.sum(weight[one_pt12to50_l_ge_1isrj])
                
                hist_isr_w[sample][tree_name]['1l_great_pt50_0isrb_w'] += np.sum(weight[one_great_pt50_l_0isrb])
                hist_isr_w[sample][tree_name]['1l_great_pt50_ge_2isrb_w'] += np.sum(weight[one_great_pt50_l_ge_2isrb])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_less_pt30_w'] += np.sum(weight[one_great_pt50_l_1isrb_less_pt30])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrb_great_pt30_w'] += np.sum(weight[one_great_pt50_l_1isrb_great_pt30])
                hist_isr_w[sample][tree_name]['1l_great_pt50_0isrj_w'] += np.sum(weight[one_great_pt50_l_0isrj])
                hist_isr_w[sample][tree_name]['1l_great_pt50_1isrj_w'] += np.sum(weight[one_great_pt50_l_1isrj])
                hist_isr_w[sample][tree_name]['1l_great_pt50_ge_1isrj_w'] += np.sum(weight[one_great_pt50_l_ge_1isrj])

                print('finished filling')
    return hist_s, hist_s_w, hist_isr, hist_isr_w


             

if __name__ == "__main__":

    samples = { 
    'SMS-T2-4bd_420' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_',
],
    'SMS-T2-4bd_490' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_',
                    #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_v2',
],
    'TTJets_2017' : [
                     '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X'
],
#    'ST_2017' : [
              #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8_Fall17_94X',
#              '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8_Fall17_94X'
#              ],
    'WJets_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
#    'DY_M50_2017' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#],
    'WW_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WWTo4Q_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/WWToLNuQQ_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
],
    'ZZ_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ZZTo2L2Nu_13TeV_powheg_pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ZZTo2Q2Nu_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/ZZTo4L_13TeV_powheg_pythia8_Fall17_94X',
]


                  }
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'PTISR', 'MiniIso_lep', 'weight']

    start_b = time.time()    
    sample_list = process_the_samples(samples, None, None)
    sample_arrays, sample_w_arrays, isr_arrays, isr_w_arrays = make_tables(sample_list, variables, None)

    write_table(sample_arrays, sample_w_arrays, './output_table_nano_samples_looseleps_regions_s_jets_test.txt')  
    write_table(isr_arrays, isr_w_arrays, './output_table_nano_samples_looseleps_regions_isr_jets_test.txt')  
    stop_b = time.time()

    print("total: ", stop_b - start_b)
 
    print('finished writing')
