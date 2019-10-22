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


def get_histograms(list_of_files_, variable_list_, cuts_to_apply_=None):
    
    hist = OrderedDict()
    counts = OrderedDict()
    for sample in list_of_files_:
        hist[sample] = OrderedDict()
        counts[sample] = OrderedDict()
        for tree_name in list_of_files_[sample]['trees']:
            print('\nReserving Histograms for:', sample, tree_name)
            hist[sample][tree_name] = OrderedDict()
            counts[sample][tree_name] = OrderedDict()
            # Reserve histograms
            hist[sample][tree_name]['MET'] = rt.TH1D('MET_'+sample+'_'+tree_name, 'E_{T}^{miss} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['S_Flavor_jet'] = rt.TH1D('S_Flavor_jet_'+sample+'_'+tree_name, 'Flavor S jets', 20, 0, 20)
            hist[sample][tree_name]['ISR_Flavor_jet'] = rt.TH1D('ISR_Flavor_jet_'+sample+'_'+tree_name, 'Flavor ISR jets', 20, 0, 20)
            hist[sample][tree_name]['S_Flavor_lep'] = rt.TH1D('S_Flavor_lep_'+sample+'_'+tree_name, 'Flavor S leps', 20, 0, 20)
            hist[sample][tree_name]['ISR_Flavor_lep'] = rt.TH1D('ISR_Flavor_lep_'+sample+'_'+tree_name, 'Flavor ISR leps', 20, 0, 20)

            hist[sample][tree_name]['Lep_to_Charge'] = rt.TH2D('Lep_to_Charge_'+sample+'_'+tree_name, 'lep Flavor to Charge', 20, 0, 20, 5, -2, 2)
            hist[sample][tree_name]['Lep_to_Lep'] = rt.TH2D('Lep_to_Lep_'+sample+'_'+tree_name, '2leps to 2 opp leps', 2, 0, 2, 2, 0, 2)

            hist[sample][tree_name]['RISR'] = rt.TH1D('risr_'+sample+'_'+tree_name, 'RISR', 500, 0, 2)
            hist[sample][tree_name]['PTISR'] = rt.TH1D('ptisr_'+sample+'_'+tree_name, 'p_{T} ISR [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PTCM'] = rt.TH1D('ptcm_'+sample+'_'+tree_name, 'p_{T} CM [GeV]', 500, 0, 1000)

            hist[sample][tree_name]['RISR_PTISR'] = rt.TH2D('RISR_PTISR_'+sample+'_'+tree_name, 'RISR_PTISR', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['RISR_PTCM'] = rt.TH2D('RISR_PTCM_'+sample+'_'+tree_name, 'RISR_PTCM', 500, 0, 2, 500, 0, 1000)

            hist[sample][tree_name]['PTCM_div_PTISR'] = rt.TH1D('PTCM_div_PTISR_'+sample+'_'+tree_name, 'PTCM_div_PTISR', 500, 0, 1)
            hist[sample][tree_name]['dphi_PTCM_div_PTISR'] = rt.TH2D('dphi_PTCM_div_PTISR_'+sample+'_'+tree_name, 'dphi_PTCM_div_PTISR', 500, 0, np.pi, 500, 0, 1)
            hist[sample][tree_name]['dphi_PTCM'] = rt.TH2D('dphi_PTCM_'+sample+'_'+tree_name, 'dphi_PTCM', 500, 0, np.pi, 500, 0, 1000)

            hist[sample][tree_name]['PTISR_PTCM'] = rt.TH2D('PTISR_PTCM_'+sample+'_'+tree_name, 'PTISR_PTCM', 500, 0, 1000, 500, 0, 1000)

            hist[sample][tree_name]['S_ISR_N_jet'] = rt.TH2D('S_ISR_N_jet_'+sample+'_'+tree_name, 'N jet, S-ISR', 15, 0, 15, 15, 0, 15)
            hist[sample][tree_name]['S_ISR_N_lep'] = rt.TH2D('S_ISR_N_lep_'+sample+'_'+tree_name, 'N lep, S-ISR', 15, 0, 15, 15, 0, 15)

            hist[sample][tree_name]['S_ISR_N_loose_jet'] = rt.TH2D('S_ISR_N_loose_jet_'+sample+'_'+tree_name, 'N loose S-ISR', 15, 0, 15, 15, 0, 15)
            hist[sample][tree_name]['S_ISR_N_medium_jet'] = rt.TH2D('S_ISR_N_medium_jet_'+sample+'_'+tree_name, 'N medium S-ISR', 15, 0, 15, 15, 0, 15)
            hist[sample][tree_name]['S_ISR_N_tight_jet'] = rt.TH2D('S_ISR_N_tight_jet_'+sample+'_'+tree_name, 'N tight S-ISR', 15, 0, 15, 15, 0, 15)

            hist[sample][tree_name]['RISR_N_jet'] = rt.TH2D('RISR_N_jet_'+sample+'_'+tree_name, 'RISR N jet', 500, 0, 2, 20, 0, 20)
            hist[sample][tree_name]['RISR_N_lep'] = rt.TH2D('RISR_N_lep_'+sample+'_'+tree_name, 'RISR N lep', 500, 0, 2, 20, 0, 20)

            hist[sample][tree_name]['RISR_N_S_jet'] = rt.TH2D('RISR_N_S_jet_'+sample+'_'+tree_name, 'RISR N S jet', 500, 0, 2, 20, 0, 20)
            hist[sample][tree_name]['RISR_N_S_lep'] = rt.TH2D('RISR_N_S_lep_'+sample+'_'+tree_name, 'RISR N S lep', 500, 0, 2, 20, 0, 20)

            hist[sample][tree_name]['RISR_N_ISR_jet'] = rt.TH2D('RISR_N_ISR_jet_'+sample+'_'+tree_name, 'RISR N ISR jet', 500, 0, 2, 20, 0, 20)
            hist[sample][tree_name]['RISR_N_ISR_lep'] = rt.TH2D('RISR_N_ISR_lep_'+sample+'_'+tree_name, 'RISR N ISR lep', 500, 0, 2, 20, 0, 20)

            hist[sample][tree_name]['PTISR_N_jet'] = rt.TH2D('PTISR_N_jet_'+sample+'_'+tree_name, 'PTISR N jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTISR_N_lep'] = rt.TH2D('PTISR_N_lep_'+sample+'_'+tree_name, 'PTISR N lep', 500, 0, 1000, 20, 0, 20)

            hist[sample][tree_name]['PTISR_N_S_jet'] = rt.TH2D('PTISR_N_S_jet_'+sample+'_'+tree_name, 'PTISR N S jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTISR_N_S_lep'] = rt.TH2D('PTISR_N_S_lep_'+sample+'_'+tree_name, 'PTISR N S lep', 500, 0, 1000, 20, 0, 20)

            hist[sample][tree_name]['PTISR_N_ISR_jet'] = rt.TH2D('PTISR_N_ISR_jet_'+sample+'_'+tree_name, 'PTISR N ISR jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTISR_N_ISR_lep'] = rt.TH2D('PTISR_N_ISR_lep_'+sample+'_'+tree_name, 'PTISR N ISR lep', 500, 0, 1000, 20, 0, 20)

            hist[sample][tree_name]['PTCM_N_jet'] = rt.TH2D('PTCM_N_jet_'+sample+'_'+tree_name, 'PTCM N jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTCM_N_lep'] = rt.TH2D('PTCM_N_lep_'+sample+'_'+tree_name, 'PTCM N lep', 500, 0, 1000, 20, 0, 20)

            hist[sample][tree_name]['PTCM_N_S_jet'] = rt.TH2D('PTCM_N_S_jet_'+sample+'_'+tree_name, 'PTCM N S jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTCM_N_S_lep'] = rt.TH2D('PTCM_N_S_lep_'+sample+'_'+tree_name, 'PTCM N S lep', 500, 0, 1000, 20, 0, 20)

            hist[sample][tree_name]['PTCM_N_ISR_jet'] = rt.TH2D('PTCM_N_ISR_jet_'+sample+'_'+tree_name, 'PTCM N ISR jet', 500, 0, 1000, 20, 0, 20)
            hist[sample][tree_name]['PTCM_N_ISR_lep'] = rt.TH2D('PTCM_N_ISR_lep_'+sample+'_'+tree_name, 'PTCM N ISR lep', 500, 0, 1000, 20, 0, 20)

        i_entries = 0    
        for itree, in_tree in enumerate(list_of_files_[sample]['trees']):
            for events in ur.tree.iterate(list_of_files_[sample]['files'], in_tree, branches=variable_list_, entrysteps=10000):
                print('\nGetting Histograms for:', sample, tree_name)
                print('tree: ', itree+1)
                i_entries += 10000
                print(i_entries)

                print(events) 
                pt_jet = events[b'PT_jet']
                flavor_jet = events[b'Flavor_jet']
                isr_index_jet = events[b'index_jet_ISR']
                s_index_jet = events[b'index_jet_S']
                bjet_tag = events[b'Btag_jet']

                pt_lep = events[b'PT_lep']
                ch_lep = events[b'Charge_lep']
                id_lep = events[b'ID_lep']
                pdgid_lep = events[b'PDGID_lep']
                isr_index_lep = events[b'index_lep_ISR']
                s_index_lep = events[b'index_lep_S']

                met = events[b'MET']
                risr = aw.fromiter(events[b'RISR'])
                ptisr = events[b'PTISR']
                ptcm = events[b'PTCM']
                dphi = events[b'dphiCMI']
                weight = events[b'weight']

                len_jet = pt_jet.stops - pt_jet.starts
                max_n_jets = np.amax(len_jet)

                # pt_jet = ([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                # flavor_jet = ([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in flavor_jet]) 
                # bjet_tag = ([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in bjet_tag])

                len_lep = pt_lep.stops - pt_lep.starts
                max_n_leps = np.amax(len_lep)

                # pt_lep = ([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_lep]) 
                # ch_lep = ([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=0) for leps in ch_lep]) 
                # pdgid_lep = ([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pdgid_lep]) 
                only_2_leps = ([True if lep == 2 else False for lep in len_lep])
                only_2_opp_leps = ([True if lep == 2 and len(charge[charge>0])>0 and len(charge[charge<0])>0 else False for lep, charge in zip(only_2_leps, ch_lep)])

                isr_index_jet = np.array(isr_index_jet)
                s_index_jet = np.array(s_index_jet)
                isr_index_lep = np.array(isr_index_lep)
                s_index_lep = np.array(s_index_lep)

                risr = risr[:, 1]
                isr_index_jet = isr_index_jet[:, 1] 
                s_index_jet = s_index_jet[:, 1] 
                isr_index_lep = isr_index_lep[:, 1] 
                s_index_lep = s_index_lep[:, 1] 
                ptcm = ptcm.content[:, 1]
                dphi = dphi.content[:, 1]

                # risr_lepV_jetI = risr[:,0]
                # risr_lepV_jetA = risr[:,1]
                # risr_lepA_jetA = risr[:,2]
                
                print('\ncreating masks and weights')
                print('-> bjet masks')
                loose_mask = bjet_tag > 0.5426
                medium_mask = bjet_tag > 0.8484
                tight_mask = bjet_tag > 0.9535
                
                has_2_loose = ([True if len(mask[mask]) >= 2 else False for mask in loose_mask])
                has_2_medium = ([True if len(mask[mask]) >= 2 else False for mask in medium_mask])
                has_2_tight = ([True if len(mask[mask]) >= 2 else False for mask in tight_mask])

                print('-> S bjet masks')
                loose_s_mask = ([mask[index] for mask, index in zip(loose_mask, s_index_jet)])
                medium_s_mask = ([mask[index] for mask, index in zip(medium_mask, s_index_jet)])
                tight_s_mask = ([mask[index] for mask, index in zip(tight_mask, s_index_jet)])

                print('-> ISR bjet masks')
                loose_isr_mask = ([mask[index] for mask, index in zip(loose_mask, isr_index_jet)])
                medium_isr_mask = ([mask[index] for mask, index in zip(medium_mask, isr_index_jet)])
                tight_isr_mask = ([mask[index] for mask, index in zip(tight_mask, isr_index_jet)])

                print('-> event bjet masks')
                is_loose = ([np.any(event) for event in loose_mask])
                is_medium = ([np.any(event) for event in medium_mask])
                is_tight = ([np.any(event) for event in tight_mask])

                print('-> jet weights')
                jet_weight = ([np.array([np.float64(event)]*len(jets[~np.isnan(jets)])) for jets, event in zip(pt_jet, weight)]) 
                # jet_weight = ([np.pad(w, (0, max_n_jets - len(w)), 'constant', constant_values=np.nan) for w in jet_weight]) 

                s_jet_weight = ([jets[index] for jets, index in zip(jet_weight, s_index_jet)])
                isr_jet_weight = ([jets[index] for jets, index in zip(jet_weight, isr_index_jet)])

                pt_s_jet = ([jets[index] for jets, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = ([jets[index] for jets, index in zip(pt_jet, isr_index_jet)])

                flavor_s_jet = ([jets[index] for jets, index in zip(flavor_jet, s_index_jet)])
                flavor_isr_jet = ([jets[index] for jets, index in zip(flavor_jet, isr_index_jet)])

                print('-> lep weights')
                lep_weight = ([np.array([np.float64(event)]*len(leps[~np.isnan(leps)])) for leps, event in zip(pt_lep, weight)]) 
                # lep_weight = ([np.pad(w, (0, max_n_leps - len(w)), 'constant', constant_values=np.nan) for w in lep_weight]) 

                s_lep_weight = ([leps[index] for leps, index in zip(lep_weight, s_index_lep)])
                isr_lep_weight = ([leps[index] for leps, index in zip(lep_weight, isr_index_lep)])

                pt_s_lep = ([leps[index] for leps, index in zip(pt_lep, s_index_lep)])
                pt_isr_lep = ([leps[index] for leps, index in zip(pt_lep, isr_index_lep)])

                pdgid_s_lep = ([leps[index] for leps, index in zip(pdgid_lep, s_index_lep)])
                pdgid_isr_lep = ([leps[index] for leps, index in zip(pdgid_lep, isr_index_lep)])

                print('\napplying masks')
                print('-> jet pt')
                loose_pt_jet = ([jet[mask] for jet, mask in zip(pt_jet, loose_mask)])
                medium_pt_jet = ([jet[mask] for jet, mask in zip(pt_jet, medium_mask)])
                tight_pt_jet = ([jet[mask] for jet, mask in zip(pt_jet, tight_mask)])

                print('-> N S jets')
                n_s_jet = ([len(jets[~np.isnan(jets)]) for jets in pt_s_jet])
                n_s_loose_jet = ([len(jets[mask]) for jets, mask in zip(pt_s_jet, loose_s_mask)])
                n_s_medium_jet = ([len(jets[mask]) for jets, mask in zip(pt_s_jet, medium_s_mask)])
                n_s_tight_jet = ([len(jets[mask]) for jets, mask in zip(pt_s_jet, tight_s_mask)])

                print('-> N ISR jets')
                n_isr_jet = ([len(jets[~np.isnan(jets)]) for jets in pt_isr_jet])
                n_isr_loose_jet = ([len(jets[mask]) for jets, mask in zip(pt_isr_jet, loose_isr_mask)])
                n_isr_medium_jet = ([len(jets[mask]) for jets, mask in zip(pt_isr_jet, medium_isr_mask)])
                n_isr_tight_jet = ([len(jets[mask]) for jets, mask in zip(pt_isr_jet, tight_isr_mask)])

                print('-> N S leps')
                n_s_lep = ([len(leps[~np.isnan(leps)]) for leps in pt_s_lep])

                print('-> N ISR leps')
                n_isr_lep = ([len(leps[~np.isnan(leps)]) for leps in pt_isr_lep])

                print('-> Event variables')
                ptcm_div_ptisr = np.divide(ptcm, ptisr)

                print('-> jet weights')
#                loose_weight = weight[is_loose]
#                medium_weight = weight[is_medium]
#                tight_weight = weight[is_tight]
#
#                loose_jet_weight = ([w[mask] for w, mask in zip(jet_weight, loose_mask)])
#                medium_jet_weight = ([w[mask] for w, mask in zip(jet_weight, medium_mask)])
#                tight_jet_weight = ([w[mask] for w, mask in zip(jet_weight, tight_mask)])
#
#                loose_s_jet_weight = ([w[mask] for w, mask in zip(s_jet_weight, loose_s_mask)])
#                medium_s_jet_weight = ([w[mask] for w, mask in zip(s_jet_weight, medium_s_mask)])
#                tight_s_jet_weight = ([w[mask] for w, mask in zip(s_jet_weight, tight_s_mask)])
#
#                loose_isr_jet_weight = (w[mask] for w, mask in zip(isr_jet_weight, loose_isr_mask)])
#                medium_isr_jet_weight = (w[mask] for w, mask in zip(isr_jet_weight, medium_isr_mask)])
#                tight_isr_jet_weight = (w[mask] for w, mask in zip(isr_jet_weight, tight_isr_mask)])


                print('-> Overall selection mask')
                evt_selection_mask = ([True if np.all([lep_mask, b_mask]) else False for lep_mask, b_mask in zip(only_2_leps, is_medium)])
 
                risr = risr[evt_selection_mask]
                ptisr = ptisr[evt_selection_mask]
                ptcm = ptcm[evt_selection_mask]
                met = met[evt_selection_mask]

                lep_weight = lep_weight[evt_selection_mask]
                pdgid_lep = pdgid_lep[evt_selection_mask]
                ch_lep = ch_lep[evt_selection_mask]
                flavor_jet = flavor_jet[evt_selection_mask]
                flavor_s_jet = flavor_s_jet[evt_selection_mask]
                flavor_isr_jet = flavor_isr_jet[evt_selection_mask]
                pdgid_s_lep = pdgid_s_lep[evt_selection_mask]
                pdgid_isr_lep = pdgid_isr_lep[evt_selection_mask]

                dphi = dphi[evt_selection_mask]
                ptcm_div_ptisr = ptcm_div_ptisr[evt_selection_mask]
                
                n_s_jet = n_s_jet[evt_selection_mask]
                n_s_loose_jet = n_s_loose_jet[evt_selection_mask]
                n_s_medium_jet = n_s_medium_jet[evt_selection_mask]
                n_s_tight_jet = n_s_tight_jet[evt_selection_mask]
                
                n_s_lep = n_s_lep[evt_selection_mask]
                 
                n_isr_jet = n_isr_jet[evt_selection_mask]
                n_isr_loose_jet = n_isr_loose_jet[evt_selection_mask]
                n_isr_medium_jet = n_isr_medium_jet[evt_selection_mask]
                n_isr_tight_jet = n_isr_tight_jet[evt_selection_mask]
                
                n_isr_lep = n_isr_lep[evt_selection_mask]
                 
                len_jet = len_jet[evt_selection_mask]
                len_lep = len_lep[evt_selection_mask]
                only_lep_weight = weight
                weight = weight[evt_selection_mask]
                s_jet_weight = s_jet_weight[evt_selection_mask]
                isr_jet_weight = isr_jet_weight[evt_selection_mask]
                s_lep_weight = s_lep_weight[evt_selection_mask]
                isr_lep_weight = isr_lep_weight[evt_selection_mask]

                print('done applying masks')
                print('\nfilling histograms')

                if not np.any(evt_selection_mask): 
                    print('finished filling')
                    continue

#                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)
#                rnp.fill_hist(hist[sample][tree_name]['S_Flavor_jet'], flavor_s_jet, s_jet_weight)
#                rnp.fill_hist(hist[sample][tree_name]['ISR_Flavor_jet'], flavor_isr_jet, isr_jet_weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['S_Flavor_lep'], pdgid_s_lep, s_lep_weight)
#                rnp.fill_hist(hist[sample][tree_name]['ISR_Flavor_lep'], pdgid_isr_lep, isr_lep_weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['Lep_to_Charge'], np.swapaxes([pdgid_lep, ch_lep],0,1), lep_weight)
#                rnp.fill_hist(hist[sample][tree_name]['Lep_to_Lep'], np.swapaxes([only_2_leps, only_2_opp_leps],0,1), only_lep_weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['RISR'], risr, weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTISR'], ptisr, weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTCM'], ptcm, weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['RISR_PTCM'], np.swapaxes([risr,ptcm],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['RISR_PTISR'], np.swapaxes([risr,ptisr],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_PTCM'], np.swapaxes([ptisr,ptcm],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM'], np.swapaxes([dphi,ptcm],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM_div_PTISR'], np.swapaxes([div_dphi,ptcm_div_ptisr],0,1), div_weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_div_PTISR'], ptcm_div_ptisr, div_weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_jet'], np.swapaxes([n_s_jet,n_isr_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_lep'], np.swapaxes([n_s_lep,n_isr_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_loose_jet'], np.swapaxes([n_s_loose_jet,n_isr_loose_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_medium_jet'], np.swapaxes([n_s_medium_jet,n_isr_medium_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_tight_jet'], np.swapaxes([n_s_tight_jet,n_isr_tight_jet],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_jet'], np.swapaxes([risr,len_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_lep'], np.swapaxes([risr,len_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_S_jet'], np.swapaxes([risr,n_s_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_S_lep'], np.swapaxes([risr,n_s_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_ISR_jet'], np.swapaxes([risr,n_isr_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['RISR_N_ISR_lep'], np.swapaxes([risr,n_isr_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_jet'], np.swapaxes([ptisr,len_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_lep'], np.swapaxes([ptisr,len_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_S_jet'], np.swapaxes([ptisr,n_s_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_S_lep'], np.swapaxes([ptisr,n_s_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_ISR_jet'], np.swapaxes([ptisr,n_isr_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_ISR_lep'], np.swapaxes([ptisr,n_isr_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_jet'], np.swapaxes([ptcm,len_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_lep'], np.swapaxes([ptcm,len_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_S_jet'], np.swapaxes([ptcm,n_s_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_S_lep'], np.swapaxes([ptcm,n_s_lep],0,1), weight)
#
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_ISR_jet'], np.swapaxes([ptcm,n_isr_jet],0,1), weight)
#                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_ISR_lep'], np.swapaxes([ptcm,n_isr_lep],0,1), weight)


                print('finished filling')
    return hist


             

if __name__ == "__main__":

    signals = { 
    'SMS-T2-4bd_420' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_',
],
    'SMS-T2-4bd_490' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_',
],
              }
    backgrounds = {
    'TTJets_2017' : ['/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X'],
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
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'PTISR', 'PTCM', 'dphiCMI', 'weight']

    start_b = time.time()    
    background_list = process_the_samples(backgrounds, None, None)
    hist_background = get_histograms(background_list, variables, None)

#    write_hists_to_file(hist_background, './output_background_2l1b_cat3_hists_uproottest.root') 
    stop_b = time.time()
#
#    signal_list = process_the_samples(signals, None, None)
#    hist_signal = get_histograms(signal_list, variables, None)
#
#    write_hists_to_file(hist_signal, './output_signal_2l1b_cat3_hists_uproottest.root')  
#    stop_s = time.time()
#
    print("background: ", stop_b - start_b)
#    print "signal:     ", stop_s - stop_b
#    print "total:      ", stop_s - start_b
# 
#    print 'finished writing'
