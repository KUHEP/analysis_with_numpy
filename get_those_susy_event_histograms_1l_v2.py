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

            hist[sample][tree_name]['MET'] = rt.TH1D('MET_'+sample+'_'+tree_name, 'E_{T}^{miss} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['S_Flavor_jet'] = rt.TH1D('S_Flavor_jet_'+sample+'_'+tree_name, 'Flavor S jets', 20, 0, 20)
            hist[sample][tree_name]['ISR_Flavor_jet'] = rt.TH1D('ISR_Flavor_jet_'+sample+'_'+tree_name, 'Flavor ISR jets', 20, 0, 20)
            hist[sample][tree_name]['S_Flavor_lep'] = rt.TH1D('S_Flavor_lep_'+sample+'_'+tree_name, 'Flavor S leps', 20, 0, 20)
            hist[sample][tree_name]['ISR_Flavor_lep'] = rt.TH1D('ISR_Flavor_lep_'+sample+'_'+tree_name, 'Flavor ISR leps', 20, 0, 20)

            hist[sample][tree_name]['Charge'] = rt.TH1D('Charge_'+sample+'_'+tree_name, 'lep Charge', 3, -1, 2)
            hist[sample][tree_name]['Lep_to_Lep'] = rt.TH2D('Lep_to_Lep_'+sample+'_'+tree_name, '2leps to 2 opp leps', 2, 0, 2, 2, 0, 2)

            hist[sample][tree_name]['N_PT_medium_jet'] = rt.TH2D('N_PT_medium_jet_'+sample+'_'+tree_name, 'N pt leading medium b jets', 20, 0, 20, 500, 0, 1000)
            hist[sample][tree_name]['N_PT_tight_jet'] = rt.TH2D('N_PT_tight_jet_'+sample+'_'+tree_name, 'N pt leading tight b jets', 20, 0, 20, 500, 0, 1000)
            hist[sample][tree_name]['N_PT_loose_jet'] = rt.TH2D('N_PT_loose_jet_'+sample+'_'+tree_name, 'N pt leading loose b jets', 20, 0, 20, 500, 0, 1000)
            hist[sample][tree_name]['N_PT_lep'] = rt.TH2D('N_PT_lep_'+sample+'_'+tree_name, 'N pt leptons', 20, 0, 20, 500, 0, 1000)
            hist[sample][tree_name]['PT_lep'] = rt.TH1D('PT_lep_'+sample+'_'+tree_name, 'lep p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PT_s_lep'] = rt.TH1D('PT_s_lep_'+sample+'_'+tree_name, 's lep p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PT_isr_lep'] = rt.TH1D('PT_isr_lep_'+sample+'_'+tree_name, 'isr lep p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PT_jet'] = rt.TH1D('PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PT_s_jet'] = rt.TH1D('PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['PT_isr_jet'] = rt.TH1D('PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['loose_PT_jet'] = rt.TH1D('loose_PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['medium_PT_jet'] = rt.TH1D('medium_PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)
            hist[sample][tree_name]['tight_PT_jet'] = rt.TH1D('tight_PT_jet_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 500, 0, 1000)

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

            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                pt_jet = np.array(sample_array[sample][tree_name]['PT_jet'])
                flavor_jet = np.array(sample_array[sample][tree_name]['Flavor_jet'])
                base_isr_index_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                base_s_index_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                bjet_tag = np.array(sample_array[sample][tree_name]['Btag_jet'])

                pt_lep = np.array(sample_array[sample][tree_name]['PT_lep'])
                ch_lep = np.array(sample_array[sample][tree_name]['Charge_lep'])
                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])
                pdgid_lep = np.array(sample_array[sample][tree_name]['PDGID_lep'])
                base_isr_index_lep = np.array(sample_array[sample][tree_name]['index_lep_ISR'])
                base_s_index_lep = np.array(sample_array[sample][tree_name]['index_lep_S'])

                met = np.array(sample_array[sample][tree_name]['MET'])
                base_risr = np.array(sample_array[sample][tree_name]['RISR'])
                base_ptisr = np.array(sample_array[sample][tree_name]['PTISR'])
                base_ptcm = np.array(sample_array[sample][tree_name]['PTCM'])
                base_dphi = np.array(sample_array[sample][tree_name]['dphiCMI'])
                weight = np.array(sample_array[sample][tree_name]['weight'])

                len_jet = np.array([len(jets) for jets in pt_jet])
                max_n_jets = np.amax(len_jet)

                pt_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_jet]) 
                flavor_jet = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in flavor_jet]) 
                bjet_tag = np.array([np.pad(jets, (0, max_n_jets - len(jets)), 'constant', constant_values=np.nan) for jets in bjet_tag])

                len_lep = np.array([len(leps) for leps in pt_lep])
                max_n_leps = np.amax(len_lep)

                pt_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_lep]) 
                ch_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=0) for leps in ch_lep]) 
                pdgid_lep = np.array([np.pad(leps, (0, max_n_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pdgid_lep]) 
                only_2_leps = np.array([True if lep == 2 else False for lep in len_lep])
                only_1_leps = np.array([True if lep == 1 else False for lep in len_lep])
                only_2_opp_leps = np.array([True if lep == 2 and len(charge[charge>0])>0 and len(charge[charge<0])>0 else False for lep, charge in zip(only_2_leps, ch_lep)])

                risr = np.array([entry[:3] for entry in base_risr])
                ptisr = np.array([entry[:3] for entry in base_ptisr])
                ptcm = np.array([entry[:3] for entry in base_ptcm])
                dphi = np.array([entry[:3] for entry in base_dphi])
                isr_index_jet = np.array([entry[:3] for entry in base_isr_index_jet])
                s_index_jet = np.array([entry[:3] for entry in base_s_index_jet])
                isr_index_lep = np.array([entry[:3] for entry in base_isr_index_lep])
                s_index_lep = np.array([entry[:3] for entry in base_s_index_lep])

                risr = risr[:, 2]
                ptisr = ptisr[:, 2]
                ptcm = ptcm[:, 2]
                isr_index_jet = isr_index_jet[:, 2]
                s_index_jet = s_index_jet[:, 2]
                isr_index_lep = isr_index_lep[:, 2]
                s_index_lep = s_index_lep[:, 2]
                dphi = dphi[:, 2]

                # risr_lepV_jetI = risr[:,0]
                # risr_lepV_jetA = risr[:,1]
                # risr_lepA_jetA = risr[:,2]
                
                print '\ncreating masks and weights'
                print '-> bjet masks'
                loose_mask = bjet_tag > 0.5426
                medium_mask = bjet_tag > 0.8484
                tight_mask = bjet_tag > 0.9535
                
                has_2_loose = np.array([True if len(mask[mask]) >= 2 else False for mask in loose_mask])
                has_2_medium = np.array([True if len(mask[mask]) >= 2 else False for mask in medium_mask])
                has_2_tight = np.array([True if len(mask[mask]) >= 2 else False for mask in tight_mask])

                print '-> S bjet masks'
                loose_s_mask = np.array([mask[index] for mask, index in zip(loose_mask, s_index_jet)])
                medium_s_mask = np.array([mask[index] for mask, index in zip(medium_mask, s_index_jet)])
                tight_s_mask = np.array([mask[index] for mask, index in zip(tight_mask, s_index_jet)])

                print '-> ISR bjet masks'
                loose_isr_mask = np.array([mask[index] for mask, index in zip(loose_mask, isr_index_jet)])
                medium_isr_mask = np.array([mask[index] for mask, index in zip(medium_mask, isr_index_jet)])
                tight_isr_mask = np.array([mask[index] for mask, index in zip(tight_mask, isr_index_jet)])

                print '-> event bjet masks'
                is_loose = np.array([np.any(event) for event in loose_mask])
                is_medium = np.array([np.any(event) for event in medium_mask])
                is_tight = np.array([np.any(event) for event in tight_mask])

                print '-> jet weights'
                jet_weight = np.array([np.array([np.float64(event)]*len(jets[~np.isnan(jets)])) for jets, event in zip(pt_jet, weight)]) 
                jet_weight = np.array([np.pad(w, (0, max_n_jets - len(w)), 'constant', constant_values=np.nan) for w in jet_weight]) 

                s_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, s_index_jet)])
                isr_jet_weight = np.array([jets[index] for jets, index in zip(jet_weight, isr_index_jet)])

                pt_s_jet = np.array([jets[index] for jets, index in zip(pt_jet, s_index_jet)])
                pt_isr_jet = np.array([jets[index] for jets, index in zip(pt_jet, isr_index_jet)])

                flavor_s_jet = np.array([jets[index] for jets, index in zip(flavor_jet, s_index_jet)])
                flavor_isr_jet = np.array([jets[index] for jets, index in zip(flavor_jet, isr_index_jet)])

                print '-> lep weights'
                lep_weight = np.array([np.array([np.float64(event)]*len(leps[~np.isnan(leps)])) for leps, event in zip(pt_lep, weight)]) 
                lep_weight = np.array([np.pad(w, (0, max_n_leps - len(w)), 'constant', constant_values=np.nan) for w in lep_weight]) 

                s_lep_weight = np.array([leps[index] for leps, index in zip(lep_weight, s_index_lep)])
                isr_lep_weight = np.array([leps[index] for leps, index in zip(lep_weight, isr_index_lep)])

                pt_s_lep = np.array([leps[index] for leps, index in zip(pt_lep, s_index_lep)])
                pt_isr_lep = np.array([leps[index] for leps, index in zip(pt_lep, isr_index_lep)])

                pdgid_s_lep = np.array([leps[index] for leps, index in zip(pdgid_lep, s_index_lep)])
                pdgid_isr_lep = np.array([leps[index] for leps, index in zip(pdgid_lep, isr_index_lep)])

                print '\napplying masks'
                print '-> jet pt'
                loose_pt_jet = np.array([jet[mask] for jet, mask in zip(pt_jet, loose_mask)])
                medium_pt_jet = np.array([jet[mask] for jet, mask in zip(pt_jet, medium_mask)])
                tight_pt_jet = np.array([jet[mask] for jet, mask in zip(pt_jet, tight_mask)])

                loose_jet_weight = np.array([jet[mask] for jet, mask in zip(jet_weight, loose_mask)])
                medium_jet_weight = np.array([jet[mask] for jet, mask in zip(jet_weight, medium_mask)])
                tight_jet_weight = np.array([jet[mask] for jet, mask in zip(jet_weight, tight_mask)])

                print '-> S jet pt + padding'
                s_len_jet = np.array([len(jets) for jets in pt_s_jet])
                max_n_s_jets = np.amax(s_len_jet)
                pt_s_jet = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_s_jet]) 
                flavor_s_jet = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in flavor_s_jet]) 
                s_jet_weight = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=np.nan) for jets in s_jet_weight]) 
                loose_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in loose_s_mask]) 
                medium_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in medium_s_mask]) 
                tight_s_mask = np.array([np.pad(jets, (0, max_n_s_jets - len(jets)), 'constant', constant_values=False) for jets in tight_s_mask]) 

                print '-> ISR jet pt + padding'
                isr_len_jet = np.array([len(jets) for jets in pt_isr_jet])
                max_n_isr_jets = np.amax(isr_len_jet)
                pt_isr_jet = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in pt_isr_jet]) 
                flavor_isr_jet = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in flavor_isr_jet]) 
                isr_jet_weight = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=np.nan) for jets in isr_jet_weight]) 
                loose_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in loose_isr_mask]) 
                medium_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in medium_isr_mask]) 
                tight_isr_mask = np.array([np.pad(jets, (0, max_n_isr_jets - len(jets)), 'constant', constant_values=False) for jets in tight_isr_mask]) 

                print '-> S lep pt + padding'
                s_len_lep = np.array([len(leps) for leps in pt_s_lep])
                max_n_s_leps = np.amax(s_len_lep)
                pt_s_lep = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_s_lep]) 
                pdgid_s_lep = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pdgid_s_lep]) 
                s_lep_weight = np.array([np.pad(leps, (0, max_n_s_leps - len(leps)), 'constant', constant_values=np.nan) for leps in s_lep_weight]) 

                print '-> ISR lep pt + padding'
                isr_len_lep = np.array([len(leps) for leps in pt_isr_lep])
                max_n_isr_leps = np.amax(isr_len_lep)
                pt_isr_lep = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pt_isr_lep]) 
                pdgid_isr_lep = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in pdgid_isr_lep]) 
                isr_lep_weight = np.array([np.pad(leps, (0, max_n_isr_leps - len(leps)), 'constant', constant_values=np.nan) for leps in isr_lep_weight]) 

                print '-> N S jets'
                n_s_jet = np.array([len(jets[~np.isnan(jets)]) for jets in pt_s_jet])
                n_s_loose_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_s_jet, loose_s_mask)])
                n_s_medium_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_s_jet, medium_s_mask)])
                n_s_tight_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_s_jet, tight_s_mask)])

                print '-> N ISR jets'
                n_isr_jet = np.array([len(jets[~np.isnan(jets)]) for jets in pt_isr_jet])
                n_isr_loose_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_isr_jet, loose_isr_mask)])
                n_isr_medium_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_isr_jet, medium_isr_mask)])
                n_isr_tight_jet = np.array([len(jets[mask]) for jets, mask in zip(pt_isr_jet, tight_isr_mask)])

                print '-> N S leps'
                n_s_lep = np.array([len(leps[~np.isnan(leps)]) for leps in pt_s_lep])

                print '-> N ISR leps'
                n_isr_lep = np.array([len(leps[~np.isnan(leps)]) for leps in pt_isr_lep])

                print '-> Event variables' 
                ptcm_div_ptisr = np.divide(ptcm, ptisr)

                print '-> jet weights'
#                loose_weight = weight[is_loose]
#                medium_weight = weight[is_medium]
#                tight_weight = weight[is_tight]
#
#                loose_jet_weight = np.array([w[mask] for w, mask in zip(jet_weight, loose_mask)])
#                medium_jet_weight = np.array([w[mask] for w, mask in zip(jet_weight, medium_mask)])
#                tight_jet_weight = np.array([w[mask] for w, mask in zip(jet_weight, tight_mask)])
#
#                loose_s_jet_weight = np.array([w[mask] for w, mask in zip(s_jet_weight, loose_s_mask)])
#                medium_s_jet_weight = np.array([w[mask] for w, mask in zip(s_jet_weight, medium_s_mask)])
#                tight_s_jet_weight = np.array([w[mask] for w, mask in zip(s_jet_weight, tight_s_mask)])
#
#                loose_isr_jet_weight = np.array(w[mask] for w, mask in zip(isr_jet_weight, loose_isr_mask)])
#                medium_isr_jet_weight = np.array(w[mask] for w, mask in zip(isr_jet_weight, medium_isr_mask)])
#                tight_isr_jet_weight = np.array(w[mask] for w, mask in zip(isr_jet_weight, tight_isr_mask)])

                risr_0p8 = risr > 0.8
                print '-> Overall selection mask'
                #evt_selection_mask = np.array([True if np.all([lep_mask, b_mask]) else False for lep_mask, b_mask in zip(only_1_leps, is_medium)])
                evt_selection_mask = np.all([only_1_leps, risr_0p8], axis=0)
 
                risr = risr[evt_selection_mask]
                ptisr = ptisr[evt_selection_mask]
                ptcm = ptcm[evt_selection_mask]
                met = met[evt_selection_mask]

                pt_lep = pt_lep[evt_selection_mask]
                pt_s_lep = pt_s_lep[evt_selection_mask]
                pt_isr_lep = pt_isr_lep[evt_selection_mask]
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
                pt_jet = pt_jet[evt_selection_mask]
                jet_weight = jet_weight[evt_selection_mask]
                len_lep = len_lep[evt_selection_mask]
                only_lep_weight = weight
                weight = weight[evt_selection_mask]
                pt_s_jet = pt_s_jet[evt_selection_mask]
                pt_isr_jet = pt_isr_jet[evt_selection_mask]
                s_jet_weight = s_jet_weight[evt_selection_mask]
                isr_jet_weight = isr_jet_weight[evt_selection_mask]
                s_lep_weight = s_lep_weight[evt_selection_mask]
                isr_lep_weight = isr_lep_weight[evt_selection_mask]

                print 'done applying masks'
                print '\nfilling histograms'

                flavor_jet = np.array([jet[~np.isnan(jet)] for jet in flavor_jet])
                jet_weight = np.array([w[~np.isnan(jet)] for w, jet in zip(jet_weight,pt_jet)])
                pt_jet = np.array([jet[~np.isnan(jet)] for jet in pt_jet])
                
                s_jet_weight = s_jet_weight[~np.isnan(s_jet_weight)]
                flavor_s_jet = flavor_s_jet[~np.isnan(flavor_s_jet)]
                pt_s_jet = pt_s_jet[~np.isnan(pt_s_jet)]
                
                isr_jet_weight = isr_jet_weight[~np.isnan(isr_jet_weight)]
                flavor_isr_jet = flavor_isr_jet[~np.isnan(flavor_isr_jet)]
                pt_isr_jet = pt_isr_jet[~np.isnan(pt_isr_jet)]
               
                pdgid_lep = pdgid_lep[~np.isnan(lep_weight)]
                ch_lep = ch_lep[~np.isnan(lep_weight)]
                lep_weight = lep_weight[~np.isnan(lep_weight)]
                pt_lep = pt_lep[~np.isnan(pt_lep)]
                
                pdgid_s_lep = pdgid_s_lep[~np.isnan(s_lep_weight)]
                s_lep_weight = s_lep_weight[~np.isnan(s_lep_weight)]
                pt_s_lep = pt_s_lep[~np.isnan(pt_s_lep)]
                
                pdgid_isr_lep = pdgid_isr_lep[~np.isnan(isr_lep_weight)]
                isr_lep_weight = isr_lep_weight[~np.isnan(isr_lep_weight)]
                pt_isr_lep = pt_isr_lep[~np.isnan(pt_isr_lep)]
 
                div_weight = weight[~np.isnan(ptcm_div_ptisr)]
                div_dphi = dphi[~np.isnan(ptcm_div_ptisr)]
                ptcm_div_ptisr = ptcm_div_ptisr[~np.isnan(ptcm_div_ptisr)]

                if not np.any(evt_selection_mask): 
                    print 'finished filling'
                    continue
                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)
                rnp.fill_hist(hist[sample][tree_name]['S_Flavor_jet'], flavor_s_jet, s_jet_weight)
                rnp.fill_hist(hist[sample][tree_name]['ISR_Flavor_jet'], flavor_isr_jet, isr_jet_weight)

                rnp.fill_hist(hist[sample][tree_name]['S_Flavor_lep'], pdgid_s_lep, s_lep_weight)
                rnp.fill_hist(hist[sample][tree_name]['ISR_Flavor_lep'], pdgid_isr_lep, isr_lep_weight)

                rnp.fill_hist(hist[sample][tree_name]['Charge'], ch_lep, lep_weight)
                rnp.fill_hist(hist[sample][tree_name]['Lep_to_Lep'], np.swapaxes([only_2_leps, only_2_opp_leps],0,1), only_lep_weight)

                rnp.fill_hist(hist[sample][tree_name]['PT_lep'], pt_lep, lep_weight)
                rnp.fill_hist(hist[sample][tree_name]['PT_s_lep'], pt_s_lep, s_lep_weight)
                rnp.fill_hist(hist[sample][tree_name]['PT_isr_lep'], pt_isr_lep, isr_lep_weight)
                rnp.fill_hist(hist[sample][tree_name]['PT_jet'], np.concatenate(pt_jet), np.concatenate(jet_weight))
                rnp.fill_hist(hist[sample][tree_name]['PT_s_jet'], (pt_s_jet), (s_jet_weight))
                rnp.fill_hist(hist[sample][tree_name]['PT_isr_jet'], (pt_isr_jet), (isr_jet_weight))
                rnp.fill_hist(hist[sample][tree_name]['loose_PT_jet'], np.concatenate(loose_pt_jet), np.concatenate(loose_jet_weight))
                rnp.fill_hist(hist[sample][tree_name]['medium_PT_jet'], np.concatenate(medium_pt_jet), np.concatenate(medium_jet_weight))
                rnp.fill_hist(hist[sample][tree_name]['tight_PT_jet'], np.concatenate(tight_pt_jet), np.concatenate(tight_jet_weight))

                rnp.fill_hist(hist[sample][tree_name]['RISR'], risr, weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR'], ptisr, weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM'], ptcm, weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_PTCM'], np.swapaxes([risr,ptcm],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['RISR_PTISR'], np.swapaxes([risr,ptisr],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR_PTCM'], np.swapaxes([ptisr,ptcm],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM'], np.swapaxes([dphi,ptcm],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['dphi_PTCM_div_PTISR'], np.swapaxes([div_dphi,ptcm_div_ptisr],0,1), div_weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM_div_PTISR'], ptcm_div_ptisr, div_weight)

                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_jet'], np.swapaxes([n_s_jet,n_isr_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_lep'], np.swapaxes([n_s_lep,n_isr_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_loose_jet'], np.swapaxes([n_s_loose_jet,n_isr_loose_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_medium_jet'], np.swapaxes([n_s_medium_jet,n_isr_medium_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['S_ISR_N_tight_jet'], np.swapaxes([n_s_tight_jet,n_isr_tight_jet],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_N_jet'], np.swapaxes([risr,len_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['RISR_N_lep'], np.swapaxes([risr,len_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_N_S_jet'], np.swapaxes([risr,n_s_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['RISR_N_S_lep'], np.swapaxes([risr,n_s_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['RISR_N_ISR_jet'], np.swapaxes([risr,n_isr_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['RISR_N_ISR_lep'], np.swapaxes([risr,n_isr_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_jet'], np.swapaxes([ptisr,len_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_lep'], np.swapaxes([ptisr,len_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_S_jet'], np.swapaxes([ptisr,n_s_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_S_lep'], np.swapaxes([ptisr,n_s_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_ISR_jet'], np.swapaxes([ptisr,n_isr_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTISR_N_ISR_lep'], np.swapaxes([ptisr,n_isr_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_jet'], np.swapaxes([ptcm,len_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_lep'], np.swapaxes([ptcm,len_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_S_jet'], np.swapaxes([ptcm,n_s_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_S_lep'], np.swapaxes([ptcm,n_s_lep],0,1), weight)

                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_ISR_jet'], np.swapaxes([ptcm,n_isr_jet],0,1), weight)
                rnp.fill_hist(hist[sample][tree_name]['PTCM_N_ISR_lep'], np.swapaxes([ptcm,n_isr_lep],0,1), weight)


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
    variables = ['MET', 'PT_jet', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'Flavor_jet', 'PT_lep', 'Charge_lep', 'ID_lep', 'PDGID_lep', 'index_lep_ISR', 'index_lep_S', 'RISR', 'PTISR', 'PTCM', 'dphiCMI', 'weight']

    start_b = time.time()    
    background_list = process_the_samples(backgrounds, None, None)
    hist_background = get_histograms(background_list, variables, None)

    write_hists_to_file(hist_background, './output_background_1l_cat3_risr_0p8_hists.root') 
    stop_b = time.time()

    signal_list = process_the_samples(signals, None, None)
    hist_signal = get_histograms(signal_list, variables, None)

    write_hists_to_file(hist_signal, './output_signal_1l_cat3_risr_0p8_hists.root')  
    stop_s = time.time()

    print "background: ", stop_b - start_b
    print "signal:     ", stop_s - stop_b
    print "total:      ", stop_s - start_b
 
    print 'finished writing'
