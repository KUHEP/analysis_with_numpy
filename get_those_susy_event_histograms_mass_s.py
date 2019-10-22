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
            hist[sample][tree_name]['dphi_lep_MET'] = rt.TH1D('dphi_lep_MET_'+sample+'_'+tree_name, 'dphi l met', 1024, 0, np.pi)
            hist[sample][tree_name]['MS'] = rt.TH1D('MS_'+sample+'_'+tree_name, 'MS', 1024, 0, 1000)
            hist[sample][tree_name]['MLL'] = rt.TH1D('MLL_'+sample+'_'+tree_name, 'MLL', 1024, 0, 200)
            hist[sample][tree_name]['PS'] = rt.TH1D('PS_'+sample+'_'+tree_name, 'PS', 1024, 0, 1000)
            hist[sample][tree_name]['PTS'] = rt.TH1D('PTS_'+sample+'_'+tree_name, 'PTS', 1024, 0, 1000)
            hist[sample][tree_name]['PzS'] = rt.TH1D('PzS_'+sample+'_'+tree_name, 'PzS', 1024, 0, 1000)
            hist[sample][tree_name]['PzS_div_PTISR'] = rt.TH1D('PzS_div_PTISR_'+sample+'_'+tree_name, 'PzS / PzS + MS', 1024, 0, 1)
            hist[sample][tree_name]['PTS_div_PTISR'] = rt.TH1D('PTS_div_PTISR_'+sample+'_'+tree_name, 'PTS / PTS + MS', 1024, 0, 1)
            hist[sample][tree_name]['MS_dphiLepMET'] = rt.TH2D('MS_dphiLepMET_'+sample+'_'+tree_name, 'MS vs dphiLepMET', 1024, 0, 1000, 1024, 0, np.pi)
            hist[sample][tree_name]['RISR_MS'] = rt.TH2D('RISR_MS_'+sample+'_'+tree_name, 'MS vs RISR', 1024, 0, 2, 1024, 0, 1000)
            hist[sample][tree_name]['RISR_PTISR'] = rt.TH2D('RISR_MS_'+sample+'_'+tree_name, 'MS vs RISR', 1024, 0, 2, 1024, 0, 1000)
            hist[sample][tree_name]['PTISR_MS'] = rt.TH2D('PTISR_MS_'+sample+'_'+tree_name, 'MS vs PTISR', 1024, 0, 1000, 1024, 0, 1000)
            hist[sample][tree_name]['cos_lep_S'] = rt.TH1D('cos_lep_S_'+sample+'_'+tree_name, 'cos_lep_S', 1024, -1, 1)
            hist[sample][tree_name]['dphi_lep_S'] = rt.TH1D('dphi_lep_S_'+sample+'_'+tree_name, 'dphi_lep_S', 1024, 0, np.pi)
            
        for ifile, in_file in enumerate(list_of_files_[sample]['files']):
            sample_array = get_tree_info_singular(sample, in_file, list_of_files_[sample]['trees'], variable_list_, cuts_to_apply_)
            for tree_name in sample_array[sample]:
                print '\nGetting Histograms for:', sample, tree_name, in_file
                print 'file: ', ifile+1, ' / ', len(list_of_files_[sample]['files'])

                met = np.array(sample_array[sample][tree_name]['MET'])
                phi_met = np.array(sample_array[sample][tree_name]['MET_phi'])
                risr = np.array(sample_array[sample][tree_name]['RISR'])
                ptisr = np.array(sample_array[sample][tree_name]['PTISR'])

                ms = np.array(sample_array[sample][tree_name]['MS'])
                ps = np.array(sample_array[sample][tree_name]['PS'])
                pts = np.array(sample_array[sample][tree_name]['PTS'])
                pzs = np.array(sample_array[sample][tree_name]['PzS'])
                dpcmi = np.array(sample_array[sample][tree_name]['dphiCMI'])
                dpsi = np.array(sample_array[sample][tree_name]['dphiSI'])

                weight = np.array(sample_array[sample][tree_name]['weight'])

                id_lep = np.array(sample_array[sample][tree_name]['ID_lep'])
                pt_lep = np.array(sample_array[sample][tree_name]['PT_lep'])
                phi_lep = np.array(sample_array[sample][tree_name]['Phi_lep'])
                eta_lep = np.array(sample_array[sample][tree_name]['Eta_lep'])
                m_lep = np.array(sample_array[sample][tree_name]['M_lep'])
                mini_lep = np.array(sample_array[sample][tree_name]['MiniIso_lep'])
                
                index_isr_jet = np.array(sample_array[sample][tree_name]['index_jet_ISR'])
                index_s_jet = np.array(sample_array[sample][tree_name]['index_jet_S'])
                btag_jet = np.array(sample_array[sample][tree_name]['Btag_jet'])

                risr = np.array([entry[:2] for entry in risr])
                ptisr = np.array([entry[:2] for entry in ptisr])
                index_isr_jet = np.array([entry[:2] for entry in index_isr_jet])
                index_s_jet = np.array([entry[:2] for entry in index_s_jet])

                ms = np.array([entry[:2] for entry in ms])
                ps = np.array([entry[:2] for entry in ps])
                pts = np.array([entry[:2] for entry in pts])
                pzs = np.array([entry[:2] for entry in pzs])
                dpcmi = np.array([entry[:2] for entry in dpcmi])
                dpsi = np.array([entry[:2] for entry in dpsi])

                risr = risr[:, 1]
                ptisr = ptisr[:, 1]
                index_isr_jet = index_isr_jet[:, 1]
                index_s_jet = index_s_jet[:, 1]

                ms = ms[:, 1]
                ps = ps[:, 1]
                pts = pts[:, 1]
                pzs = pzs[:, 1]
                dpcmi = dpcmi[:, 1]
                dpsi = dpsi[:, 1]

                if 'SMS-T2-4bd_490' in sample:
                    weight = np.array([(137000 * 0.51848) / 1207007. for w in weight])
                else:
                    weight = 137. * weight
                

                ################        Medium       #####################
                mini_lep = np.array([mini[lid>=2] for mini, lid in zip(mini_lep, id_lep)])

                pt_med_lep = np.array([lep[lid>=2] for lep, lid in zip(pt_lep, id_lep)]) 
                pt_mini_lep = np.array([lep[mini<0.1] for lep, mini in zip(pt_med_lep, mini_lep)])
                eta_med_lep = np.array([lep[lid>=2] for lep, lid in zip(eta_lep, id_lep)]) 
                eta_mini_lep = np.array([lep[mini<0.1] for lep, mini in zip(eta_med_lep, mini_lep)])
                phi_med_lep = np.array([lep[lid>=2] for lep, lid in zip(phi_lep, id_lep)]) 
                phi_mini_lep = np.array([lep[mini<0.1] for lep, mini in zip(phi_med_lep, mini_lep)])
                m_med_lep = np.array([lep[lid>=2] for lep, lid in zip(m_lep, id_lep)]) 
                m_mini_lep = np.array([lep[mini<0.1] for lep, mini in zip(m_med_lep, mini_lep)])

                mini_med_mask = np.array([ True if len(mini[mini<0.1]) == 1 and len(lid) == 1 else False for mini, lid in zip(mini_lep,id_lep)])
                mini_med_ll_mask = np.array([ True if len(mini[mini<0.1]) == 2 else False for mini, lid in zip(mini_lep,id_lep)])

                
                btag_isr = np.array([jet[index] for jet, index in zip(btag_jet, index_isr_jet)])

                sj_ge_1 = np.array([ True if len(jet) >=1 else False for jet in index_s_jet])
                sj_0 = np.array([ True if len(jet) == 1 else False for jet in index_s_jet])

                b_isr_ge_1 = np.array([ True if len(jet[jet>0.4941]) >= 1 else False for jet in btag_isr])
                b_isr_none = np.array([ True if len(jet[jet>0.4941]) == 0 else False for jet in btag_isr])

                risr_0p95 = risr > 0.95
                risr_0p8 = risr > 0.8
                met_200 = met > 200
                ptisr_200 = ptisr > 200
                ms_100 = ms < 100             
                evt_2l_selection_mask = np.all([risr_0p8, met_200, ptisr_200, mini_med_ll_mask], axis=0)
                ############  >= 1 s jet ################ 
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_ge_1], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_ge_1], axis=0)

                ############  == 0 s jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_0], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_0], axis=0)

                ############  >= 1 isr b jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, b_isr_ge_1], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, b_isr_ge_1], axis=0)
 
                ############  == 0 isr b jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, b_isr_none], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, b_isr_none], axis=0)
 
                ############  >= 1 s jet; >= 1 isr b jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_ge_1, b_isr_ge_1], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_ge_1, b_isr_ge_1], axis=0)
 
                ############  >= 1 s jet; == 0 isr b jet ################
                evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_ge_1, b_isr_none], axis=0)
                evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_ge_1, b_isr_none], axis=0)
 
                ############  == 0 s jet; >= 1 isr b jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_0, b_isr_ge_1], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_0, b_isr_ge_1], axis=0)
 
                #############  == 0 s jet; == 0 isr b jet ################
                #evt_selection_mask = np.all([risr_0p95, met_200, ptisr_200, mini_med_mask, sj_0, b_isr_none], axis=0)
                #evt_risr_selection_mask = np.all([met_200, ptisr_200, mini_med_mask, sj_0, b_isr_none], axis=0)
 
                phi_evt_lep = np.array([lep[0] for lep, mask in zip(phi_mini_lep, evt_selection_mask) if mask])
                pt_ll_lep = np.array([lep[:2] for lep, mask in zip(pt_mini_lep, evt_2l_selection_mask) if mask])
                eta_ll_lep = np.array([lep[:2] for lep, mask in zip(eta_mini_lep, evt_2l_selection_mask) if mask])
                phi_ll_lep = np.array([lep[:2] for lep, mask in zip(phi_mini_lep, evt_2l_selection_mask) if mask])
                m_ll_lep = np.array([lep[:2] for lep, mask in zip(m_mini_lep, evt_2l_selection_mask) if mask])

                phi_met = phi_met[evt_selection_mask]
                ll_weight = weight[evt_2l_selection_mask]

                ps = ps[evt_selection_mask]
                pts = pts[evt_selection_mask]
                pzs = pzs[evt_selection_mask]
                dpcmi = dpcmi[evt_selection_mask]
                dpsi = dpsi[evt_selection_mask]
                risr = risr[evt_risr_selection_mask]
                ptisr = ptisr[evt_risr_selection_mask]
                ms_risr = ms[evt_risr_selection_mask]
                weight_risr = weight[evt_risr_selection_mask]
                
                ms = ms[evt_selection_mask]
                weight = weight[evt_selection_mask]
##################  delta phi ###########################
                dphi_l_met = np.abs(phi_evt_lep - phi_met)

                dphi_l_met = np.array([ phi + 2*np.pi if phi < -np.pi else phi for phi in dphi_l_met])
                dphi_l_met = np.array([ phi - 2*np.pi if phi > np.pi else phi for phi in dphi_l_met])
#########################################################
#################### dilepton invariant mass ###################
                if np.any(evt_2l_selection_mask):
                    px_l1 = np.abs(pt_ll_lep[:,0])*np.cos(phi_ll_lep[:,0])
                    py_l1 = np.abs(pt_ll_lep[:,0])*np.sin(phi_ll_lep[:,0])
                    pz_l1 = np.abs(pt_ll_lep[:,0])*np.sinh(eta_ll_lep[:,0])
                    p_l1 = np.array([px_l1, py_l1, pz_l1])
                    e_l1 = np.sqrt(np.sum(np.power(p_l1, 2), axis=0) + np.power(m_ll_lep[:,0], 2))
                    
                    px_l2 = np.abs(pt_ll_lep[:,1])*np.cos(phi_ll_lep[:,1])
                    py_l2 = np.abs(pt_ll_lep[:,1])*np.sin(phi_ll_lep[:,1])
                    pz_l2 = np.abs(pt_ll_lep[:,1])*np.sinh(eta_ll_lep[:,1])
                    p_l2 = np.array([px_l2, py_l2, pz_l2])
                    e_l2 = np.sqrt(np.sum(np.power(p_l2, 2), axis=0) + np.power(m_ll_lep[:,1], 2))
               
                    p_ll = p_l1 + p_l2
                    e_ll = e_l1 + e_l2
                    
                    mag_sq_ll = np.power(e_ll, 2) - np.sum(np.power(p_ll, 2), axis=0)
                    
                    m_ll = np.array([np.sqrt(mag) if mag > 0 else -np.sqrt(-mag) for mag in mag_sq_ll])
##################################################################

                div_pz_ms = np.abs(pzs) / (np.abs(pzs) + ms)
                div_pts_ms = pts / (pts + ms)
                pzs = np.abs(pzs) 
                ps = np.abs(ps) 

                if np.any(evt_risr_selection_mask):
                    rnp.fill_hist(hist[sample][tree_name]['PTISR_MS'], np.swapaxes([ptisr, ms_risr],0,1), weight_risr) 
                    rnp.fill_hist(hist[sample][tree_name]['RISR_MS'], np.swapaxes([risr, ms_risr],0,1), weight_risr) 
                if np.any(evt_2l_selection_mask):
                    rnp.fill_hist(hist[sample][tree_name]['MLL'], m_ll, ll_weight)

                if not np.any(evt_selection_mask):
                    print 'finished filling'
                    continue
#                if '490' in sample:
#                    rnp.fill_hist(hist[sample][tree_name]['dphi_lep_MET'], dphi_l_met) 
#                    rnp.fill_hist(hist[sample][tree_name]['dphiCMI'], dpcmi) 
#                    rnp.fill_hist(hist[sample][tree_name]['dphiSI'], dpsi) 
#                    rnp.fill_hist(hist[sample][tree_name]['MS'], ms) 
#                    rnp.fill_hist(hist[sample][tree_name]['PS'], ps) 
#                    rnp.fill_hist(hist[sample][tree_name]['PTS'], pts) 
#                    rnp.fill_hist(hist[sample][tree_name]['PzS'], pzs) 
#                    rnp.fill_hist(hist[sample][tree_name]['PzS_div_MS'], div_pz_ms) 
#                    rnp.fill_hist(hist[sample][tree_name]['MS_PzS'], np.swapaxes([ms, pzs],0,1)) 
#                else:
                rnp.fill_hist(hist[sample][tree_name]['dphi_lep_MET'], dphi_l_met, weight) 
                rnp.fill_hist(hist[sample][tree_name]['dphiCMI'], dpcmi, weight) 
                rnp.fill_hist(hist[sample][tree_name]['dphiSI'], dpsi, weight) 
                rnp.fill_hist(hist[sample][tree_name]['MS'], ms, weight) 
                rnp.fill_hist(hist[sample][tree_name]['PS'], ps, weight) 
                rnp.fill_hist(hist[sample][tree_name]['PTS'], pts, weight) 
                rnp.fill_hist(hist[sample][tree_name]['PzS'], pzs, weight) 
                rnp.fill_hist(hist[sample][tree_name]['PzS_div_MS'], div_pz_ms, weight) 
                rnp.fill_hist(hist[sample][tree_name]['PTS_div_MS'], div_pts_ms, weight) 
                rnp.fill_hist(hist[sample][tree_name]['MS_PzS'], np.swapaxes([ms, pzs],0,1), weight) 
                rnp.fill_hist(hist[sample][tree_name]['MS_PTS'], np.swapaxes([ms, pts],0,1), weight) 
                rnp.fill_hist(hist[sample][tree_name]['MS_dphiLepMET'], np.swapaxes([ms, dphi_l_met],0,1), weight) 
#                rnp.fill_hist(hist[sample][tree_name]['MET'], met, weight)

                print 'finished filling'
    return hist


             

if __name__ == "__main__":

    signals = { 
#    'SMS-T2-4bd_420' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_',
#],
#    'SMS-T2-4bd_490' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_',
#                    #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_v2',
#],
#    'SMS-T2-4bd_490_lowpt' : [
#                    #'/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_v2',
#],
    'SMS-T2-4bd_490_I' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_I/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'SMS-T2-4bd_490_II' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_II/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'SMS-T2-4bd_490_III' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_III/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'SMS-T2-4bd_490_IV' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_IV/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
#    'SMS-T2-4bd_490_V' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_V/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#],
#    'SMS-T2-4bd_490_VI' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_tmp/Fall17_94X_SMS_VI/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#],
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
    'DY_M50_2017' : [
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_samples_NANO/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
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
    variables = ['MET', 'MET_phi', 'MS', 'PS', 'PTS', 'PzS', 'dphiSI', 'dphiCMI', 'RISR', 'PTISR', 'PT_lep', 'Eta_lep', 'Phi_lep', 'M_lep', 'MiniIso_lep', 'ID_lep', 'MX3a', 'index_jet_ISR', 'index_jet_S', 'Btag_jet', 'MX3b', 'MVa', 'MVb', 'EVa', 'EVb', 'weight']

    start_b = time.time()    
#    background_list = process_the_samples(backgrounds, None, None)
#    hist_background = get_histograms(background_list, variables, None)
#
#    #write_hists_to_file(hist_background, './output_background_mab_hists_sj_ge_1.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_sj_none.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_isrb_ge_1.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_isrb_none.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_sj_ge_1_isrb_ge_1.root') 
#    write_hists_to_file(hist_background, './output_background_mab_hists_sj_ge_1_isrb_none.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_sj_none_isrb_ge_1.root') 
#    #write_hists_to_file(hist_background, './output_background_mab_hists_sj_none_isrb_none.root') 
    stop_b = time.time()

    signal_list = process_the_samples(signals, None, None)
    hist_signal = get_histograms(signal_list, variables, None)

    #write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_ge_1.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_none.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_isrb_ge_1.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_isrb_none.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_ge_1_isrb_ge_1.root') 
    write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_ge_1_isrb_none_looseleps.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_none_isrb_ge_1.root') 
    #write_hists_to_file(hist_signal, './output_signal_mab_hists_sj_none_isrb_none.root') 
    stop_s = time.time()

    print "background: ", stop_b - start_b
    print "signal:     ", stop_s - stop_b
    print "total:      ", stop_s - start_b
 
    print 'finished writing'
