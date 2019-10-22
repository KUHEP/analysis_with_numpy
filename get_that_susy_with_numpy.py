#!/usr/bin/env python

"""
New thing to try out, doing analysis with numpy, converting back to ROOT to do histogramming
Creator: Erich Schmitz
Date: Feb 22, 2019
"""

import ROOT as rt
import numpy as np
import root_numpy as rnp
import uproot as ur
import numpy.lib.recfunctions as rfc
import os
from collections import OrderedDict

rt.gROOT.SetBatch()
rt.TH1.AddDirectory(rt.kFALSE)

def get_tree_info_singular(file_name_, tree_name_, variable_list_, cuts_to_apply_=None):
    """
    Same as get_tree_info_plural, but runs over a single file
    returns structured array containing the list of variables
    """

    tmp_f = rt.TFile(file_name_, 'r')
    tmp_t = tmp_f.Get(tree_name_)
    tmp_array = rnp.tree2array(tmp_t, branches=variable_list_, selection=cuts_to_apply_)
    
    return tmp_array

            
def get_tree_info_plural(file_list_, tree_list_, variable_list_, cuts_to_apply_=None):
    """
    Get the variables listed as variable list, turn into numpy arrays, used over a list of multiple files
    """
    event_dict = OrderedDict()
    

    for tree_name in tree_list_:
        event_dict[tree_name] = []
    n_files = len(file_list_)

    for ifile, file_name in enumerate(file_list_):
        print 'reading file:', ifile+1, '/', n_files

        tmp_f = rt.TFile(file_name, 'r')
        print file_name
        for tree_name in tree_list_:
            tmp_t = tmp_f.Get(tree_name)
            if bool(tmp_t): 
                tmp_array = rnp.tree2array(tmp_t, branches=variable_list_, selection=cuts_to_apply_)
            else:
                print 'tree: ' + tree_name + ' was not saved correctly, please check'

            event_dict[tree_name].append(tmp_array)
    print 'done reading'
    print 'restructuring arrays...'

    
    for tree in event_dict:
        event_dict[tree] = np.concatenate([struc for struc in event_dict[tree]])

    print 'done restructuring'
    return event_dict 


def reduce_and_condense(file_list_of_file_lists, variable_list):
    print 'for posterity'

    tree_chain = rt.TChain('deepntuplizer/tree')
    for file_list in file_list_of_file_lists:
        files = open(file_list).readlines()
        files = [f.replace('\n','') for f in files]
        for file_name in files:
            tree_chain.Add(file_name)
        branches = [b.GetName() for b in tree_chain.GetListOfBranches()]
    for branch in branches:
        if branch not in variable_list:
            tree_chain.SetBranchStatus(branch, 0)
    file_out = rt.TFile('output_condensed.root', 'recreate')
    reduced_tree = tree_chain.CloneTree()
    reduced_tree.Write()
    file_out.Close()


def reduce_singular(in_file_name_, out_file_name_, variable_list_):
    tmp_f = rt.TFile(int_file_name_, 'r')
    tmp_t = tmp_f.Get('deepntuplizer/tree')

    branches = [b.GetName() for b in tmp_t.GetListOfBranches()]        
    for branch in branches:
        if branch not in variable_list:
            tmp_t.SetBranchStatus(branch, 0)

    out_file = rt.TFile(out_file_name_, 'recreate')
    reduced_tree = tmp_t.CloneTree()
    reduced_tree.Write()
    file_out.Close()
 
       
def do_SMS_splitting(input_array_):
    mom = 0.
    child = 0.

    func_part = np.vectorize(lambda x: 1000000 < abs(x) < 3000000)

    print 'to be continued...'


def process_the_samples(input_sample_list_, variable_list_, cut_list_, truncate_file_ = None):
    array_list = OrderedDict()

    for sample, folder in input_sample_list_.items():
        print sample, folder
        file_list = [os.path.join(folder, f) for f in os.listdir(folder) if (os.path.isfile(os.path.join(folder, f)) and ('.root' in f))]
    
        # Get file structure, in case there is a grid of mass points
        f_struct_tmp = rt.TFile(file_list[0], 'r')
        tree_list = [tree.GetName() for tree in f_struct_tmp.GetListOfKeys()]

        if truncate_file_ is not None:
            file_list = file_list[:truncate_file_]

        if 'SMS' in sample:
            tree_name_mass = [(int(mass.split('_')[0]), int(mass.split('_')[1])) for mass in tree_list]
            tree_name_mass.sort(key=lambda x: int(x[0]))
            tree_list = [str(mom) + '_' + str(child) for mom, child in tree_name_mass]
 
        if variable_list_ is None:
            variable_list_ = [branch.GetName() for branch in f_struct_tmp.Get(tree_list[0]).GetListOfBranches()]
        f_struct_tmp.Close()
        variable_list_ = list(OrderedDict.fromkeys(variable_list_))
        array_list[sample] = get_tree_info_plural(file_list, tree_list, variable_list_) 

    return array_list


def get_histograms(sample_array):
    hist = OrderedDict()
    for sample in sample_array:
        hist[sample] = OrderedDict()
        for tree_name in sample_array[sample]:
            print '\nGetting Histograms for:', sample, tree_name 
            hist[sample][tree_name] = OrderedDict()
  
            # Reserve histograms
            hist[sample][tree_name]['PT_jet'] = rt.TH1D('jetpt_'+sample+'_'+tree_name, 'jet p_{T} [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['Eta_jet'] = rt.TH1D('jeteta_'+sample+'_'+tree_name, 'jet #eta', 100, -5, 5)
            hist[sample][tree_name]['M_jet'] = rt.TH1D('jetm_'+sample+'_'+tree_name, 'jet M [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['loose_bjet_pt'] = rt.TH1D('loose_bjetpt_'+sample+'_'+tree_name, 'loose_bjet p_{T} [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['loose_bjet_eta'] = rt.TH1D('loose_bjeteta_'+sample+'_'+tree_name, 'loose_bjet #eta', 100, -5, 5)
            hist[sample][tree_name]['loose_bjet_m'] = rt.TH1D('loose_bjetm_'+sample+'_'+tree_name, 'loose_bjet M [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['medium_bjet_pt'] = rt.TH1D('medium_bjetpt_'+sample+'_'+tree_name, 'medium_bjet p_{T} [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['medium_bjet_eta'] = rt.TH1D('medium_bjeteta_'+sample+'_'+tree_name, 'medium_bjet #eta', 100, -5, 5)
            hist[sample][tree_name]['medium_bjet_m'] = rt.TH1D('medium_bjetm_'+sample+'_'+tree_name, 'medium_bjet M [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['tight_bjet_pt'] = rt.TH1D('tight_bjetpt_'+sample+'_'+tree_name, 'tight_bjet p_{T} [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['tight_bjet_eta'] = rt.TH1D('tight_bjeteta_'+sample+'_'+tree_name, 'tight_bjet #eta', 100, -5, 5)
            hist[sample][tree_name]['tight_bjet_m'] = rt.TH1D('tight_bjetm_'+sample+'_'+tree_name, 'tight_bjet M [GeV]', 100, 0, 1000)
            hist[sample][tree_name]['RISR'] = rt.TH1D('loose_RISR_'+sample+'_'+tree_name, 'loose bjet RISR_comb', 100, 0, 2)
            hist[sample][tree_name]['loose_RISR'] = rt.TH1D('loose_RISR_'+sample+'_'+tree_name, 'loose bjet RISR_comb', 100, 0, 2)
            hist[sample][tree_name]['medium_RISR'] = rt.TH1D('medium_RISR_'+sample+'_'+tree_name, 'medium bjet RISR_comb', 100, 0, 2)
            hist[sample][tree_name]['tight_RISR'] = rt.TH1D('tight_RISR_'+sample+'_'+tree_name, 'tight bjet RISR_comb', 100, 0, 2)
            hist[sample][tree_name]['PTISR'] = rt.TH1D('loose_PTISR_'+sample+'_'+tree_name, 'loose bjet PTISR_comb', 100, 0, 1000)
            hist[sample][tree_name]['loose_PTISR'] = rt.TH1D('loose_PTISR_'+sample+'_'+tree_name, 'loose bjet PTISR_comb', 100, 0, 1000)
            hist[sample][tree_name]['medium_PTISR'] = rt.TH1D('medium_PTISR_'+sample+'_'+tree_name, 'medium bjet PTISR_comb', 100, 0, 1000)
            hist[sample][tree_name]['tight_PTISR'] = rt.TH1D('tight_PTISR_'+sample+'_'+tree_name, 'tight bjet PTISR_comb', 100, 0, 1000)
            hist[sample][tree_name]['PTCM'] = rt.TH1D('loose_PTCM_'+sample+'_'+tree_name, 'loose bjet PTCM_comb', 100, 0, 1000)
            hist[sample][tree_name]['loose_PTCM'] = rt.TH1D('loose_PTCM_'+sample+'_'+tree_name, 'loose bjet PTCM_comb', 100, 0, 1000)
            hist[sample][tree_name]['medium_PTCM'] = rt.TH1D('medium_PTCM_'+sample+'_'+tree_name, 'medium bjet PTCM_comb', 100, 0, 1000)
            hist[sample][tree_name]['tight_PTCM'] = rt.TH1D('tight_PTCM_'+sample+'_'+tree_name, 'tight bjet PTCM_comb', 100, 0, 1000)

            hist[sample][tree_name]['RISR_jetpt'] = rt.TH2D('loose_RISR_jetpt_'+sample+'_'+tree_name, 'loose bjet RISR_jetpt_comb', 100, 0, 1000, 100, 0, 2)
            hist[sample][tree_name]['loose_RISR_jetpt'] = rt.TH2D('loose_RISR_jetpt_'+sample+'_'+tree_name, 'loose bjet RISR_jetpt_comb',100, 0, 1000, 100, 0, 2)
            hist[sample][tree_name]['medium_RISR_jetpt'] = rt.TH2D('medium_RISR_jetpt_'+sample+'_'+tree_name, 'medium bjet RISR_jetpt_comb',100, 0, 1000, 100, 0, 2)
            hist[sample][tree_name]['tight_RISR_jetpt'] = rt.TH2D('tight_RISR_jetpt_'+sample+'_'+tree_name, 'tight bjet RISR_jetpt_comb',100, 0, 1000, 100, 0, 2)
    
            hist[sample][tree_name]['PTISR_jetpt'] = rt.TH2D('loose_PTISR_jetpt_'+sample+'_'+tree_name, 'loose bjet PTISR_jetpt_comb', 100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['loose_PTISR_jetpt'] = rt.TH2D('loose_PTISR_jetpt_'+sample+'_'+tree_name, 'loose bjet PTISR_jetpt_comb',100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['medium_PTISR_jetpt'] = rt.TH2D('medium_PTISR_jetpt_'+sample+'_'+tree_name, 'medium bjet PTISR_jetpt_comb',100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['tight_PTISR_jetpt'] = rt.TH2D('tight_PTISR_jetpt_'+sample+'_'+tree_name, 'tight bjet PTISR_jetpt_comb',100, 0, 1000, 100, 0, 1000)
    
            hist[sample][tree_name]['PTCM_jetpt'] = rt.TH2D('loose_PTCM_jetpt_'+sample+'_'+tree_name, 'loose bjet PTCM_jetpt_comb', 100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['loose_PTCM_jetpt'] = rt.TH2D('loose_PTCM_jetpt_'+sample+'_'+tree_name, 'loose bjet PTCM_jetpt_comb',100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['medium_PTCM_jetpt'] = rt.TH2D('medium_PTCM_jetpt_'+sample+'_'+tree_name, 'medium bjet PTCM_jetpt_comb',100, 0, 1000, 100, 0, 1000)
            hist[sample][tree_name]['tight_PTCM_jetpt'] = rt.TH2D('tight_PTCM_jetpt_'+sample+'_'+tree_name, 'tight bjet PTCM_jetpt_comb',100, 0, 1000, 100, 0, 1000)
    
            jet_pt = sample_array[sample][tree_name]['PT_jet'] 
            jet_eta = sample_array[sample][tree_name]['Eta_jet']
            jet_m = sample_array[sample][tree_name]['M_jet']
            bjet_tag = sample_array[sample][tree_name]['Btag_jet']
            risr = sample_array[sample][tree_name]['RISR']
            ptisr = sample_array[sample][tree_name]['PTISR']
            ptcm = sample_array[sample][tree_name]['PTCM']
            weight = sample_array[sample][tree_name]['weight']

            # create arrays with combinatoric assignation for event observables
            risr_comb = []
            ptisr_comb = []
            ptcm_comb = []
            for iev, event in enumerate(risr):
                # create arrays with combinatoric assignation for event observables
                risr_comb.append(np.array(risr[iev][2]))
                ptisr_comb.append(np.array(ptisr[iev][2]))
                ptcm_comb.append(np.array(ptcm[iev][2]))

            loose_mask = []
            medium_mask = []
            tight_mask = []
            test_loose = [1,2,3]
            test_medium = [2,3]
            test_tight = [3]

            risr_comb_jets = []
            ptisr_comb_jets = []
            ptcm_comb_jets = []
            weight_jets = []
            print '\ncreating btag masks'
            for iev, event in enumerate(bjet_tag):
                # create b-tag masks
                loose_mask.append(np.isin(event, test_loose))
                medium_mask.append(np.isin(event, test_medium))
                tight_mask.append(np.isin(event, test_tight))

                jet_length = len(event)
                risr_comb_jets.append(np.array([np.float64(risr_comb[iev])] * jet_length))
                ptisr_comb_jets.append(np.array([np.float64(ptisr_comb[iev])] * jet_length))
                ptcm_comb_jets.append(np.array([np.float64(ptcm_comb[iev])] * jet_length))
                weight_jets.append(np.array([np.float64(weight[iev])] * jet_length))

            print 'finished mask creation'

            bjet_loose_pt = []
            bjet_medium_pt = []
            bjet_tight_pt = []

            bjet_loose_eta = []
            bjet_medium_eta = []
            bjet_tight_eta = []
                
            bjet_loose_m = []
            bjet_medium_m = []
            bjet_tight_m = []

            bjet_loose_risr = []
            bjet_medium_risr = []
            bjet_tight_risr = []

            bjet_loose_ptisr = []
            bjet_medium_ptisr = []
            bjet_tight_ptisr = []

            bjet_loose_ptcm = []
            bjet_medium_ptcm = []
            bjet_tight_ptcm = []

            bjet_loose_risr_pt = []
            bjet_medium_risr_pt = []
            bjet_tight_risr_pt = []

            bjet_loose_ptisr_pt = []
            bjet_medium_ptisr_pt = []
            bjet_tight_ptisr_pt = []

            bjet_loose_ptcm_pt = []
            bjet_medium_ptcm_pt = []
            bjet_tight_ptcm_pt = []

            bjet_loose_weight = []
            bjet_medium_weight = []
            bjet_tight_weight = []

            loose_weight = []
            medium_weight = []
            tight_weight = []
            print '\napplying masks'
            for ievent in xrange(len(jet_pt)):
                bjet_loose_pt.append(jet_pt[ievent][loose_mask[ievent]])
                bjet_medium_pt.append(jet_pt[ievent][medium_mask[ievent]])
                bjet_tight_pt.append(jet_pt[ievent][tight_mask[ievent]])

                bjet_loose_eta.append(jet_eta[ievent][loose_mask[ievent]])
                bjet_medium_eta.append(jet_eta[ievent][medium_mask[ievent]])
                bjet_tight_eta.append(jet_eta[ievent][tight_mask[ievent]])

                bjet_loose_m.append(jet_m[ievent][loose_mask[ievent]])
                bjet_medium_m.append(jet_m[ievent][medium_mask[ievent]])
                bjet_tight_m.append(jet_m[ievent][tight_mask[ievent]])

                bjet_loose_risr_pt.append(risr_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_risr_pt.append(risr_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_risr_pt.append(risr_comb_jets[ievent][tight_mask[ievent]])

                bjet_loose_ptisr_pt.append(ptisr_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_ptisr_pt.append(ptisr_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_ptisr_pt.append(ptisr_comb_jets[ievent][tight_mask[ievent]])

                bjet_loose_ptcm_pt.append(ptcm_comb_jets[ievent][loose_mask[ievent]])
                bjet_medium_ptcm_pt.append(ptcm_comb_jets[ievent][medium_mask[ievent]])
                bjet_tight_ptcm_pt.append(ptcm_comb_jets[ievent][tight_mask[ievent]])

                is_loose = np.any(loose_mask[ievent])
                is_medium = np.any(medium_mask[ievent])
                is_tight = np.any(tight_mask[ievent])
                
                bjet_loose_risr.append(risr_comb[ievent][is_loose])
                bjet_medium_risr.append(risr_comb[ievent][is_medium])
                bjet_tight_risr.append(risr_comb[ievent][is_tight])
                   
                bjet_loose_ptisr.append(ptisr_comb[ievent][is_loose])
                bjet_medium_ptisr.append(ptisr_comb[ievent][is_medium])
                bjet_tight_ptisr.append(ptisr_comb[ievent][is_tight])

                bjet_loose_ptcm.append(ptcm_comb[ievent][is_loose])
                bjet_medium_ptcm.append(ptcm_comb[ievent][is_medium])
                bjet_tight_ptcm.append(ptcm_comb[ievent][is_tight])

                loose_weight.append(weight[ievent][is_loose])
                medium_weight.append(weight[ievent][is_medium])
                tight_weight.append(weight[ievent][is_tight])

                bjet_loose_weight.append(weight_jets[ievent][loose_mask[ievent]])
                bjet_medium_weight.append(weight_jets[ievent][medium_mask[ievent]])
                bjet_tight_weight.append(weight_jets[ievent][tight_mask[ievent]])

            print 'done applying mask'
            print '\nfilling histograms'

            bjet_loose_weight = np.concatenate(bjet_loose_weight)
            bjet_medium_weight = np.concatenate(bjet_medium_weight)
            bjet_tight_weight = np.concatenate(bjet_tight_weight)


            loose_weight = filter(None, loose_weight)
            medium_weight = filter(None, medium_weight)
            tight_weight = filter(None, tight_weight)

            loose_weight = np.concatenate(loose_weight)
            medium_weight = np.concatenate(medium_weight)
            tight_weight = np.concatenate(tight_weight)
            weight_jets = np.concatenate(weight_jets)

            rnp.fill_hist(hist[sample][tree_name]['PT_jet'], np.concatenate(jet_pt), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['Eta_jet'], np.concatenate(jet_eta), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['M_jet'], np.concatenate(jet_m), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['loose_bjet_pt'], np.concatenate(bjet_loose_pt), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_bjet_eta'],np.concatenate(bjet_loose_eta), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_bjet_m'] ,np.concatenate(bjet_loose_m), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_bjet_pt'],np.concatenate(bjet_medium_pt), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_bjet_eta'],np.concatenate(bjet_medium_eta), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_bjet_m'] ,np.concatenate(bjet_medium_m), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_bjet_pt'] ,np.concatenate(bjet_tight_pt), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_bjet_eta'],np.concatenate(bjet_tight_eta), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_bjet_m'],np.concatenate(bjet_tight_m), bjet_tight_weight)

            rnp.fill_hist(hist[sample][tree_name]['RISR'], risr_comb, weight)
            rnp.fill_hist(hist[sample][tree_name]['PTISR'], ptisr_comb, weight)
            rnp.fill_hist(hist[sample][tree_name]['PTCM'], ptcm_comb, weight)

            rnp.fill_hist(hist[sample][tree_name]['RISR_jetpt'], np.swapaxes([np.concatenate(jet_pt), np.concatenate(risr_comb_jets)],0,1), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['PTISR_jetpt'], np.swapaxes([np.concatenate(jet_pt), np.concatenate(ptisr_comb_jets)],0,1), weight_jets)
            rnp.fill_hist(hist[sample][tree_name]['PTCM_jetpt'], np.swapaxes([np.concatenate(jet_pt), np.concatenate(ptcm_comb_jets)],0,1), weight_jets)

            bjet_loose_risr = filter(None, bjet_loose_risr)
            bjet_loose_ptisr = filter(None, bjet_loose_ptisr)
            bjet_loose_ptcm = filter(None, bjet_loose_ptcm)
            bjet_loose_pt = np.concatenate(bjet_loose_pt)
 
            #rnp.fill_hist(hist[sample][tree_name]['loose_RISR'], np.concatenate(bjet_loose_risr), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_RISR'], np.concatenate(bjet_loose_risr))
            rnp.fill_hist(hist[sample][tree_name]['loose_PTISR'], np.concatenate(bjet_loose_ptisr), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTCM'], np.concatenate(bjet_loose_ptcm), loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_RISR_jetpt'], np.swapaxes([bjet_loose_pt, np.concatenate(bjet_loose_risr_pt)],0,1), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTISR_jetpt'], np.swapaxes([bjet_loose_pt, np.concatenate(bjet_loose_ptisr_pt)],0,1), bjet_loose_weight)
            rnp.fill_hist(hist[sample][tree_name]['loose_PTCM_jetpt'], np.swapaxes([bjet_loose_pt, np.concatenate(bjet_loose_ptcm_pt)],0,1), bjet_loose_weight)


            bjet_medium_risr = filter(None, bjet_medium_risr)
            bjet_medium_ptisr = filter(None, bjet_medium_ptisr)
            bjet_medium_ptcm = filter(None, bjet_medium_ptcm)
            bjet_medium_pt = np.concatenate(bjet_medium_pt)
            #rnp.fill_hist(hist[sample][tree_name]['medium_RISR'], np.concatenate(bjet_medium_risr), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_RISR'], np.concatenate(bjet_medium_risr))
            rnp.fill_hist(hist[sample][tree_name]['medium_PTISR'], np.concatenate(bjet_medium_ptisr), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTCM'], np.concatenate(bjet_medium_ptcm), medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_RISR_jetpt'], np.swapaxes([bjet_medium_pt, np.concatenate(bjet_medium_risr_pt)],0,1), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTISR_jetpt'], np.swapaxes([bjet_medium_pt, np.concatenate(bjet_medium_ptisr_pt)],0,1), bjet_medium_weight)
            rnp.fill_hist(hist[sample][tree_name]['medium_PTCM_jetpt'], np.swapaxes([bjet_medium_pt, np.concatenate(bjet_medium_ptcm_pt)],0,1), bjet_medium_weight)


            bjet_tight_risr = filter(None, bjet_tight_risr)
            bjet_tight_ptisr = filter(None, bjet_tight_ptisr)
            bjet_tight_ptcm = filter(None, bjet_tight_ptcm)
            bjet_tight_pt = np.concatenate(bjet_tight_pt)
            #rnp.fill_hist(hist[sample][tree_name]['tight_RISR'], np.concatenate(bjet_tight_risr), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_RISR'], np.concatenate(bjet_tight_risr))
            rnp.fill_hist(hist[sample][tree_name]['tight_PTISR'], np.concatenate(bjet_tight_ptisr), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTCM'], np.concatenate(bjet_tight_ptcm), tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_RISR_jetpt'], np.swapaxes([bjet_tight_pt, np.concatenate(bjet_tight_risr_pt)],0,1), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTISR_jetpt'], np.swapaxes([bjet_tight_pt, np.concatenate(bjet_tight_ptisr_pt)],0,1), bjet_tight_weight)
            rnp.fill_hist(hist[sample][tree_name]['tight_PTCM_jetpt'], np.swapaxes([bjet_tight_pt, np.concatenate(bjet_tight_ptcm_pt)],0,1), bjet_tight_weight)

            print 'finished filling'
    return hist


             

if __name__ == "__main__":
    signals = { 
    'SMS-T2bW' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis/output_samples/SMS-T2bW_v3/root/SMS-T2bW_TuneCUETP8M1_13TeV-madgraphMLM-pythia8'
              }
    backgrounds = {
    'TTJets' : '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev/output_samples/ttbar_2016/root/TTJets_TuneCUETP8M2T4_13TeV-amcatnloFXFX-pythia8'
                  }
    variables = ['PT_jet', 'Eta_jet', 'Phi_jet', 'M_jet', 'Btag_jet', 'RISR', 'PTISR', 'PTCM', 'weight']
    

    # signal_array = process_the_samples(signals, variables, None)
    background_array = process_the_samples(backgrounds, variables, None, 5)

    # hist_signal = get_histograms(signal_array)
    hist_background = get_histograms(background_array)

    out_file = rt.TFile.Open("./output_histos.root", "recreate")
    print 'writing histograms to: ', out_file.GetName()
    out_file.cd()
    #for sample in hist_signal:
    #    sample_dir = out_file.mkdir(sample)
    #    for tree in hist_signal[sample]:
    #        sample_dir.cd()
    #        tmp_dir = sample_dir.mkdir(tree)
    #        tmp_dir.cd()
    #        for hist in hist_signal[sample][tree].values():
    #            hist.Write()

    for sample in hist_background:
        out_file.cd()
        sample_dir = out_file.mkdir(sample)
        sample_dir.cd()
        for tree in hist_background[sample]:
            for hist in hist_background[sample][tree].values():
                hist.Write()

    out_file.Close()
    print 'finished writing'
