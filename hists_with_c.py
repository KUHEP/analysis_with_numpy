import sys
import os
import subprocess
from file_table_functions import *

def run_c_code(file_list_, out_file_name_):
    for isam, sample in enumerate(file_list_):
        for itr, tree_name in enumerate(file_list_[sample]['trees']):
            if isam == 0 and itr == 0:
                out_file_opt = "RECREATE"
            else:
                out_file_opt = "UPDATE"
          
            # cmd = "./src/compiledthreads 8 "+tree_name+" "+sample+"__"+tree_name+"__"+" "+out_file_name_+" "+out_file_opt+" "
            cmd = ['./src/compiledthreads', '8', tree_name, sample+'__'+tree_name+'__', out_file_name_, out_file_opt]
            for in_file in file_list_[sample]['files']:
                # cmd += "\""+ in_file + "\" "
                cmd.append(in_file)
            print cmd
            return_value = subprocess.call(cmd, stdout=open("/dev/null","w"))
            print return_value
            print 'done'


if __name__ == "__main__":
    signals = {
#    'SMS-T2bW_dM' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS_Stop/SMS-T2bW_X05_dM-10to80_genHT-160_genMET-80_mWMin-0p1_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X',
#], 
#    'SMS-T2bW' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2bW_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-TChiWH' : [ 
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-TChiWH_WToLNu_HToBB_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2cc' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2cc_genHT-160_genMET-80_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#], 
#    'SMS-T2cc_175_95' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2cc_genMET-80_mStop-175_mLSP-95_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#], 
#    'SMS-T2bb' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2bb_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#], 
#    'SMS-T2tt_dM' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_dM-10to80_2Lfilter_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#], 
#    'SMS-T2tt_150to250' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-150to250_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_1200_100' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-1200_mLSP-100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_650_350' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-650_mLSP-350_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_850_100' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-850_mLSP-100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_250to350' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-250to350_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_350to400' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-350to400_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
#    'SMS-T2tt_400to1200' : [
#                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS/SMS-T2tt_mStop-400to1200_TuneCP2_13TeV-madgraphMLM-pythia8_Fall17_94X.root',
#],
    'SMS-T2-4bd_420' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS_Stop/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_SMS_I/SMS-T2-4bd_genMET-80_mStop-500_mLSP-420_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'SMS-T2-4bd_490' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_SMS_Stop/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_SMS_I/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
#    'SMS-T2-4bd_490_IV' : [
#                    '/home/t3-ku/erichjs/work/Ewkinos/reducer/CMSSW_10_1_4_patch1/src/KUEWKinoAnalysis_dev_v2/output_10Sep19/Fall17_94X_SMS_IV/SMS-T2-4bd_genMET-80_mStop-500_mLSP-490_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
#],
              }
    backgrounds = {
    'TTJets_2017' : [
#                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X'
#                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X'
                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkgextra/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_Fall17_94X',
                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkgextra/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkgextra/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                     '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkgextra/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'WJets_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'ZJets_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-100To200_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-200To400_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-400To600_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-600To800_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-800To1200_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-1200To2500_13TeV-madgraph_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZJetsToNuNu_HT-2500ToInf_13TeV-madgraph_Fall17_94X',
],
    'DY_M50_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-70to100_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Fall17_94X',
],
    'WW_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWTo4Q_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWToLNuQQ_NNPDF31_TuneCP5_13TeV-powheg-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
],
    'ZZ_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZZTo2L2Nu_13TeV_powheg_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZZTo2Q2Nu_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZZTo4L_13TeV_powheg_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
],
    'WZ_2017' : [
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZG_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZTo1L3Nu_13TeV_amcatnloFXFX_madspin_pythia8_v2_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZZ_TuneCP5_13TeV-amcatnlo-pythia8_Fall17_94X',
                    '/home/t3-ku/crogan/NTUPLES/NANO/NoHadd/Fall17_94X_bkg/WZ_TuneCP5_13TeV-pythia8_Fall17_94X',
],
                  }

    #background_list = process_the_samples(backgrounds, None, None)
    #out_file = "hists_background_mass_s_sep_leps_risr_0p8_0_isrb_ge_1sv_15Oct19.root"
    #run_c_code(background_list, out_file)        

    #ordered_list = get_ordered_list_of_masses(signal_list)
    #make_table_of_masses(ordered_list)

    signal_list = process_the_samples(signals, None, None)
    out_file = "hists_signal_mass_s_sep_leps_risr_0p8_isrb_ge_1sv_15Oct19.root"
    run_c_code(signal_list, out_file)

