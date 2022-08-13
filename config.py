import os
import numpy as np

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "dali"
postfix = "3t"
sourceDir = {
  "CONDOR": "root://cmseos.fnal.gov//store/user/{}/".format( eosUserName ),
  "LPC": "root://cmsxrootd.fnal.gov//store/user/{}/".format( eosUserName ),
  "BRUX": "root://brux30.hep.brown.edu:1094//isilon/hadoop/store/user/dali/"
}
sampleDir = {
  year: "FWLJMET106XUL_singleLep{}UL_RunIISummer20_{}_step3/nominal/".format( year, postfix ) for year in [ "2016APV", "2016", "2017", "2018" ]
}

variables = {
  "AK4HT": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,3000.],
    "LATEX": "H_T\ \mathrm{(GeV)}"
  },
  "DNN_1to40_3t": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,1.],
    "LATEX": "DNN"
  },
  "NJets_JetSubCalc": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [4,6],
    "LATEX": "N_j"
  },
  "NJetsCSV_JetSubCalc": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [2,3],
    "LATEX": "N_b"
  }
}

selection = { # edit these accordingly
  "NJets_JetSubCalc": { "VALUE": [ 4 ], "CONDITION": [ ">=" ] },
  "NJetsCSV_JetSubCalc": { "VALUE": [ 2 ], "CONDITION": [ ">=" ] },
  "corr_met_MultiLepCalc": { "VALUE": [ 20. ], "CONDITION": [ ">" ] },
  "MT_lepMet": { "VALUE": [ 0 ], "CONDITION": [ ">" ] },
  "minDR_lepJet": { "VALUE": [ 0.4 ], "CONDITION": [ ">" ] },
  "AK4HT": { "VALUE": [ 350. ], "CONDITION": [ ">" ] },
  "DataPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
  "MCPastTriggerX": { "VALUE": [ 1 ], "CONDITION": [ "==" ] },
  #"isTraining": { "VALUE": [ "1" ], "CONDITION": [ "==" ] },
}

regions = {
  "X": {
    "VARIABLE": "NJets_JetSubCalc",
    "INCLUSIVE": True,
    "MIN": 4,
    "MAX": 6,
    "SIGNAL": 6
  },
  "Y": {
    "VARIABLE": "NJetsCSV_JetSubCalc",
    "INCLUSIVE": True,
    "MIN": 2,
    "MAX": 3,
    "SIGNAL": 3
  }
}

params = {
  "EVENTS": {
    "MCWEIGHT": None
  },
  "MODEL": { # parameters for setting up the NAF model
    "NODES_COND": 64,
    "HIDDEN_COND": 1,
    "NODES_TRANS": 16,
    "LRATE": 1e-4,
    "DECAY": 1e-1,
    "GAP": 1000,
    "DEPTH": 1,
    "REGULARIZER": "DROPOUT", # DROPOUT, NONE
    "ACTIVATION": "swish",
    "BETA1": 0.9,
    "BETA2": 0.999,
    "MMD SIGMAS": [1.0],
    "MMD WEIGHTS": None,
    "MINIBATCH": 2**11,
    "RETRAIN": True,
    "SEED": 101, # this can be overridden when running train_abcdnn.py
    "SAVEDIR": "./Results/",
    "VERBOSE": False   
  },
  "TRAIN": {
    "EPOCHS": 10000,
    "PATIENCE": 10000,
    "MONITOR": 100,
    "LOSS THRESHOLD": 1e-2,  # only save model if the loss is beneath this value
    "PERIODIC SAVE": True,  # saves model at each epoch step according to "MONITOR" 
    "SHOWLOSS": True,
    "EARLY STOP": True,    # early stop if validation loss begins diverging
  },
  "PLOT": {
    "RATIO": [ 0.25, 2.0 ], # y limits for the ratio plot
    "YSCALES": [ "log" ],   # which y-scale plots to produce
    "NBINS": 20,            # histogram x-bins
    "ERRORBARS": True,      # include errorbars on hist
    "NORMED": True,         # normalize histogram counts/density
    "SAVE": False,          # save the plots as png
    "PLOT_KS": True,        # include the KS p-value in plots
  }
}
        
hyper = {
  "OPTIMIZE": {
    "NODES_COND": ( [8,16,32,64,128], "CAT" ),
    "HIDDEN_COND": ( [1,4], "INT" ),
    "NODES_TRANS": ( [1,8,16,32,64,128], "CAT" ),
    "LRATE": ( [1e-5,1e-4,1e-3], "CAT" ),
    "DECAY": ( [1,1e-1,1e-2], "CAT" ),
    "GAP": ( [100,500,1000,5000], "CAT" ),
    "DEPTH": ( [1,4], "INT" ),
    "REGULARIZER": ( ["L1","L2","L1+L2","None"], "CAT" ),
    "ACTIVATION": ( ["swish","relu","elu","softplus"], "CAT" ),
    "BETA1": ( [0.90,0.99,0.999], "CAT" ),
    "BETA2": ( [0.90,0.99,0.999], "CAT" ),
    "SIGMA": ( [0.05,0.35], "FLOAT" )
  },
  "PARAMS": {
    "PATIENCE": 10000,
    "EPOCHS": 2000,
    "N_RANDOM": 20,
    "N_CALLS": 30,
    "MINIBATCH": 2**12,
    "VERBOSE": True
  }
}

samples_input = {
  "2016": {
    "DATA": [
"SingleElectron_hadd.root",
"SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  },
  "2017": {
    "DATA": [
      "SingleElectron_hadd.root",
      "SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  },
  "2018": {
    "DATA": [
"EGamma_hadd.root",
"SingleMuon_hadd.root",
    ],
    "MC": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
    ]
  }
}

samples_apply = {
  "2016": [
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
#"QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
#"QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_hadd.root",
  ],
  "2017": [
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_3_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_4_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_5_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_6_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_7_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_8_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_9_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_10_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
#"QCD_HT200to300_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT500to700_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8_hadd.root",
#"QCD_HT2000toInf_TuneCP5_13TeV-madgraph-pythia8_hadd.root"
  ],
  "2018": [
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTToHadronic_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt1b_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_tt2b_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttbb_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttcc_hadd.root",
#"TTToSemiLepton_HT500Njet9_TuneCP5_13TeV-powheg-pythia8_ttjj_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt1b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_tt2b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttbb_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttcc_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_1_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT0Njet0_ttjj_2_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt1b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_tt2b_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttbb_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttcc_hadd.root",
#"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_HT500Njet9_ttjj_hadd.root",
#"QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root",
"QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8_hadd.root"
  ]
}
