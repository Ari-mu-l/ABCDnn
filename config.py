import os
import numpy as np
from samples import *

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "jmanagan"
#postfix = "hadd"

condorDir = "root://cmseos.fnal.gov//store/user/xshen/"

sourceDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Sep2023_2018/".format( eosUserName ),
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

targetDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/xshen/BtoTW_Sep2023_2018/",
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

sampleDir = {
"2018UL":""
#  year: "FWLJMET106XUL_singleLep{}UL_RunIISummer20_{}_step3/nominal/".format( year, postfix ) for year in [ "2016APV", "2016", "2017", "2018" ]
}

variables = {
  "Bprime_mass": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0., 5000.], # what is this limit? should it include max and min? or does it specify the interesting range?
    "LATEX": "M_{reco}"
  },
  "gcJet_ST": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,6000.], 
    "LATEX": "ST_{gcJet}"
  },
  "NJets_forward": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [0,8],
    "LATEX": "N_{forward}"
  },
  "NJets_DeepFlavL": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [0,8],
    "LATEX": "N_b"
  },
}

selection = { # edit these accordingly
  "Bdecay_obs":{ "VALUE": [ 1, 4 ], "CONDITION": [ "==", "==" ] },
}

regions = {
  "X": {
    "VARIABLE": "NJets_DeepFlavL",
    "INCLUSIVE": True,
    "MIN": 0,
    "MAX": None,
    "SIGNAL": 0
  },
  "Y": {
    "VARIABLE": "NJets_forward",
    "INCLUSIVE": True,
    "MIN": 0,
    "MAX": 1,
    "SIGNAL": 1
  }
}

params = {
  "MODEL": { # parameters for setting up the NAF model
    "NODES_COND": 8,
    "HIDDEN_COND": 1,
    "NODES_TRANS": 8,
    "LRATE": 1e-2,
    "DECAY": 0.1,
    "GAP": 200,
    "DEPTH": 1,
    "REGULARIZER": "NONE", # DROPOUT, BATCHNORM, ALL, NONE
    "INITIALIZER": "RandomNormal", # he_normal, RandomNormal
    "ACTIVATION": "swish", # softplus, relu, swish
    "BETA1": 0.9,
    "BETA2": 0.999,
    "MMD SIGMAS": [0.05,0.1,0.2],
    "MMD WEIGHTS": None,
    "MINIBATCH": 2**6,
    "RETRAIN": True,
    "PERMUTE": False,
    "SEED": 101, # this can be overridden when running train_abcdnn.py
    "SAVEDIR": "./Results/",
    "CLOSURE": 0.03,
    "VERBOSE": True  
  },
  "TRAIN": {
    "EPOCHS": 2000,
    "PATIENCE": 0,
    "MONITOR": 100,
    "MONITOR THRESHOLD": 0,  # only save model past this epoch
    "PERIODIC SAVE": True,   # saves model at each epoch step according to "MONITOR" 
    "SHOWLOSS": True,
    "EARLY STOP": False,      # early stop if validation loss begins diverging
  },
  "PLOT": {
    "RATIO": [ 0.75, 1.25 ], # y limits for the ratio plot
    "YSCALE": "linear",      # which y-scale plots to produce
    "NBINS": 51,             # histogram x-bins
    "ERRORBARS": True,       # include errorbars on hist
    "NORMED": True,          # normalize histogram counts/density
    "SAVE": False,           # save the plots as png
    "PLOT_KS": True,         # include the KS p-value in plots
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

branches = [
  "NJets_forward", "NJets_DeepFlavL",
  "Bdecay_obs",
  "Bprime_mass", "gcJet_ST"
  #"leptonPt_MultiLepCalc",
  #"isElectron", "isMuon", "isTraining",
  #"MT_lepMet",
  #"NresolvedTops1pFake", "NJetsCSV_JetSubCalc", "NJets_JetSubCalc",
  #"NJetsTtagged", "NJetsWtagged",
  #"corr_met_MultiLepCalc",
  #"MT_lepMet",
  #"minDR_lepJet",
  #"AK4HT",
  #"DataPastTriggerX", "MCPastTriggerX"
]

for vName in variables:
  if vName not in branches: branches.append( str( vName ) )

samples_input = {
  "2018UL": {
    "DATA": [
      TTToSemiLeptonic2018UL,
    ], # FIXME later

    "MAJOR MC": [
      Bprime_M1400_2018UL,
      QCDHT10002018UL,
      QCDHT15002018UL,
      #QCDHT20002018UL, # has no events
      QCDHT2002018UL,
      QCDHT3002018UL,
      QCDHT5002018UL,
      QCDHT7002018UL,
      TTToSemiLeptonic2018UL,
      WJetsHT12002018UL,
      WJetsHT2002018UL,
      WJetsHT25002018UL,
      WJetsHT4002018UL,
      WJetsHT6002018UL,
      WJetsHT8002018UL,
    ],
    "MINOR MC": [], # no need to add minor MC for now, because using ttbar MC as data
    "CLOSURE": []
  },
  # add other years here
}

samples_apply = {
  "2018UL": [
    Bprime_M1400_2018UL,
    QCDHT10002018UL,
    QCDHT15002018UL,
    QCDHT20002018UL,
    #QCDHT2002018UL,
    QCDHT3002018UL,
    QCDHT5002018UL,
    QCDHT7002018UL,
    TTToSemiLeptonic2018UL,
    WJetsHT12002018UL,
    WJetsHT2002018UL,
    WJetsHT25002018UL,
    WJetsHT4002018UL,
    WJetsHT6002018UL,
    WJetsHT8002018UL,
  ],
}
