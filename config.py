import os
import numpy as np
from samples import *

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "jmanagan"
#postfix = "hadd"

condorDir = "root://cmseos.fnal.gov//store/user/xshen/"

sourceDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Sep2023_fullRun2/".format( eosUserName ),
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

targetDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Sep2023_fullRun2/".format( eosUserName ),
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

sampleDir = {
"2018":""
#  year: "FWLJMET106XUL_singleLep{}UL_RunIISummer20_{}_step3/nominal/".format( year, postfix ) for year in [ "2016APV", "2016", "2017", "2018" ]
}

variables = {
  "Bprime_mass": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0., 5000.], # may be set to None?
    "LATEX": "M_{reco}"
  },
  "gcJet_ST": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,6000.], 
    "LATEX": "ST_{gcJet}"
  },
  #"gcOSFatJet_pNetJ[0]":{
  #  "CATEGORICAL": False,
  #  "TRANSFORM": True,
  #  "LIMIT": [0,1],
  #  "LATEX": "gcOSFatJet pNetQCD"
  #},
  "NJets_forward": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [0,1],
    "LATEX": "N_{forward}"
  },
  "NJets_DeepFlavL": {
    "CATEGORICAL": True,
    "TRANSFORM": False,
    "LIMIT": [2,4],
    "LATEX": "N_b"
  },
}

selection = { # edit these accordingly
  "Bdecay_obs":{ "VALUE": [ 1, 4 ], "CONDITION": [ "==", "==" ] },
  "W_MT":{ "VALUE": [ 200 ], "CONDITION": [ "<=" ] },
}

regions = {
  "X": {
    "VARIABLE": "NJets_DeepFlavL",
    "INCLUSIVE": True,
    "MIN": None,
    "MAX": None,
    "SIGNAL": 2,
    "CONDITION": "<=",
    "A": ["==",3],
    "B": ["<=",2],
    "C": ["==",3],
    "D": ["<=",2],
    "X": [">=",4],
    "Y": [">=",4],
  },
  "Y": {
    "VARIABLE": "NJets_forward",
    "INCLUSIVE": True,
    "MIN": None,
    "MAX": None,
    "SIGNAL": 1,
    "CONDITION": ">=",
    "A":["==", 0],
    "B":["==", 0],
    "C":[">=", 1],
    "D":[">=", 1],
    "X":["==", 0],
    "Y":[">=", 1],
  }
}

params = {
  "MODEL": { # parameters for setting up the NAF model
    "NODES_COND": 8,
    "HIDDEN_COND": 1,
    "NODES_TRANS": 1,
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
  "Bprime_mass", "gcJet_ST",
  "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]",
  "btagWeights[17]"
  #"gcOSFatJet_pNetJ[0]"
]

for vName in variables:
  if vName not in branches: branches.append( str( vName ) )

samples_input = {
  "DATA": [ # NEED TO UPDATE FOR NEWER DATASET
    SingleElecRun2016APVB,
    SingleElecRun2016APVC,
    SingleElecRun2016APVD,
    SingleElecRun2016APVE,
    SingleElecRun2016APVF,
    SingleElecRun2016F,
    SingleElecRun2016G,
    SingleElecRun2016H,
    SingleElecRun2017B,
    SingleElecRun2017C,
    SingleElecRun2017D,
    SingleElecRun2017E,
    SingleElecRun2017F,
    SingleElecRun2018A,
    SingleElecRun2018B,
    SingleElecRun2018C,
    SingleElecRun2018D,
    SingleMuonRun2016APVB,
    SingleMuonRun2016APVC,
    SingleMuonRun2016APVD,
    SingleMuonRun2016APVE,
    SingleMuonRun2016APVF,
    SingleMuonRun2016F,
    SingleMuonRun2016G,
    SingleMuonRun2016H,
    SingleMuonRun2017B,
    SingleMuonRun2017C,
    SingleMuonRun2017D,
    SingleMuonRun2017E,
    SingleMuonRun2017F,
    SingleMuonRun2018A,
    SingleMuonRun2018B,
    SingleMuonRun2018C,
    SingleMuonRun2018D
  ], 
  "MAJOR MC": [
    QCDHT10002016APV,
    QCDHT10002016,
    QCDHT10002017,
    QCDHT10002018,
    QCDHT15002016APV,
    QCDHT15002016,
    QCDHT15002017,
    QCDHT15002018,
    QCDHT20002016APV,
    QCDHT20002016,
    QCDHT20002017,
    QCDHT20002018,
    #QCDHT2002016APV,
    #QCDHT2002016,
    #QCDHT2002017,
    #QCDHT2002018,
    QCDHT3002016APV,
    QCDHT3002016,
    QCDHT3002017,
    QCDHT3002018,
    QCDHT5002016APV,
    QCDHT5002016,
    QCDHT5002018,
    QCDHT7002016APV,
    QCDHT7002016,
    QCDHT7002017,
    QCDHT7002018,
    TTToSemiLeptonic2016APV,
    TTToSemiLeptonic2016,
    TTToSemiLeptonic2017,
    TTToSemiLeptonic2018,
    WJetsHT12002016APV,
    WJetsHT12002016,
    WJetsHT12002017,
    WJetsHT12002018,
    WJetsHT2002016APV,
    WJetsHT2002016,
    WJetsHT2002017,
    WJetsHT2002018,
    WJetsHT25002016APV,
    WJetsHT25002016,
    WJetsHT25002017,
    WJetsHT25002018,
    WJetsHT4002016APV,
    WJetsHT4002016,
    WJetsHT4002017,
    WJetsHT4002018,
    WJetsHT6002016APV,
    WJetsHT6002016,
    WJetsHT6002017,
    WJetsHT6002018,
    WJetsHT8002016APV,
    WJetsHT8002016,
    WJetsHT8002017,
    WJetsHT8002018
  ],
  "MINOR MC": [
    DYMHT12002016APV,
    DYMHT12002016,
    DYMHT12002017,
    DYMHT12002018,
    DYMHT2002016APV,
    DYMHT2002016,
    DYMHT2002017,
    DYMHT2002018,
    DYMHT25002016APV,
    DYMHT25002016,
    DYMHT25002017,
    DYMHT25002018,
    DYMHT4002016APV,
    DYMHT4002016,
    DYMHT4002017,
    DYMHT4002018,
    DYMHT6002016APV,
    DYMHT6002016,
    DYMHT6002017,
    DYMHT6002018,
    DYMHT8002016APV,
    DYMHT8002016,
    DYMHT8002017,
    DYMHT8002018,
    JetHTRun2016APVB,
    JetHTRun2016APVC,
    JetHTRun2016APVD,
    JetHTRun2016APVE,
    JetHTRun2016APVF,
    JetHTRun2016F,
    JetHTRun2016G,
    JetHTRun2016H,
    JetHTRun2017B,
    JetHTRun2017C,
    JetHTRun2017D,
    JetHTRun2017E,
    JetHTRun2017F,
    JetHTRun2018A,
    JetHTRun2018B,
    JetHTRun2018C,
    JetHTRun2018D,
    STs2016APV,
    STs2016,
    STs2017,
    STs2018,
    STt2016APV,
    STt2016,
    STt2017,
    STt2018,
    STtb2016APV,
    STtb2016,
    STtb2017,
    STtb2018,
    STtW2016APV,
    STtW2016,
    STtW2017,
    STtW2018,
    STtWb2016APV,
    STtWb2016,
    STtWb2017,
    STtWb2018,
    TTHB2016APV,
    TTHB2016,
    TTHB2017,
    TTHB2018,
    TTHnonB2016APV,
    TTHnonB2016,
    TTHnonB2017,
    TTHnonB2018,
    TTMT10002016APV,
    TTMT10002016,
    TTMT10002017,
    TTMT10002018,
    TTMT7002016APV,
    TTMT7002016,
    TTMT7002017,
    TTMT7002018,
    TTTo2L2Nu2016APV,
    TTTo2L2Nu2016,
    TTTo2L2Nu2017,
    TTTo2L2Nu2018,
    TTToHadronic2016APV,
    TTToHadronic2016,
    TTToHadronic2017,
    TTToHadronic2018,
    TTWl2016APV,
    TTWl2016,
    TTWl2017,
    TTWl2018,
    TTWq2016APV,
    TTWq2016,
    TTWq2017,
    TTWq2018,
    TTZM102016APV,
    TTZM102016,
    TTZM102017,
    TTZM102018,
    TTZM1to102016APV,
    TTZM1to102016,
    TTZM1to102017,
    TTZM1to102018,
    WW2016APV,
    WW2016,
    WW2017,
    WW2018,
    WZ2016APV,
    WZ2016,
    WZ2017,
    WZ2018,
    ZZ2016APV,
    ZZ2016,
    ZZ2017,
    ZZ2018
  ], # no need to add minor MC for now, because using ttbar MC as data
  "CLOSURE": []
}

samples_apply = {
  # UPDATE later
}
