import os
import numpy as np
from samples import *

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "xshen"
#postfix = "hadd"

condorDir = "root://cmseos.fnal.gov//store/user/xshen/"

sourceDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/jmanagan/BtoTW_Oct2023_fullRun2/",
  #"BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

targetDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Oct2023_fullRun2/".format( eosUserName ),
  #"BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

#sampleDir = {
  #"root://cmseos.fnal.gov//store/user/{}/BtoTW_Oct2023_fullRun2/ABCDnn_Jan2024/".format( eosUserName )
  #year: "FWLJMET106XUL_singleLep{}UL_RunIISummer20_{}_step3/nominal/".format( year, postfix ) for year in [ "2016APV", "2016", "2017", "2018" ]
#}
sampleDir = "root://cmseos.fnal.gov//store/user/xshen/BtoTW_Oct2023_fullRun2/ABCDnn_Jan2024/".format( eosUserName )

variables = {
  "Bprime_mass": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0., 2500.], # was 5000
    "LATEX": "M_{reco}"
  },
  "gcJet_ST": {
    "CATEGORICAL": False,
    "TRANSFORM": True,
    "LIMIT": [0.,1500.], # was 6000
    "LATEX": "ST_{gcJet}"
  },
  #"gcLeadingOSFatJet_pNetJ":{
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
  "case1" : "Bdecay_obs==1",
  "case2" : "Bdecay_obs==2",
  "case3" : "Bdecay_obs==3",
  "case4" : "Bdecay_obs==4",
  "case14": "(Bdecay_obs==1) || (Bdecay_obs==4)",
  "case23": "(Bdecay_obs==2) || (Bdecay_obs==3)",
  "W_MT"  : "W_MT<=200",
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
    "NODES_COND": 4,
    "HIDDEN_COND": 5,
    "NODES_TRANS": 9,
    "LRATE": 0.01,
    "DECAY": 0.1,
    "GAP": 200,
    "DEPTH": 2,
    "REGULARIZER": "L1", # DROPOUT, BATCHNORM, ALL, NONE
    "INITIALIZER": "RandomNormal", # he_normal, RandomNormal
    "ACTIVATION": "swish", # softplus, relu, swish
    "BETA1": 0.999,
    "BETA2": 0.999,
    "MMD SIGMAS": [0.041203371333727076, 0.6334576112365489, 0.23175814103352135],
    "MMD WEIGHTS": None,
    "MINIBATCH": 512,
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
  "2016APV": {
    "DATA": [
      SingleElecRun2016APVB,
      SingleElecRun2016APVC,
      SingleElecRun2016APVD,
      SingleElecRun2016APVE,
      SingleElecRun2016APVF,
      SingleMuonRun2016APVB,
      SingleMuonRun2016APVC,
      SingleMuonRun2016APVD,
      SingleMuonRun2016APVE,
      SingleMuonRun2016APVF
    ],
    "MAJOR MC": [ # QCD200 excluded. no event.
      QCDHT10002016APV,
      QCDHT15002016APV,
      QCDHT20002016APV,
      QCDHT3002016APV,
      QCDHT5002016APV,
      QCDHT7002016APV,
      TTMT10002016APV,
      TTMT7002016APV,
      TTTo2L2Nu2016APV,
      TTToHadronic2016APV,
      TTToSemiLeptonic2016APV,
      WJetsHT12002016APV,
      WJetsHT2002016APV,
      WJetsHT25002016APV,
      WJetsHT4002016APV,
      WJetsHT6002016APV,
      WJetsHT8002016APV,
      STs2016APV,
      STt2016APV,
      STtb2016APV,
      STtW2016APV,
      STtWb2016APV,
    ],
    "MINOR MC": [
      DYMHT12002016APV,
      DYMHT2002016APV,
      DYMHT25002016APV,
      DYMHT4002016APV,
      DYMHT6002016APV,
      DYMHT8002016APV,
      TTHB2016APV,
      TTHnonB2016APV,
      TTWl2016APV,
      TTWq2016APV,
      TTZM102016APV,
      TTZM1to102016APV,
      WW2016APV,
      WZ2016APV,
      ZZ2016APV
    ],
    "CLOSURE": [
      
    ]
  },
  "2016": {
    "DATA": [
      SingleElecRun2016F,
      SingleElecRun2016G,
      SingleElecRun2016H,
      SingleMuonRun2016F,
      SingleMuonRun2016G,
      SingleMuonRun2016H
    ],
    "MAJOR MC":[
      QCDHT10002016,
      QCDHT15002016,  
      QCDHT20002016,
      QCDHT3002016,
      QCDHT5002016,
      QCDHT7002016,
      TTMT10002016,
      TTMT7002016,
      TTTo2L2Nu2016,
      TTToHadronic2016,
      TTToSemiLeptonic2016,
      WJetsHT12002016,
      WJetsHT2002016,
      WJetsHT25002016,
      WJetsHT4002016,
      WJetsHT6002016,
      WJetsHT8002016,
      STs2016,
      STt2016,
      STtb2016,
      STtW2016,
      STtWb2016
    ],
    "MINOR MC": [
      DYMHT12002016,
      DYMHT2002016,
      DYMHT25002016,
      DYMHT4002016,
      DYMHT6002016,
      DYMHT8002016,
      TTHB2016,
      TTHnonB2016,
      TTWl2016,
      TTWq2016,
      TTZM102016,
      TTZM1to102016,
      WW2016,
      WZ2016,
      ZZ2016
    ],
    "CLOSURE": []
  },
  "2017": {
    "DATA": [
      SingleElecRun2017B,
      SingleElecRun2017C,
      SingleElecRun2017D,
      SingleElecRun2017E,
      SingleElecRun2017F,
      SingleMuonRun2017B,
      SingleMuonRun2017C,
      SingleMuonRun2017D,
      SingleMuonRun2017E,
      SingleMuonRun2017F
    ],
    "MAJOR MC":[
      QCDHT10002017,
      QCDHT15002017,
      QCDHT20002017,
      QCDHT3002017,
      QCDHT7002017,
      TTMT10002017,
      TTMT7002017,
      TTTo2L2Nu2017,
      TTToHadronic2017,
      TTToSemiLeptonic2017,
      WJetsHT12002017,
      WJetsHT2002017,
      WJetsHT25002017,
      WJetsHT4002017,
      WJetsHT6002017,
      WJetsHT8002017,
      STs2017,
      STt2017,
      STtb2017,
      STtW2017,
      STtWb2017
    ],
    "MINOR MC": [
      DYMHT12002017,
      DYMHT2002017,
      DYMHT25002017,
      DYMHT4002017,
      DYMHT6002017,
      DYMHT8002017,
      TTHB2017,
      TTHnonB2017,
      TTWl2017,
      TTWq2017,
      TTZM102017,
      TTZM1to102017,
      WW2017,
      WZ2017,
      ZZ2017
    ],
    "CLOSURE": []
  },
  "2018": {
    "DATA": [
      SingleElecRun2018A,
      SingleElecRun2018B,
      SingleElecRun2018C,
      SingleElecRun2018D,
      SingleMuonRun2018A,
      SingleMuonRun2018B,
      SingleMuonRun2018C,
      SingleMuonRun2018D
    ],
    "MAJOR MC":[
      QCDHT10002018,
      QCDHT15002018,
      #QCDHT2002018,
      QCDHT20002018,
      QCDHT3002018,
      QCDHT5002018,
      QCDHT7002018,
      TTMT10002018,
      TTMT7002018,
      TTTo2L2Nu2018,
      TTToHadronic2018,
      TTToSemiLeptonic2018,
      WJetsHT12002018,
      WJetsHT2002018,
      WJetsHT25002018,
      WJetsHT4002018,
      WJetsHT6002018,
      WJetsHT8002018,
      STs2018,
      STt2018,
      STtb2018,
      STtW2018,
      STtWb2018
    ],
    "MINOR MC": [
      DYMHT12002018,
      DYMHT2002018,
      DYMHT25002018,
      DYMHT4002018,
      DYMHT6002018,
      DYMHT8002018,
      TTHB2018,
      TTHnonB2018,
      TTWl2018,
      TTWq2018,
      TTZM102018,
      TTZM1to102018,
      WW2018,
      WZ2018,
      ZZ2018
    ],
    "CLOSURE": []
  }
}

samples_apply = { # TODO: check list
  "2016":[
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_11.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_16.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_105.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_126.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_21.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_42.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_63.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016_84.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_14.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_28.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_35.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_42.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016_55.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2016_15.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2016_10.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016_12.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2016_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2016_21.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2016_42.root",
  ],
  "2016APV":[
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_19.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_28.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_5.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_23.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_45.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_67.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_17.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_26.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2016APV_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_108.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_18.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_36.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_54.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_72.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2016APV_90.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_17.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_14.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_27.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_7.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_15.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_11.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_2.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_3.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_4.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2016APV_22.root",
  ],
  "2017":[
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_21.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_31.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_12.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_18.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_6.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_11.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_21.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_20.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_0.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_11.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_21.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2017_31.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_108.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_126.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_144.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_162.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_18.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_180.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_198.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_207.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_212.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_214.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_216.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_234.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_252.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_270.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_288.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_36.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_54.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_72.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2017_90.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017_15.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017_22.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017_29.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2017_8.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017_15.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017_30.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017_4.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2017_8.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2017_14.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2017_22.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2017_43.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2017_15.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2017_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2017_25.root",
  ],
  "2018":[
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_13.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_26.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_39.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_12.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_23.root",
    #"RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_6.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_116.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_135.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_39.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_58.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_77.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_97.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_24.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_0.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_13.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_17.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_26.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_34.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_43.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_51.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_60.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_2018_9.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_102.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_119.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_136.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_145.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_149.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_151.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_153.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_17.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_170.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_187.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_204.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_221.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_238.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_255.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_272.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_289.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_306.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_323.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_34.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_340.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_357.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_374.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_51.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_68.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_2018_85.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_12.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_2.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_23.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_4.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_46.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_6.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_2018_7.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8ext1_2018_0.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8ext1_2018_12.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_12.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_23.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_35.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_46.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_69.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_2018_92.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_2018_46.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_10.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_20.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_25.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_30.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_2018_40.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8ext1_2018_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_15.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_16.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_30.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_32.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_2018_8.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8ext1_2018_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_10.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_15.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_29.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_5.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_2018_8.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8ext1_2018_0.root",
  ],
}
