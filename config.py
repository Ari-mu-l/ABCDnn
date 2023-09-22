import os
import numpy as np

data_path = os.path.join( os.getcwd() )
results_path = os.path.join( os.getcwd(), "Results" )
eosUserName = "jmanagan/"
#postfix = "hadd"

condorDir = "root://cmseos.fnal.gov//store/user/{}/".format( eosUserName ) 

sourceDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Sep2023_2018/".format( eosUserName ),
  "BRUX": "root://brux30.hep.brown.edu:1094//store/user/{}/".format( eosUserName )
}

targetDir = {
  "LPC": "root://cmseos.fnal.gov//store/user/{}/BtoTW_Sep2023_2018/".format( eosUserName ),
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
    "LATEX": "ST_{gcJets}"
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
      "RDF_SingleMuon_finalsel_0.root",
      "RDF_SingleMuon_finalsel_110.root",
      "RDF_SingleMuon_finalsel_12.root",
      "RDF_SingleMuon_finalsel_13.root",
      "RDF_SingleMuon_finalsel_132.root",
      "RDF_SingleMuon_finalsel_154.root",
      "RDF_SingleMuon_finalsel_176.root",
      "RDF_SingleMuon_finalsel_22.root",
      "RDF_SingleMuon_finalsel_23.root",
      "RDF_SingleMuon_finalsel_26.root",
      "RDF_SingleMuon_finalsel_28.root",
      "RDF_SingleMuon_finalsel_38.root",
      "RDF_SingleMuon_finalsel_44.root",
      "RDF_SingleMuon_finalsel_46.root",
      "RDF_SingleMuon_finalsel_48.root",
      "RDF_SingleMuon_finalsel_50.root",
      "RDF_SingleMuon_finalsel_51.root",
      "RDF_SingleMuon_finalsel_54.root",
      "RDF_SingleMuon_finalsel_59.root",
      "RDF_SingleMuon_finalsel_66.root",
      "RDF_SingleMuon_finalsel_69.root",
      "RDF_SingleMuon_finalsel_76.root",
      "RDF_SingleMuon_finalsel_88.root",
      "RDF_EGamma_finalsel_0.root",
      "RDF_EGamma_finalsel_128.root",
      "RDF_EGamma_finalsel_160.root",
      "RDF_EGamma_finalsel_192.root",
      "RDF_EGamma_finalsel_224.root",
      "RDF_EGamma_finalsel_25.root",
      "RDF_EGamma_finalsel_32.root",
      "RDF_EGamma_finalsel_50.root",
      "RDF_EGamma_finalsel_55.root",
      "RDF_EGamma_finalsel_58.root",
      "RDF_EGamma_finalsel_62.root",
      "RDF_EGamma_finalsel_64.root",
      "RDF_EGamma_finalsel_96.root",
    ], # FIXME later

    "MAJOR MC": [
      "RDF_BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_7.root",
      "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_23.root",
      "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_77.root",
      "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_24.root",
      "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
      "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_34.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_0.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_114.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_133.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_142.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_143.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_144.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_146.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_152.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_171.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_19.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_190.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_209.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_228.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_247.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_266.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_285.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_304.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_323.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_342.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_361.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_365.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_370.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_38.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_380.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_57.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_76.root",
      "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_95.root",
      "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_12.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_14.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_18.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_21.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_25.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_5.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_51.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_76.root",
      "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_8.root",
      "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
      "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_5.root",
    ],
    "MINOR MC": [], # no need to add minor MC for now, because using ttbar MC as data
    "CLOSURE": []
  },
  # add other years here
}

samples_apply = {
  "2018UL": [
    "RDF_BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_7.root",
    "RDF_QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_23.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_77.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_24.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_0.root",
    "RDF_QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8_finalsel_34.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_0.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_114.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_133.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_142.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_143.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_144.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_146.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_152.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_171.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_19.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_190.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_209.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_228.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_247.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_266.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_285.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_304.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_323.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_342.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_361.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_365.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_370.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_38.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_380.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_57.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_76.root",
    "RDF_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_finalsel_95.root",
    "RDF_WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_12.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_14.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_18.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_21.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_25.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_5.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_51.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_76.root",
    "RDF_WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_8.root",
    "RDF_WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_0.root",
    "RDF_WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_finalsel_5.root",
  ],
}
