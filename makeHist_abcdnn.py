# python3 makeHist_abcdnn.py -s rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctMajor_all_mc_p100.root -b rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctMinor_all_mc_p100.root -t rootFiles_logBpMlogST_Oct2024Run2_pNetWeight/Case14/OctData_all_data_p100.root -f 1.0 -m logBpMlogST_mmd1_case14_random104

################
# NOTE: Temporarily commented out tgt and mnr related lines, becuase exceed memory limit with pNet applied
################

import numpy as np
import os, tqdm
import abcdnn
import uproot, ROOT
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import samples
from utils import *

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True  )
parser.add_argument( "-t", "--target", required = True  )
parser.add_argument( "-b", "--minor" , required = True  )
parser.add_argument( "-m", "--tag"   , required = True  )
#parser.add_argument( "-c", "--case"  , required = False )

args = parser.parse_args()

# histogram settings
bin_lo = 400 #0
bin_hi = 2500
Nbins  = 420 # 42
validationCut = 850

log = True # set to False if want BpM instead of log(BpM)
if log:
  validationCut = np.log(validationCut)
  bin_lo = np.log(bin_lo)
  bin_hi = np.log(bin_hi)

folder = config.params[ "MODEL" ][ "SAVEDIR" ]
folder_contents = os.listdir( folder )

isTest = False
if isTest:
  testDir = args.tag
  if not os.path.exists(testDir):
    os.makedirs(testDir+'/')
else: testDir = ''

print( ">> Reading in {}.json for hyper parameters...".format( args.tag ) )
with open( os.path.join( folder, args.tag + ".json" ), "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )

print( ">> Load the data" )
sFile = uproot.open( args.source )
tFile = uproot.open( args.target )
mFile = uproot.open( args.minor )
sTree = sFile[ "Events" ]
tTree = tFile[ "Events" ]
mTree = mFile[ "Events" ]

variables = [ str( key ) for key in sorted( config.variables.keys() ) if config.variables[key]["TRANSFORM"] ]
if config.regions["Y"]["VARIABLE"] in config.variables and config.regions["X"]["VARIABLE"] in config.variables:
  variables.append( config.regions["Y"]["VARIABLE"] )
  variables.append( config.regions["X"]["VARIABLE"] )
else:
  sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )

categorical = [ config.variables[ vName ][ "CATEGORICAL" ] for vName in variables ]
lowerlimit  = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ]
upperlimit  = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]

print( ">> Found {} variables: ".format( len( variables ) ) )
for i, variable in enumerate( variables ):
  print( "  + {}: [{},{}], Categorical = {}".format( variable, lowerlimit[i], upperlimit[i], categorical[i] ) )

inputs_src = sTree.arrays( variables, library="pd" )
inputs_tgt = tTree.arrays( variables, library="pd" )
inputs_mnr = mTree.arrays( variables + ["xsecWeight"], library="pd" )

Bdecay_src = sTree.arrays( ["Bdecay_obs", "pNetTtagWeight", "pNetTtagWeightUp", "pNetTtagWeightDn", "pNetWtagWeight", "pNetWtagWeightUp", "pNetWtagWeightDn"], library="pd" )
Bdecay_tgt = tTree.arrays( ["Bdecay_obs"], library="pd" )
Bdecay_mnr = mTree.arrays( ["Bdecay_obs", "pNetTtagWeight", "pNetWtagWeight"], library="pd" )
##Bdecay_tgt = tTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_tgt["Bprime_mass"]>400]
##Bdecay_mnr = mTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_mnr["Bprime_mass"]>400]

##inputs_tgt = inputs_tgt[inputs_tgt["Bprime_mass"]>400] # take only BpM>400. pred cut made later.
##inputs_mnr = inputs_mnr[inputs_mnr["Bprime_mass"]>400]

inputs_src_region = { region: None for region in [ "A", "B", "C", "D" ] }
inputs_tgt_region = { region: None for region in [ "A", "B", "C", "D" ] }
inputs_mnr_region = { region: None for region in [ "A", "B", "C", "D" ] }

Bdecay_src_region = { region: None for region in [ "A", "B", "C", "D" ] }
Bdecay_tgt_region = { region: None for region in [ "A", "B", "C", "D" ] }
Bdecay_mnr_region = { region: None for region in [ "A", "B", "C", "D" ] }

X_MIN = inputs_src[ variables[-1] ].min()
X_MAX = inputs_src[ variables[-1] ].max()
Y_MIN = inputs_src[ variables[-2] ].min()
Y_MAX = inputs_src[ variables[-2] ].max()

x_region = np.linspace( X_MIN, X_MAX, X_MAX - X_MIN + 1 )
y_region = np.linspace( Y_MIN, Y_MAX, Y_MAX - Y_MIN + 1 )

for variable in variables:
  inputs_tgt[variable] = inputs_tgt[variable].clip(upper = config.variables[variable]["LIMIT"][1])
  inputs_mnr[variable] = inputs_mnr[variable].clip(upper = config.variables[variable]["LIMIT"][1])

for region in [ "A", "B", "C", "D" ]:
  if config.regions["X"][region][0] == ">=":
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
  elif config.regions["X"][region][0] == "<=":
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
  else:
    select_src = inputs_src[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]
    select_tgt = inputs_tgt[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]
    select_mnr = inputs_mnr[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]

  if config.regions["Y"][region][0] == ">=":
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
  elif config.regions["Y"][region][0] == "<=":
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
  else:
    select_src &= inputs_src[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]
    select_tgt &= inputs_tgt[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]
    select_mnr &= inputs_mnr[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]

  inputs_src_region[region] = inputs_src.loc[select_src]
  inputs_tgt_region[region] = inputs_tgt.loc[select_tgt]
  inputs_mnr_region[region] = inputs_mnr.loc[select_mnr]

  Bdecay_src_region[region] = Bdecay_src.loc[select_src]
  Bdecay_tgt_region[region] = Bdecay_tgt.loc[select_tgt]
  Bdecay_mnr_region[region] = Bdecay_mnr.loc[select_mnr]
  
print( ">> Encoding and normalizing source inputs" )
inputs_enc_region = {}
encoder = {}
inputs_nrm_region = {}
inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
inputsigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )

for region in inputs_src_region:
  encoder[region] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
  inputs_enc_region[ region ] = encoder[region].encode( inputs_src_region[ region ].to_numpy( dtype = np.float32 ) )
  inputs_nrm_region[ region ] = ( inputs_enc_region[ region ] - inputmeans ) / inputsigmas
  inputs_src_region[ region ][ variables[0] ] = inputs_src_region[ region ][ variables[0] ].clip(upper=config.variables[variables[0]]["LIMIT"][1])
  inputs_src_region[ region ][ variables[1] ] = inputs_src_region[ region ][ variables[1] ].clip(upper=config.variables[variables[1]]["LIMIT"][1])
  

#get predictions
predictions = { region: [] for region in [ "A", "B", "C", "D" ] }

NAF = abcdnn.NAF( 
    inputdim = params["INPUTDIM"],
    conddim = params["CONDDIM"],
    activation = params["ACTIVATION"], 
    regularizer = params["REGULARIZER"],
    initializer = params["INITIALIZER"],
    nodes_cond = params["NODES_COND"],
    hidden_cond = params["HIDDEN_COND"],
    nodes_trans = params["NODES_TRANS"],
    depth = params["DEPTH"],
    permute = bool( params["PERMUTE"] )
  )
NAF.load_weights( os.path.join( folder, args.tag ) )

for region in tqdm.tqdm(predictions):
  NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] ) ) )
  predictions[ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2]
  ##Bdecay_src_region[region] = Bdecay_src_region[region][predictions[ region ][:,0]>400]
  ##predictions[ region ] = predictions[region][predictions[ region ][:,0]>400] # take only BpM>400
del NAF


def makeHists_plot(case, inputs_tgt_array, inputs_mnr_array, weight_mnr_array, predict_array, pNet_mnr_array, pNet_src_array, pNetUp_src_array, pNetDn_src_array):
  ##predict_array = predictions["D"]
  ##inputs_tgt_array = inputs_tgt_region["D"].to_numpy(dtype='d')
  ##inputs_mnr_array = inputs_mnr_region["D"].to_numpy(dtype='d')
  ##weight_mnr_array = inputs_mnr_region["D"]["xsecWeight"].to_numpy(dtype='d')

  hist_predict_val        = ROOT.TH1D(f'Bprime_mass_pre_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  hist_predict_val_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_V_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  hist_predict_val_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_V_pNetDn', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  hist_data_val           = ROOT.TH1D(f'Bprime_mass_dat_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  hist_minor_val          = ROOT.TH1D(f'Bprime_mass_mnr_V', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)

  for i in range(len(inputs_tgt_array)):
   if inputs_tgt_array[i][1]<validationCut and inputs_tgt_array[i][1]>bin_lo:
     hist_data_val.Fill(inputs_tgt_array[i][0])

  if case=="case1" or case=="case3":
    print(f'{case}: applying pNet weights')
    # fill prediction hist
    for i in range(len(predict_array)):
      if predict_array[i][1]<validationCut and predict_array[i][1]>bin_lo:
        hist_predict_val.Fill(predict_array[i][0], pNet_src_array[i]) # pNetSF
        hist_predict_val_pNetUp.Fill(predict_array[i][0], pNetUp_src_array[i]) # pNetSF_Up
        hist_predict_val_pNetDn.Fill(predict_array[i][0], pNetDn_src_array[i]) # pNetSF_Down
    # fill minor hist
    for i in range(len(inputs_mnr_array)):
     if inputs_mnr_array[i][1]<validationCut and inputs_mnr_array[i][1]>bin_lo:
       hist_minor_val.Fill(inputs_mnr_array[i][0], weight_mnr_array[i] * pNet_mnr_array[i])
  else:
    # untagged cases do not need pNetSFs
    for i in range(len(predict_array)):
      if predict_array[i][1]<validationCut and predict_array[i][1]>bin_lo:
        hist_predict_val.Fill(predict_array[i][0])
    for i in range(len(inputs_mnr_array)):
     if inputs_mnr_array[i][1]<validationCut and inputs_mnr_array[i][1]>bin_lo:
       hist_minor_val.Fill(inputs_mnr_array[i][0], weight_mnr_array[i])
  
  hist_data_val.Write()
  hist_minor_val.Write()
  hist_predict_val.Write()
  if case=="case1" or case=="case3":
    hist_predict_val_pNetUp.Write()
    hist_predict_val_pNetDn.Write()
  

def makeHists_fit(region, case):
  if case=="case1":
    inputs_tgt_array = inputs_tgt_region[region][ Bdecay_tgt_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')[:,:2]
    inputs_mnr_array = inputs_mnr_region[region][ Bdecay_mnr_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')[:,:2]
    weight_mnr_array = inputs_mnr_region[region]["xsecWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')
    pNet_mnr_array = Bdecay_mnr_region[region]["pNetTtagWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d') # case1: Ttag
    pNet_src_array = Bdecay_src_region[region]["pNetTtagWeight"][ Bdecay_src_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')
    pNetUp_src_array = Bdecay_src_region[region]["pNetTtagWeightUp"][ Bdecay_src_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')
    pNetDn_src_array = Bdecay_src_region[region]["pNetTtagWeightDn"][ Bdecay_src_region[region]["Bdecay_obs"]==1 ].to_numpy(dtype='d')
    prediction_array = predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==1 ]
    ##prediction_array = np.clip(predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==1 ], 0, 2500)
  elif case=="case2":
    inputs_tgt_array = inputs_tgt_region[region][ Bdecay_tgt_region[region]["Bdecay_obs"]==2 ].to_numpy(dtype='d')[:,:2]
    inputs_mnr_array = inputs_mnr_region[region][ Bdecay_mnr_region[region]["Bdecay_obs"]==2 ].to_numpy(dtype='d')[:,:2]
    weight_mnr_array = inputs_mnr_region[region]["xsecWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==2 ].to_numpy(dtype='d')
    pNet_mnr_array   = np.zeros(1) # dummies: untagged cases do not need pNetSF
    pNet_src_array   = np.zeros(1)
    pNetUp_src_array = np.zeros(1)
    pNetDn_src_array = np.zeros(1)
    prediction_array = predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==2 ]
    ##prediction_array = np.clip(predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==2 ], 0, 2500)
  elif case=="case3":
    inputs_tgt_array = inputs_tgt_region[region][ Bdecay_tgt_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')[:,:2]
    inputs_mnr_array = inputs_mnr_region[region][ Bdecay_mnr_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')[:,:2]
    weight_mnr_array = inputs_mnr_region[region]["xsecWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')
    pNet_mnr_array = Bdecay_mnr_region[region]["pNetWtagWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')
    pNet_src_array = Bdecay_src_region[region]["pNetWtagWeight"][ Bdecay_src_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')
    pNetUp_src_array = Bdecay_src_region[region]["pNetWtagWeightUp"][ Bdecay_src_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')
    pNetDn_src_array = Bdecay_src_region[region]["pNetWtagWeightDn"][ Bdecay_src_region[region]["Bdecay_obs"]==3 ].to_numpy(dtype='d')
    prediction_array = predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==3 ]
    ##prediction_array = np.clip(predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==3 ], 0, 2500)
  elif case=="case4":
    inputs_tgt_array = inputs_tgt_region[region][ Bdecay_tgt_region[region]["Bdecay_obs"]==4 ].to_numpy(dtype='d')[:,:2]
    inputs_mnr_array = inputs_mnr_region[region][ Bdecay_mnr_region[region]["Bdecay_obs"]==4 ].to_numpy(dtype='d')[:,:2]
    weight_mnr_array = inputs_mnr_region[region]["xsecWeight"][ Bdecay_mnr_region[region]["Bdecay_obs"]==4 ].to_numpy(dtype='d')
    pNet_mnr_array   = np.zeros(1) # dummies
    pNet_src_array   = np.zeros(1)
    pNetUp_src_array = np.zeros(1)
    pNetDn_src_array = np.zeros(1)
    prediction_array = predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==4 ]
    ##prediction_array = np.clip(predictions[region][ Bdecay_src_region[region]["Bdecay_obs"]==4 ], 0, 2500)
  else:
    inputs_tgt_array = inputs_tgt_region[region].to_numpy(dtype='d')[:,:2]
    inputs_mnr_array = inputs_mnr_region[region].to_numpy(dtype='d')[:,:2]
    weight_mnr_array = inputs_mnr_region[region]["xsecWeight"].to_numpy(dtype='d')
    pNet_mnr_array   = np.zeros(1) # dummies: pNet tag should be implemented for case14 and 23, but we do not need the combined cases for now.
    pNet_src_array   = np.zeros(1)
    pNetUp_src_array = np.zeros(1)
    pNetDn_src_array = np.zeros(1)
    prediction_array = predictions[region]
    ##prediction_array = np.clip(predictions[region], 0, 2500)

  if not log:
    inputs_tgt_array = np.exp(inputs_tgt_array)
    inputs_mnr_array = np.exp(inputs_mnr_array)
    prediction_array = np.exp(prediction_array)

  inputs_tgt_array = np.clip(inputs_tgt_array, 0, 2500)
  inputs_mnr_array = np.clip(inputs_mnr_array, 0, 2500)
  prediction_array = np.clip(prediction_array, 0, 2500)

  hist_tgt   = ROOT.TH1D(f'Bprime_mass_dat_{region}', "Bprime_mass"       , Nbins, bin_lo, bin_hi)
  hist_mnr   = ROOT.TH1D(f'Bprime_mass_mnr_{region}', "Bprime_mass"       , Nbins, bin_lo, bin_hi)
  hist_pre   = ROOT.TH1D(f'Bprime_mass_pre_{region}', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
  if case=="case1" or case=="case3":
    hist_pre_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_{region}_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    hist_pre_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_{region}_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)

  # fill target hist. Same for all cases. No pNetSF needed
  for i in range(len(inputs_tgt_array)):
    if inputs_tgt_array[i][0]>bin_lo:
      hist_tgt.Fill(inputs_tgt_array[i][0])
      
  if case=="case1" or case=="case3":
    print(f'{case}: applying pNet weights')
    # fill minor hist with pNet. Shift not needed. Taken care of by analyze_RDF.py
    for i in range(len(inputs_mnr_array)):
      if inputs_mnr_array[i][0]>bin_lo:
        hist_mnr.Fill(inputs_mnr_array[i][0], weight_mnr_array[i] * pNet_mnr_array[i])
    # fill prediction hist with pNet and pNetShifts
    for i in range(len(prediction_array)):
      if prediction_array[i][0]>bin_lo:
        hist_pre.Fill(prediction_array[i][0], pNet_src_array[i])
        hist_pre_pNetUp.Fill(prediction_array[i][0], pNetUp_src_array[i])
        hist_pre_pNetDn.Fill(prediction_array[i][0], pNetDn_src_array[i])
  else:
    # fill minor hist
    for i in range(len(inputs_mnr_array)):
      if inputs_mnr_array[i][0]>bin_lo:
        hist_mnr.Fill(inputs_mnr_array[i][0], weight_mnr_array[i])
    # fill prediction hist
    for i in range(len(prediction_array)):
      if prediction_array[i][0]>bin_lo:
        hist_pre.Fill(prediction_array[i][0])
  
  hist_tgt.Write()
  hist_mnr.Write()
  hist_pre.Write()
  if case=="case1" or case=="case3":
    hist_pre_pNetUp.Write()
    hist_pre_pNetDn.Write()

  return inputs_tgt_array, inputs_mnr_array , weight_mnr_array, prediction_array, pNet_mnr_array, pNet_src_array, pNetUp_src_array, pNetDn_src_array

if 'case14' in args.tag:
  ##case_list = ["case14", "case1", "case4"]
  case_list = ["case1", "case4"]
elif 'case23' in args.tag:
  ##case_list = ["case23", "case2", "case3"]
  case_list = ["case2", "case3"]
  
for case in case_list:
  if log:
    histFile = ROOT.TFile.Open(f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_log_pNet.root', "recreate")
  else:
    histFile = ROOT.TFile.Open(f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_pNet.root', "recreate")

  makeHists_fit("A", case)
  makeHists_fit("B", case)
  makeHists_fit("C", case)
  inputs_tgt_array, inputs_mnr_array , weight_mnr_array, prediction_array, pNet_mnr_array, pNet_src_array, pNetUp_src_array, pNetDn_src_array = makeHists_fit("D", case)
  #makeHists_fit("X", case)
  #makeHists_fit("Y", case)

  makeHists_plot(case, inputs_tgt_array, inputs_mnr_array, weight_mnr_array, prediction_array, pNet_mnr_array, pNet_src_array, pNetUp_src_array, pNetDn_src_array)
  histFile.Close()
