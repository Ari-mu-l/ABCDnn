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
parser.add_argument( "-s", "--source", required = True )
parser.add_argument( "-t", "--target", required = True )
parser.add_argument( "-b", "--minor" , required = True )
parser.add_argument( "-m", "--tag"   , required = True )

args = parser.parse_args()

folder = config.params[ "MODEL" ][ "SAVEDIR" ]
folder_contents = os.listdir( folder )

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
lowerlimit = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ]
upperlimit = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]

print( ">> Found {} variables: ".format( len( variables ) ) )
for i, variable in enumerate( variables ):
  print( "  + {}: [{},{}], Categorical = {}".format( variable, lowerlimit[i], upperlimit[i], categorical[i] ) )

inputs_src = sTree.arrays( variables, library="pd" )
inputs_tgt = sTree.arrays( variables, library="pd" )
inputs_mnr = mTree.arrays( variables + [ "xsecWeight" ], library="pd" )

inputs_src_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
inputs_tgt_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
inputs_mnr_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }

X_MIN = inputs_src[ variables[-1] ].min()
X_MAX = inputs_src[ variables[-1] ].max()
Y_MIN = inputs_src[ variables[-2] ].min()
Y_MAX = inputs_src[ variables[-2] ].max()

x_region = np.linspace( X_MIN, X_MAX, X_MAX - X_MIN + 1 )
y_region = np.linspace( Y_MIN, Y_MAX, Y_MAX - Y_MIN + 1 )

for variable in variables:
  inputs_tgt[variable] = inputs_tgt[variable].clip(upper = config.variables[variable]["LIMIT"][1])
  inputs_mnr[variable] = inputs_mnr[variable].clip(upper = config.variables[variable]["LIMIT"][1])

for region in [ "X", "Y", "A", "B", "C", "D" ]:
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
predictions = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }

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
  NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] )[::2] ) )
  predictions[ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2] 
del NAF

def makeHists_plot():
  predict_array = predictions["D"]
  
  hist_predict = ROOT.TH1D(f'Bprime_mass_ABCDnn', "Bprime_mass_ABCDnn", 51, 0, 2500)
  hist_predict_val = ROOT.TH1D(f'Bprime_mass_ABCDnn_val_{region}', "Bprime_mass_ABCDnn", 51, 0, 2500)

  for i in range(len(predict_array)):
    hist_predict.Fill(predict_array[i][0])
    if predict_array[i][1]<850:
      hist_predict_val.Fill(predict_array[i][0])

  hist_predict.Write()
  hist_predict_val.Write()
  

def makeHists_fit(region):
  inputs_tgt_array = inputs_tgt_region[region].to_numpy(dtype='d')[:,0]
  inputs_mnr_array = inputs_mnr_region[region].to_numpy(dtype='d')[:,0]
  prediction_array = predictions[region][:,0]

  hist_tgt = ROOT.TH1D(f'Bprime_mass_tgt_{region}', "Bprime_mass_ABCDnn", 51, 0, 2500)
  hist_mnr = ROOT.TH1D(f'Bprime_mass_mnr_{region}', "Bprime_mass_ABCDnn", 51, 0, 2500)
  hist_pre = ROOT.TH1D(f'Bprime_mass_pre_{region}', "Bprime_mass_ABCDnn", 51, 0, 2500)

  for i in range(len(inputs_tgt_array)):
    hist_tgt.Fill(inputs_tgt_array[i])

  for i in range(len(inputs_mnr_array)):
    hist_mnr.Fill(inputs_mnr_array[i])

  for i in range(len(prediction_array)):
    hist_pre.Fill(prediction_array[i])
    
  hist_tgt.Write()
  hist_mnr.Write()

histFile = ROOT.TFile.Open("hists_ABCDnn.root", "recreate")

makeHists_fit("A")
makeHists_fit("B")
makeHists_fit("C")
makeHists_fit("D")
makeHists_fit("X")
makeHists_fit("Y")

makeHists_plot()

histFile.Close()
